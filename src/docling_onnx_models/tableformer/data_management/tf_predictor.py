#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ...common.base_predictor import BaseONNXPredictor
from ...common.utils import resize_image_with_aspect_ratio

_log = logging.getLogger(__name__)


class TFPredictor(BaseONNXPredictor):
    """
    ONNX-based table structure predictor.

    This class provides the same interface as the original TFPredictor
    from docling-ibm-models but uses ONNX Runtime for inference.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
        num_threads: int = 4,
        onnx_model_name: str = "model.onnx",
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the ONNX Table Structure Predictor.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing model settings and paths.
        device : str, optional
            Device to run inference on ('cpu' or 'cuda'), by default "cpu".
        num_threads : int, optional
            Number of threads for CPU inference, by default 4.
        onnx_model_name : str, optional
            Name of the ONNX model file, by default "model.onnx".
        providers : List[str], optional
            ONNX Runtime execution providers, by default None.

        Raises
        ------
        FileNotFoundError
            When required model files are missing.
        """
        self._config = config

        # Get model path from config
        model_dir = config["model"]["save_dir"]
        model_path = os.path.join(model_dir, onnx_model_name)

        # Initialize base ONNX predictor
        super().__init__(model_path, device, num_threads, providers)

        # Load additional configuration
        self._load_table_config()

        _log.debug("TFPredictor initialized with ONNX backend")

    def _load_table_config(self):
        """Load table-specific configuration."""
        # Extract table processing parameters from config
        self.max_steps = self._config.get("predict", {}).get("max_steps", 512)
        self.beam_size = self._config.get("predict", {}).get("beam_size", 1)

        # Set input preprocessing parameters
        self.input_size = self._config.get("model", {}).get("input_size", [640, 640])
        if isinstance(self.input_size, int):
            self.input_size = [self.input_size, self.input_size]

        _log.debug(
            f"Table config loaded: max_steps={self.max_steps}, beam_size={self.beam_size}"
        )

    def resize_img(
        self, image: np.ndarray, height: int = 1024
    ) -> tuple[np.ndarray, float]:
        """
        Resize image while maintaining aspect ratio.

        Parameters
        ----------
        image : np.ndarray
            Input image array.
        height : int, optional
            Target height, by default 1024.

        Returns
        -------
        tuple[np.ndarray, float]
            Tuple of (resized_image, scale_factor).
        """
        return resize_image_with_aspect_ratio(image, height, cv2.INTER_LINEAR)

    def _preprocess_table_image(
        self, table_image: np.ndarray, page_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Preprocess table image for model input.

        Parameters
        ----------
        table_image : np.ndarray
            Cropped table image.
        page_data : Dict[str, Any]
            Page metadata and token information.

        Returns
        -------
        np.ndarray
            Preprocessed image tensor.
        """
        # Resize to model input size
        if len(table_image.shape) == 3:
            h, w, c = table_image.shape
        else:
            h, w = table_image.shape
            table_image = cv2.cvtColor(table_image, cv2.COLOR_GRAY2RGB)

        # Resize to input size
        target_h, target_w = self.input_size
        resized_image = cv2.resize(table_image, (target_w, target_h))

        # Normalize to [0, 1] and convert to float32
        image_tensor = resized_image.astype(np.float32) / 255.0

        # Convert HWC to CHW
        image_tensor = np.transpose(image_tensor, (2, 0, 1))

        # Add batch dimension
        image_tensor = np.expand_dims(image_tensor, axis=0)

        return image_tensor

    def _postprocess_table_predictions(
        self,
        outputs: List[np.ndarray],
        page_data: Dict[str, Any],
        table_bbox: List[float],
        do_matching: bool = True,
    ) -> tuple[List[Dict], Dict]:
        """
        Post-process model outputs to table structure format.

        Parameters
        ----------
        outputs : List[np.ndarray]
            Raw model outputs.
        page_data : Dict[str, Any]
            Page metadata and token information.
        table_bbox : List[float]
            Table bounding box coordinates.
        do_matching : bool, optional
            Whether to perform cell matching, by default True.

        Returns
        -------
        tuple[List[Dict], Dict]
            Tuple of (tf_responses, predict_details).
        """
        # This is a simplified implementation
        # You'll need to adapt this based on your specific ONNX model outputs

        # For now, create a basic structure that matches the expected format
        tf_responses = []

        # Extract table structure information from model outputs
        # This will vary significantly based on your ONNX model architecture

        if len(outputs) > 0:
            # Assuming the model outputs table cell predictions
            predictions = outputs[0]  # Shape varies by model

            # Create mock responses that match the expected format
            # You'll need to replace this with actual post-processing logic
            for i in range(
                min(10, len(predictions) if len(predictions.shape) > 0 else 1)
            ):
                tf_responses.append(
                    {
                        "bbox": [
                            table_bbox[0] + i * 20,  # Mock coordinates
                            table_bbox[1] + i * 15,
                            table_bbox[0] + (i + 1) * 20,
                            table_bbox[1] + (i + 1) * 15,
                        ],
                        "text": f"Cell_{i}",  # Mock text
                        "row_id": i // 2,  # Mock row assignment
                        "col_id": i % 2,  # Mock column assignment
                    }
                )

        # Create predict details
        predict_details = {
            "num_rows": len(set(resp.get("row_id", 0) for resp in tf_responses)),
            "num_cols": len(set(resp.get("col_id", 0) for resp in tf_responses)),
            "prediction": {
                "rs_seq": ["fcel", "ecel"] * (len(tf_responses) // 2),  # Mock sequence
            },
        }

        return tf_responses, predict_details

    def predict_dummy(
        self,
        iocr_page: Dict[str, Any],
        table_bbox: List[float],
        table_image: np.ndarray,
        scale_factor: float,
        eval_res_preds: Optional[Dict] = None,
    ) -> tuple[List[Dict], Dict]:
        """
        Dummy prediction method for fallback scenarios.

        Parameters
        ----------
        iocr_page : Dict[str, Any]
            Page data with tokens and image information.
        table_bbox : List[float]
            Table bounding box [x1, y1, x2, y2].
        table_image : np.ndarray
            Cropped table image.
        scale_factor : float
            Scale factor applied to the image.
        eval_res_preds : Optional[Dict], optional
            Pre-computed predictions, by default None.

        Returns
        -------
        tuple[List[Dict], Dict]
            Tuple of (tf_responses, predict_details).
        """
        # Return empty structure for dummy prediction
        return [], {"num_rows": 0, "num_cols": 0, "prediction": {"rs_seq": []}}

    def predict(
        self,
        iocr_page: Dict[str, Any],
        table_bbox: List[float],
        table_image: np.ndarray,
        scale_factor: float,
        eval_res_preds: Optional[Dict] = None,
        correct_overlapping_cells: bool = False,
    ) -> tuple[List[Dict], Dict]:
        """
        Predict table structure for a single table.

        Parameters
        ----------
        iocr_page : Dict[str, Any]
            Page data with tokens and image information.
        table_bbox : List[float]
            Table bounding box [x1, y1, x2, y2].
        table_image : np.ndarray
            Cropped table image.
        scale_factor : float
            Scale factor applied to the image.
        eval_res_preds : Optional[Dict], optional
            Pre-computed predictions, by default None.
        correct_overlapping_cells : bool, optional
            Whether to correct overlapping cells, by default False.

        Returns
        -------
        tuple[List[Dict], Dict]
            Tuple of (tf_responses, predict_details).
        """
        try:
            # Preprocess the table image
            image_tensor = self._preprocess_table_image(table_image, iocr_page)

            # Run inference
            input_name = self.input_names[0] if self.input_names else "input"
            inputs = {input_name: image_tensor}

            outputs = self.run_inference(inputs)

            # Post-process predictions
            tf_responses, predict_details = self._postprocess_table_predictions(
                outputs, iocr_page, table_bbox, do_matching=True
            )

            return tf_responses, predict_details

        except Exception as e:
            _log.error(f"Table prediction failed: {e}")
            return self.predict_dummy(
                iocr_page, table_bbox, table_image, scale_factor, eval_res_preds
            )

    def multi_table_predict(
        self,
        iocr_page: Dict[str, Any],
        table_bboxes: List[List[float]],
        do_matching: bool = True,
        correct_overlapping_cells: bool = False,
        sort_row_col_indexes: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict structure for multiple tables in a page.

        Parameters
        ----------
        iocr_page : Dict[str, Any]
            Page data with tokens and image information.
        table_bboxes : List[List[float]]
            List of table bounding boxes, each as [x1, y1, x2, y2].
        do_matching : bool, optional
            Whether to perform cell matching, by default True.
        correct_overlapping_cells : bool, optional
            Whether to correct overlapping cells, by default False.
        sort_row_col_indexes : bool, optional
            Whether to sort row/column indexes, by default True.

        Returns
        -------
        List[Dict[str, Any]]
            List of prediction results, each containing:
            {"tf_responses": [...], "predict_details": {...}}
        """
        multi_tf_output = []
        page_image = iocr_page["image"]

        # Prevent large image submission by resizing input
        page_image_resized, scale_factor = self.resize_img(page_image, height=1024)

        for table_bbox in table_bboxes:
            # Downscale table bounding box to the size of new image
            scaled_bbox = [coord * scale_factor for coord in table_bbox]

            # Extract table image
            table_image = page_image_resized[
                round(scaled_bbox[1]) : round(scaled_bbox[3]),
                round(scaled_bbox[0]) : round(scaled_bbox[2]),
            ]

            # Predict table structure
            if do_matching:
                tf_responses, predict_details = self.predict(
                    iocr_page,
                    scaled_bbox,
                    table_image,
                    scale_factor,
                    None,
                    correct_overlapping_cells,
                )
            else:
                tf_responses, predict_details = self.predict_dummy(
                    iocr_page, scaled_bbox, table_image, scale_factor, None
                )

            # Store results
            multi_tf_output.append(
                {
                    "tf_responses": tf_responses,
                    "predict_details": predict_details,
                }
            )

        return multi_tf_output
