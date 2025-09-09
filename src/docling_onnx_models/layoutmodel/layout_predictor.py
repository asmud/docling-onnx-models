#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json
import logging
import os
from typing import Dict, Generator, List, Optional, Set, Union

import numpy as np
from PIL import Image

from ..common.base_predictor import BaseONNXPredictor
from ..common.utils import prepare_batch_input, resize_image_with_aspect_ratio
from .labels import LayoutLabels

_log = logging.getLogger(__name__)


class LayoutPredictor(BaseONNXPredictor):
    """
    ONNX-based document layout prediction.

    This class provides the same interface as the original LayoutPredictor
    from docling-ibm-models but uses ONNX Runtime for inference.
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
        onnx_model_name: str = "model.onnx",
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the ONNX Layout Predictor.

        Parameters
        ----------
        artifact_path : str
            Path to the directory containing the ONNX model and config files.
        device : str, optional
            Device to run inference on ('cpu' or 'cuda'), by default "cpu".
        num_threads : int, optional
            Number of threads for CPU inference, by default 4.
        base_threshold : float, optional
            Score threshold for filtering predictions, by default 0.3.
        blacklist_classes : Set[str], optional
            Set of class names to filter out, by default empty set.
        onnx_model_name : str, optional
            Name of the ONNX model file, by default "model.onnx".
        providers : List[str], optional
            ONNX Runtime execution providers, by default None.

        Raises
        ------
        FileNotFoundError
            When required model files are missing.
        """
        # Set basic parameters
        self._threshold = base_threshold
        self._black_classes = blacklist_classes
        self._labels = LayoutLabels()

        # Construct paths to required files
        model_path = os.path.join(artifact_path, onnx_model_name)
        self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        self._model_config = os.path.join(artifact_path, "config.json")

        # Verify required files exist
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(
                f"Missing processor config file: {self._processor_config}"
            )
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        # Load configuration
        self._load_configs()

        # Initialize base ONNX predictor
        super().__init__(model_path, device, num_threads, providers)

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def _load_configs(self):
        """Load processor and model configurations."""
        # Load processor config
        with open(self._processor_config, "r") as f:
            self._processor_cfg = json.load(f)

        # Load model config
        with open(self._model_config, "r") as f:
            self._model_cfg = json.load(f)

        # Extract image preprocessing parameters
        self._image_mean = self._processor_cfg.get("image_mean", [0.485, 0.456, 0.406])
        self._image_std = self._processor_cfg.get("image_std", [0.229, 0.224, 0.225])

        # Determine class mapping based on model type
        model_type = self._model_cfg.get("model_type", "")
        if "rtdetr" in model_type.lower():
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

        _log.debug(f"Using label offset: {self._label_offset}")

    def info(self) -> Dict:
        """
        Get predictor configuration information.

        Returns
        -------
        Dict
            Dictionary containing predictor settings and model information.
        """
        base_info = super().info()
        base_info.update(
            {
                "threshold": self._threshold,
                "blacklist_classes": list(self._black_classes),
                "label_offset": self._label_offset,
                "num_classes": len(self._classes_map),
            }
        )
        return base_info

    def _preprocess_images(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> tuple[np.ndarray, List[tuple[int, int]]]:
        """
        Preprocess images for model input.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of input images.

        Returns
        -------
        tuple[np.ndarray, List[tuple[int, int]]]
            Tuple of (preprocessed_batch, original_sizes).
        """
        if not images:
            return np.array([]), []

        # Convert to PIL Images and get original sizes
        pil_images = []
        original_sizes = []

        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            else:
                raise TypeError("Unsupported input image format")

            pil_images.append(pil_img)
            original_sizes.append(pil_img.size)  # (width, height)

        # Get input size from model config or use default
        input_size = self._model_cfg.get("input_size", [640, 640])  # [height, width]
        if len(input_size) == 1:
            input_size = [input_size[0], input_size[0]]

        # Prepare batch input
        batch_input = prepare_batch_input(
            pil_images,
            target_size=(input_size[0], input_size[1]),  # (height, width)
            normalize=True,
            mean=self._image_mean,
            std=self._image_std,
        )

        return batch_input, original_sizes

    def _postprocess_predictions(
        self, outputs: List[np.ndarray], original_sizes: List[tuple[int, int]]
    ) -> List[List[dict]]:
        """
        Post-process model outputs to standard format.

        Parameters
        ----------
        outputs : List[np.ndarray]
            Raw model outputs.
        original_sizes : List[tuple[int, int]]
            Original image sizes as (width, height).

        Returns
        -------
        List[List[dict]]
            List of predictions per image, each containing:
            {"label", "confidence", "l", "t", "r", "b"}
        """
        # This is a generic implementation - you may need to adjust
        # based on your specific ONNX model's output format

        # Assuming outputs are: [boxes, scores, labels] or similar
        # You'll need to adapt this based on your actual ONNX model structure

        if len(outputs) >= 3:
            # Standard object detection format
            boxes = outputs[0]  # [batch, num_detections, 4]
            scores = outputs[1]  # [batch, num_detections]
            labels = outputs[2]  # [batch, num_detections]
        else:
            _log.warning("Unexpected output format from ONNX model")
            return [[] for _ in original_sizes]

        all_predictions = []

        for batch_idx, (w, h) in enumerate(original_sizes):
            predictions = []

            # Get predictions for this image
            if boxes.ndim == 3:
                img_boxes = boxes[batch_idx]
                img_scores = scores[batch_idx]
                img_labels = labels[batch_idx]
            else:
                # Handle different output formats
                img_boxes = boxes
                img_scores = scores
                img_labels = labels

            for box, score, label_id in zip(img_boxes, img_scores, img_labels):
                # Filter by threshold
                if score < self._threshold:
                    continue

                # Convert label ID to string
                label_id_int = int(label_id) + self._label_offset
                if label_id_int not in self._classes_map:
                    continue

                label_str = self._classes_map[label_id_int]

                # Filter blacklisted classes
                if label_str in self._black_classes:
                    continue

                # Convert box coordinates (assuming normalized [0,1] format)
                # Adjust this based on your ONNX model's output format
                if box.max() <= 1.0:
                    # Normalized coordinates
                    l = max(0, min(w, box[0] * w))
                    t = max(0, min(h, box[1] * h))
                    r = max(0, min(w, box[2] * w))
                    b = max(0, min(h, box[3] * h))
                else:
                    # Absolute coordinates
                    l = max(0, min(w, box[0]))
                    t = max(0, min(h, box[1]))
                    r = max(0, min(w, box[2]))
                    b = max(0, min(h, box[3]))

                predictions.append(
                    {
                        "l": float(l),
                        "t": float(t),
                        "r": float(r),
                        "b": float(b),
                        "label": label_str,
                        "confidence": float(score),
                    }
                )

            all_predictions.append(predictions)

        return all_predictions

    def predict(
        self, page_img: Union[Image.Image, np.ndarray]
    ) -> Generator[dict, None, None]:
        """
        Predict layout elements for a single image.

        Parameters
        ----------
        page_img : Union[Image.Image, np.ndarray]
            Input page image.

        Yields
        ------
        dict
            Prediction dictionary with keys: "label", "confidence", "l", "t", "r", "b".
        """
        predictions = self.predict_batch([page_img])
        for pred in predictions[0]:
            yield pred

    def predict_batch(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to process in a single batch.

        Returns
        -------
        List[List[dict]]
            List of prediction lists, one per input image. Each prediction dict contains:
            "label", "confidence", "l", "t", "r", "b".
        """
        if not images:
            return []

        # Preprocess images
        batch_input, original_sizes = self._preprocess_images(images)

        if batch_input.size == 0:
            return []

        # Run inference
        # Note: Input name might vary - you may need to check your ONNX model
        input_name = self.input_names[0] if self.input_names else "input"
        inputs = {input_name: batch_input}

        try:
            outputs = self.run_inference(inputs)
        except Exception as e:
            _log.error(f"Inference failed: {e}")
            return [[] for _ in images]

        # Post-process predictions
        predictions = self._postprocess_predictions(outputs, original_sizes)

        return predictions
