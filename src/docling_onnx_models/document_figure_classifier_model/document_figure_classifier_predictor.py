#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import json
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..common.base_predictor import BaseONNXPredictor
from ..common.utils import prepare_batch_input

_log = logging.getLogger(__name__)


class DocumentFigureClassifierPredictor(BaseONNXPredictor):
    """
    ONNX-based document figure classifier.

    Classifies figures as 1 out of 16 possible classes, providing the same
    interface as the original DocumentFigureClassifierPredictor from docling-ibm-models.

    The classes are:
        1. "bar_chart"
        2. "bar_code"
        3. "chemistry_markush_structure"
        4. "chemistry_molecular_structure"
        5. "flow_chart"
        6. "icon"
        7. "line_chart"
        8. "logo"
        9. "map"
        10. "other"
        11. "pie_chart"
        12. "qr_code"
        13. "remote_sensing"
        14. "screenshot"
        15. "signature"
        16. "stamp"
    """

    def __init__(
        self,
        artifacts_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        onnx_model_name: str = "model.onnx",
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the ONNX Document Figure Classifier.

        Parameters
        ----------
        artifacts_path : str
            Path to the directory containing the ONNX model and config files.
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
        # Set up paths
        model_path = os.path.join(artifacts_path, onnx_model_name)
        self._config_path = os.path.join(artifacts_path, "config.json")

        # Load configuration if available
        self._load_config()

        # Initialize base ONNX predictor
        super().__init__(model_path, device, num_threads, providers)

        _log.debug("DocumentFigureClassifierPredictor initialized")

    def _load_config(self):
        """Load model configuration."""
        # Default class names for document figure classification
        self._classes = [
            "bar_chart",
            "bar_code",
            "chemistry_markush_structure",
            "chemistry_molecular_structure",
            "flow_chart",
            "icon",
            "line_chart",
            "logo",
            "map",
            "other",
            "pie_chart",
            "qr_code",
            "remote_sensing",
            "screenshot",
            "signature",
            "stamp",
        ]

        # Default preprocessing parameters (EfficientNet-style)
        self._image_mean = [0.485, 0.456, 0.406]
        self._image_std = [0.229, 0.224, 0.225]
        self._image_size = (224, 224)  # (height, width)

        # Load config if it exists
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r") as f:
                    config = json.load(f)

                # Override defaults with config values
                self._classes = config.get("id2label", {})
                if isinstance(self._classes, dict):
                    # Convert id2label dict to list
                    self._classes = [
                        self._classes[str(i)] for i in range(len(self._classes))
                    ]

                # Get preprocessing parameters
                self._image_mean = config.get("image_mean", self._image_mean)
                self._image_std = config.get("image_std", self._image_std)

                # Get image size
                if "image_size" in config:
                    size = config["image_size"]
                    if isinstance(size, list):
                        self._image_size = tuple(size)
                    else:
                        self._image_size = (size, size)

            except Exception as e:
                _log.warning(f"Failed to load config, using defaults: {e}")

        _log.debug(f"Loaded {len(self._classes)} classes: {self._classes}")

    def info(self) -> dict:
        """
        Get predictor configuration information.

        Returns
        -------
        dict
            Dictionary containing predictor configuration details.
        """
        base_info = super().info()
        base_info.update(
            {
                "num_classes": len(self._classes),
                "classes": self._classes,
                "image_size": self._image_size,
            }
        )
        return base_info

    def _preprocess_images(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        Preprocess images for model input.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of input images.

        Returns
        -------
        np.ndarray
            Preprocessed batch tensor.
        """
        if not images:
            return np.array([])

        # Use the common utility for batch preparation
        batch_input = prepare_batch_input(
            images,
            target_size=self._image_size,
            normalize=True,
            mean=self._image_mean,
            std=self._image_std,
        )

        return batch_input

    def _postprocess_predictions(
        self, outputs: List[np.ndarray]
    ) -> List[List[Tuple[str, float]]]:
        """
        Post-process model outputs to class predictions.

        Parameters
        ----------
        outputs : List[np.ndarray]
            Raw model outputs.

        Returns
        -------
        List[List[Tuple[str, float]]]
            List of predictions per image, each containing tuples of
            (class_name, confidence) sorted by confidence descending.
        """
        # Get logits (first output)
        logits = outputs[0]  # Shape: [batch_size, num_classes]

        # Apply softmax to get probabilities
        probabilities = self._softmax(logits)

        batch_predictions = []

        for probs in probabilities:
            # Create (class_name, confidence) pairs
            predictions = [
                (self._classes[i], float(probs[i])) for i in range(len(self._classes))
            ]

            # Sort by confidence descending
            predictions.sort(key=lambda x: x[1], reverse=True)

            batch_predictions.append(predictions)

        return batch_predictions

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function to convert logits to probabilities."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict figure classes for input images.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to classify.

        Returns
        -------
        List[List[Tuple[str, float]]]
            List of predictions per image, each containing tuples of
            (class_name, confidence) sorted by confidence descending.
        """
        if not images:
            return []

        # Preprocess images
        batch_input = self._preprocess_images(images)

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
        predictions = self._postprocess_predictions(outputs)

        return predictions
