#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utility functions for ONNX model processing."""

import logging
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)


def prepare_image_input(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Prepare a single image for ONNX model input.

    Parameters
    ----------
    image : Union[Image.Image, np.ndarray]
        Input image to process.
    target_size : Tuple[int, int], optional
        Target size (height, width) for resizing, by default (224, 224).
    normalize : bool, optional
        Whether to normalize using ImageNet statistics, by default True.
    mean : List[float], optional
        Mean values for normalization, by default [0.485, 0.456, 0.406].
    std : List[float], optional
        Standard deviation values for normalization, by default [0.229, 0.224, 0.225].

    Returns
    -------
    np.ndarray
        Processed image as numpy array with shape (1, C, H, W).
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image
    image = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Apply ImageNet normalization if requested
    if normalize:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img_array = (img_array - mean) / std

    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def prepare_batch_input(
    images: List[Union[Image.Image, np.ndarray]],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Prepare a batch of images for ONNX model input.

    Parameters
    ----------
    images : List[Union[Image.Image, np.ndarray]]
        List of input images to process.
    target_size : Tuple[int, int], optional
        Target size (height, width) for resizing, by default (224, 224).
    normalize : bool, optional
        Whether to normalize using ImageNet statistics, by default True.
    mean : List[float], optional
        Mean values for normalization, by default [0.485, 0.456, 0.406].
    std : List[float], optional
        Standard deviation values for normalization, by default [0.229, 0.224, 0.225].

    Returns
    -------
    np.ndarray
        Processed images as numpy array with shape (B, C, H, W).
    """
    if not images:
        raise ValueError("Image list cannot be empty")

    batch_arrays = []

    for image in images:
        # Process each image individually
        img_array = prepare_image_input(image, target_size, normalize, mean, std)
        # Remove batch dimension since we'll stack them
        img_array = img_array[0]
        batch_arrays.append(img_array)

    # Stack all images into a batch
    batch_array = np.stack(batch_arrays, axis=0)

    return batch_array


def resize_image_with_aspect_ratio(
    image: Union[Image.Image, np.ndarray],
    max_size: int = 1024,
    interpolation=cv2.INTER_LINEAR,
) -> Tuple[Union[Image.Image, np.ndarray], float]:
    """
    Resize image while maintaining aspect ratio.

    Parameters
    ----------
    image : Union[Image.Image, np.ndarray]
        Input image to resize.
    max_size : int, optional
        Maximum dimension size, by default 1024.
    interpolation : int, optional
        OpenCV interpolation method, by default cv2.INTER_LINEAR.

    Returns
    -------
    Tuple[Union[Image.Image, np.ndarray], float]
        Tuple of (resized_image, scale_factor).
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        is_pil = True
    else:
        height, width = image.shape[:2]
        is_pil = False

    # Calculate scale factor
    scale_factor = min(max_size / width, max_size / height)

    if scale_factor >= 1.0:
        # No need to resize
        return image, 1.0

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    if is_pil:
        resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    else:
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=interpolation
        )

    return resized_image, scale_factor


def postprocess_bboxes(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    image_size: Tuple[int, int],
    score_threshold: float = 0.3,
) -> List[dict]:
    """
    Post-process object detection results.

    Parameters
    ----------
    bboxes : np.ndarray
        Bounding boxes with shape (N, 4) in format [x1, y1, x2, y2].
    scores : np.ndarray
        Confidence scores with shape (N,).
    labels : np.ndarray
        Class labels with shape (N,).
    image_size : Tuple[int, int]
        Image dimensions (width, height).
    score_threshold : float, optional
        Minimum score threshold for filtering, by default 0.3.

    Returns
    -------
    List[dict]
        List of detection dictionaries with keys: 'bbox', 'score', 'label'.
    """
    width, height = image_size
    results = []

    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_threshold:
            continue

        # Clip coordinates to image boundaries
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        results.append(
            {"bbox": [x1, y1, x2, y2], "score": float(score), "label": int(label)}
        )

    return results
