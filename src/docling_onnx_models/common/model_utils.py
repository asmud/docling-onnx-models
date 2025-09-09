#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utilities for ONNX model detection and management."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_log = logging.getLogger(__name__)


def detect_onnx_model(
    artifact_path: Union[str, Path], onnx_model_name: str = "model.onnx"
) -> Optional[str]:
    """
    Detect if ONNX model exists in the artifact path.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to search for ONNX model.
    onnx_model_name : str, optional
        Name of the ONNX model file, by default "model.onnx".

    Returns
    -------
    Optional[str]
        Full path to ONNX model if found, None otherwise.
    """
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        return None

    onnx_path = artifact_path / onnx_model_name

    if onnx_path.exists() and onnx_path.is_file():
        _log.debug(f"Found ONNX model: {onnx_path}")
        return str(onnx_path)

    return None


def has_onnx_support(artifact_path: Union[str, Path]) -> bool:
    """
    Check if a model directory has ONNX support.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to check for ONNX model files.

    Returns
    -------
    bool
        True if ONNX model is available.
    """
    return detect_onnx_model(artifact_path) is not None


def get_model_info(artifact_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about available models in the artifact path.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to examine for model files.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model information.
    """
    artifact_path = Path(artifact_path)
    info = {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "has_onnx": False,
        "has_pytorch": False,
        "has_config": False,
        "has_preprocessor_config": False,
        "onnx_model_path": None,
        "pytorch_model_files": [],
        "config_files": [],
    }

    if not artifact_path.exists():
        return info

    # Check for ONNX models
    onnx_path = detect_onnx_model(artifact_path)
    if onnx_path:
        info["has_onnx"] = True
        info["onnx_model_path"] = onnx_path

    # Check for PyTorch models
    pytorch_extensions = [".pt", ".pth", ".bin", ".safetensors"]
    for file in artifact_path.iterdir():
        if file.is_file():
            if any(file.name.endswith(ext) for ext in pytorch_extensions):
                info["pytorch_model_files"].append(str(file))
                info["has_pytorch"] = True

            if file.name in ["config.json", "model_config.json"]:
                info["has_config"] = True
                info["config_files"].append(str(file))

            if file.name == "preprocessor_config.json":
                info["has_preprocessor_config"] = True
                info["config_files"].append(str(file))

    return info


def validate_onnx_model_directory(artifact_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate that a directory contains the required files for ONNX model loading.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to validate.

    Returns
    -------
    Dict[str, Any]
        Validation results with status and missing files.
    """
    artifact_path = Path(artifact_path)

    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "missing_files": [],
        "found_files": [],
    }

    if not artifact_path.exists():
        result["errors"].append(f"Artifact path does not exist: {artifact_path}")
        return result

    # Required files
    required_files = ["model.onnx"]
    optional_files = ["config.json", "preprocessor_config.json"]

    # Check required files
    missing_required = []
    for file in required_files:
        file_path = artifact_path / file
        if file_path.exists():
            result["found_files"].append(file)
        else:
            missing_required.append(file)
            result["missing_files"].append(file)

    # Check optional files
    for file in optional_files:
        file_path = artifact_path / file
        if file_path.exists():
            result["found_files"].append(file)
        else:
            result["warnings"].append(f"Optional file missing: {file}")

    # Determine if valid
    if missing_required:
        result["errors"].append(f"Missing required files: {missing_required}")
        result["valid"] = False
    else:
        result["valid"] = True

    return result


def create_onnx_model_config(
    base_config: Optional[Dict] = None,
    onnx_model_path: str = "model.onnx",
    execution_providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create configuration for ONNX model loading.

    Parameters
    ----------
    base_config : Optional[Dict], optional
        Base configuration to extend, by default None.
    onnx_model_path : str, optional
        Path to ONNX model file, by default "model.onnx".
    execution_providers : Optional[List[str]], optional
        Execution providers to use, by default None.

    Returns
    -------
    Dict[str, Any]
        ONNX model configuration.
    """
    config = base_config.copy() if base_config else {}

    # Add ONNX-specific settings
    config.update(
        {
            "model_format": "onnx",
            "onnx_model_path": onnx_model_path,
            "use_onnx": True,
        }
    )

    if execution_providers:
        config["execution_providers"] = execution_providers

    return config


def log_model_info(artifact_path: Union[str, Path]) -> None:
    """
    Log detailed information about available models in a directory.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to examine and log information about.
    """
    info = get_model_info(artifact_path)

    _log.info(f"Model directory: {info['path']}")
    _log.info(f"  Exists: {info['exists']}")

    if info["exists"]:
        _log.info(f"  Has ONNX model: {info['has_onnx']}")
        if info["has_onnx"]:
            _log.info(f"    ONNX model path: {info['onnx_model_path']}")

        _log.info(f"  Has PyTorch model: {info['has_pytorch']}")
        if info["pytorch_model_files"]:
            for pt_file in info["pytorch_model_files"]:
                _log.info(f"    PyTorch file: {pt_file}")

        _log.info(f"  Has config: {info['has_config']}")
        _log.info(f"  Has preprocessor config: {info['has_preprocessor_config']}")

        for config_file in info["config_files"]:
            _log.info(f"    Config file: {config_file}")


def prefer_onnx_model(artifact_path: Union[str, Path]) -> bool:
    """
    Determine if ONNX model should be preferred over PyTorch model.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to check for models.

    Returns
    -------
    bool
        True if ONNX model should be preferred.
    """
    info = get_model_info(artifact_path)

    # Prefer ONNX if:
    # 1. ONNX model exists
    # 2. Either no PyTorch model or ONNX is complete
    if info["has_onnx"]:
        if not info["has_pytorch"]:
            return True

        # Both exist - prefer ONNX if it has required configs
        validation = validate_onnx_model_directory(artifact_path)
        return validation["valid"]

    return False
