#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Base class for ONNX predictors."""

import logging
import os
import platform
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) acceleration is available via CoreML.

    Returns
    -------
    bool
        True if MPS acceleration is available through CoreMLExecutionProvider.
    """
    if platform.system() != "Darwin":
        return False
    
    available_providers = ort.get_available_providers()
    return "CoreMLExecutionProvider" in available_providers


def get_provider_options(device: str = "auto") -> List[Union[str, tuple]]:
    """
    Get execution providers with configuration options for optimal performance.

    Parameters
    ----------
    device : str, optional
        Device preference ('auto', 'cpu', 'cuda', 'coreml', 'mps'), by default "auto".

    Returns
    -------
    List[Union[str, tuple]]
        List of execution providers with configuration options.
    """
    available_providers = ort.get_available_providers()
    providers = []

    if device == "auto":
        system = platform.system()

        if system == "Darwin":  # macOS
            if "CoreMLExecutionProvider" in available_providers:
                # Configure CoreML with GPU+Neural Engine for optimal performance
                coreml_config = {
                    "MLComputeUnits": "ALL",  # Use CPU, GPU, and Neural Engine
                    "ModelFormat": "MLProgram",  # Use the newer MLProgram format
                    "RequireStaticInputShapes": False,
                }
                providers.append(("CoreMLExecutionProvider", coreml_config))
                _log.debug("Added CoreMLExecutionProvider with GPU+Neural Engine support for macOS")

            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")

        elif system in ["Linux", "Windows"]:
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
                _log.debug("Added CUDAExecutionProvider")

            if system == "Windows" and "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")
                _log.debug("Added DmlExecutionProvider for Windows")

            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")

        else:
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")

    elif device.lower() == "cuda":
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        else:
            _log.warning("CUDA requested but not available, falling back to CPU")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    elif device.lower() in ["coreml", "mps"]:
        if "CoreMLExecutionProvider" in available_providers:
            # Configure CoreML with optimal settings for MPS/GPU acceleration
            compute_units = "ALL"  # Default to using all available compute units
            if device.lower() == "mps":
                # For explicit MPS request, prioritize GPU but allow fallback
                compute_units = "CPUAndGPU"
                _log.info("Configuring CoreML for MPS (Metal Performance Shaders) acceleration")
            
            coreml_config = {
                "MLComputeUnits": compute_units,
                "ModelFormat": "MLProgram",
                "RequireStaticInputShapes": False,
            }
            providers.append(("CoreMLExecutionProvider", coreml_config))
        else:
            provider_name = "CoreML/MPS" if device.lower() == "mps" else "CoreML"
            _log.warning(f"{provider_name} requested but CoreMLExecutionProvider not available, falling back to CPU")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    elif device.lower() == "cpu":
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    else:
        # Custom device specification - assume it's a provider name
        if device in available_providers:
            providers.append(device)
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    # Always ensure we have CPU as final fallback
    if not providers or not any(
        p == "CPUExecutionProvider" or (isinstance(p, tuple) and p[0] == "CPUExecutionProvider")
        for p in providers
    ):
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    _log.info(f"Selected execution providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
    return providers


def get_optimal_providers(device: str = "auto") -> List[str]:
    """
    Get optimal execution providers based on system capabilities and device preference.

    Parameters
    ----------
    device : str, optional
        Device preference ('auto', 'cpu', 'cuda', 'coreml', 'mps'), by default "auto".

    Returns
    -------
    List[str]
        Ordered list of execution providers to try.
    """
    available_providers = ort.get_available_providers()
    providers = []

    if device == "auto":
        # Auto-select based on platform and available providers
        system = platform.system()

        if system == "Darwin":  # macOS
            # Prioritize CoreML on macOS
            if "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider")
                _log.debug("Added CoreMLExecutionProvider for macOS")

            # Apple Silicon Macs may also benefit from CPU optimization
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")

        elif system in ["Linux", "Windows"]:
            # Check for CUDA first on Linux/Windows
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
                _log.debug("Added CUDAExecutionProvider")

            # DirectML for Windows
            if system == "Windows" and "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")
                _log.debug("Added DmlExecutionProvider for Windows")

            # CPU as fallback
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")

        else:
            # Default to CPU for unknown systems
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")

    elif device.lower() == "cuda":
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        else:
            _log.warning("CUDA requested but not available, falling back to CPU")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    elif device.lower() in ["coreml", "mps"]:
        if "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")
        else:
            provider_name = "CoreML/MPS" if device.lower() == "mps" else "CoreML"
            _log.warning(f"{provider_name} requested but CoreMLExecutionProvider not available, falling back to CPU")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    elif device.lower() == "cpu":
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    else:
        # Custom device specification - assume it's a provider name
        if device in available_providers:
            providers.append(device)
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    # Always ensure we have CPU as final fallback
    if not providers or "CPUExecutionProvider" not in providers:
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

    _log.info(f"Selected execution providers: {providers}")
    return providers


class BaseONNXPredictor(ABC):
    """
    Base class for all ONNX-based predictors.

    Provides common functionality for loading ONNX models and managing
    inference sessions with proper threading and device management.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        num_threads: int = 4,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the ONNX predictor.

        Parameters
        ----------
        model_path : str
            Path to the ONNX model file.
        device : str, optional
            Device to run inference on ('auto', 'cpu', 'cuda', 'coreml', 'mps'), by default "auto".
        num_threads : int, optional
            Number of threads for CPU inference, by default 4.
        providers : List[str], optional
            ONNX Runtime execution providers, by default None.
            If None, will auto-select optimal providers based on device and system.
        """
        self.model_path = model_path
        self.device = device
        self.num_threads = num_threads

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        # Set up execution providers
        if providers is None:
            providers = get_provider_options(device)

        # Configure session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = num_threads

        # Create inference session with thread safety
        with _model_init_lock:
            try:
                self.session = ort.InferenceSession(
                    model_path, providers=providers, sess_options=session_options
                )
                _log.info(f"Loaded ONNX model from {model_path}")
                _log.info(f"Available providers: {self.session.get_providers()}")
            except Exception as e:
                _log.error(f"Failed to load ONNX model: {e}")
                raise

        # Get model input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        _log.debug(f"Model inputs: {self.input_names}")
        _log.debug(f"Model outputs: {self.output_names}")

    def info(self) -> Dict[str, Any]:
        """
        Get predictor information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing predictor configuration details.
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "num_threads": self.num_threads,
            "providers": self.session.get_providers(),
            "input_names": self.input_names,
            "output_names": self.output_names,
        }

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Run ONNX model inference.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Dictionary of input arrays keyed by input names.

        Returns
        -------
        List[np.ndarray]
            List of output arrays from the model.
        """
        try:
            outputs = self.session.run(self.output_names, inputs)
            return outputs
        except Exception as e:
            _log.error(f"Inference failed: {e}")
            raise

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Abstract method for running prediction on inputs.

        Must be implemented by concrete predictor classes.

        Parameters
        ----------
        inputs : Any
            Model-specific input data.

        Returns
        -------
        Any
            Model-specific prediction results.
        """
        pass
