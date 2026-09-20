"""
Model Loader — Lazy Loading & Quantization Support
Loads models on-demand with ONNX Runtime optimization for CPU.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st

from ultralytics import YOLO
import onnxruntime as ort


class ModelLoader:
    """Manages lazy loading of quantized models for CPU inference."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}

    def get_model_path(self, module: str, model_name: str, quantized: bool = True) -> Optional[Path]:
        """
        Get path to model file, preferring quantized versions.

        Args:
            module: 'detection', 'radiology', or 'chatbot'
            model_name: e.g., 'malaria', 'mri', 'ct', 'xray'
            quantized: Whether to prefer quantized (.onnx) over original (.pt)

        Returns:
            Path to model file or None if not found
        """
        module_dir = self.models_dir / module
        if not module_dir.exists():
            return None

        # Try quantized ONNX first
        if quantized:
            onnx_path = module_dir / f"{model_name}_yolov8n.onnx"
            if onnx_path.exists():
                return onnx_path

        # Try PyTorch weights
        pt_path = module_dir / f"{model_name}_yolov8n.pt"
        if pt_path.exists():
            return pt_path

        # Try any matching file
        matches = list(module_dir.glob(f"{model_name}*.pt")) + list(module_dir.glob(f"{model_name}*.onnx"))
        return matches[0] if matches else None

    @st.cache_resource
    def load_yolo_model(_self, model_path: str, device: str = "cpu") -> Optional[YOLO]:
        """Load YOLO model (cached)."""
        try:
            model = YOLO(model_path)
            model.to(device)
            print(f"Loaded YOLO model from {model_path} on {device}")
            return model
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return None

    def load_onnx_session(self, model_path: str, providers: list = None) -> Optional[ort.InferenceSession]:
        """Load ONNX model with ONNX Runtime for optimized CPU inference."""
        if providers is None:
            providers = ["CPUExecutionProvider"]

        try:
            session = ort.InferenceSession(model_path, providers=providers)
            print(f"Loaded ONNX model from {model_path} with providers: {providers}")
            return session
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return None

    def get_or_load_model(
        self,
        module: str,
        model_name: str,
        device: str = "cpu",
        prefer_onnx: bool = True,
    ) -> Optional[Any]:
        """
        Get cached model or load it lazily.

        Args:
            module: Module name ('detection', 'radiology')
            model_name: Model identifier ('malaria', 'mri', etc.)
            device: 'cpu' or 'cuda'
            prefer_onnx: Prefer ONNX Runtime over PyTorch

        Returns:
            Loaded model or None
        """
        cache_key = f"{module}_{model_name}_{device}"

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        model_path = self.get_model_path(module, model_name, quantized=prefer_onnx)
        if not model_path:
            print(f"No model found for {module}/{model_name}")
            return None

        if prefer_onnx and model_path.suffix == ".onnx":
            model = self.load_onnx_session(str(model_path))
        else:
            model = self.load_yolo_model(str(model_path), device)

        if model:
            self._loaded_models[cache_key] = model

        return model

    def unload_model(self, module: str, model_name: str, device: str = "cpu") -> None:
        """Unload a model from cache to free memory."""
        cache_key = f"{module}_{model_name}_{device}"
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]

    def unload_all(self) -> None:
        """Unload all models."""
        self._loaded_models.clear()

    def get_model_info(self, module: str, model_name: str) -> Dict:
        """Get information about available model files."""
        module_dir = self.models_dir / module
        if not module_dir.exists():
            return {"available": False, "files": []}

        files = []
        for ext in [".pt", ".onnx"]:
            for f in module_dir.glob(f"{model_name}*{ext}"):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "format": ext[1:].upper(),
                })

        return {
            "available": len(files) > 0,
            "files": files,
        }


@st.cache_resource
def get_model_loader(models_dir: str = "models") -> ModelLoader:
    """Get cached ModelLoader instance."""
    return ModelLoader(models_dir)


def load_model_for_module(module: str, submodule: str, device: str = "cpu") -> Optional[Any]:
    """
    Convenience function to load model for a specific module/submodule.

    Args:
        module: 'detection', 'radiology', 'chatbot'
        submodule: e.g., 'malaria', 'mri', 'ct', 'xray'
        device: 'cpu' or 'cuda'

    Returns:
        Loaded model or None
    """
    loader = get_model_loader()
    return loader.get_or_load_model(module, submodule, device)