"""Thread-safe singleton inference service for the fixed Patch Proposed export."""

from __future__ import annotations

import hashlib
import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from infer_patch_proposed_single_roi import (
    CLASS_MAPPING,
    Phase2Transform,
    load_export_model,
    resolve_device,
    sha256_file,
)


EXPECTED_MODEL_SHA256 = "8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}

_singleton: "PatchROIInferenceService | None" = None
_singleton_lock = threading.Lock()


def _image_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _coerce_image(value: Any) -> tuple[Image.Image, str]:
    if isinstance(value, Image.Image):
        filename = Path(getattr(value, "filename", "uploaded_roi.png") or "uploaded_roi.png").name
        return ImageOps.exif_transpose(value).copy(), filename
    if isinstance(value, np.ndarray):
        if value.ndim not in {2, 3}:
            raise ValueError("Numpy image must have shape HxW, HxWx3, or HxWx4")
        if value.ndim == 3 and value.shape[2] not in {1, 3, 4}:
            raise ValueError("Numpy image channel count must be 1, 3, or 4")
        array = value
        if array.dtype != np.uint8:
            if not np.isfinite(array).all():
                raise ValueError("Numpy image contains NaN or Inf")
            maximum = float(array.max()) if array.size else 0.0
            if maximum <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array), "numpy_roi.png"
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("ROI file must be PNG, JPG, or JPEG")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"ROI file is missing or empty: {path}")
        try:
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as opened:
                return ImageOps.exif_transpose(opened).copy(), path.name
        except Exception as exc:
            raise ValueError("ROI file cannot be decoded by Pillow") from exc
    raise TypeError("ROI input must be a PIL Image, numpy array, or PNG/JPG/JPEG path")


def validate_roi_image(value: Any) -> tuple[Image.Image, dict[str, Any]]:
    image, filename = _coerce_image(value)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be greater than zero")
    if image.mode not in {"L", "RGB", "RGBA"}:
        image = image.convert("RGB")
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    if grayscale.size == 0 or int(grayscale.min()) == int(grayscale.max()):
        raise ValueError("ROI is blank or has no pixel variation")
    warnings: list[str] = []
    if (width, height) != (224, 224):
        warnings.append(
            f"輸入尺寸為 {width}x{height}；模型仍會套用固定 Resize 236 與 Center Crop 224。"
        )
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > 3.0:
        warnings.append(
            f"輸入長寬比為 {aspect_ratio:.2f}，可能不是典型病灶 ROI；請確認已先依 Ground Truth BBox 裁切。"
        )
    return image, {
        "filename": filename,
        "width": width,
        "height": height,
        "mode": image.mode,
        "aspect_ratio": aspect_ratio,
        "pixel_min": int(grayscale.min()),
        "pixel_max": int(grayscale.max()),
        "image_sha256": _image_sha256(image),
        "warnings": warnings,
    }


class PatchROIInferenceService:
    def __init__(self, model_path: Path, device_name: str) -> None:
        self.model_path = model_path.expanduser().resolve()
        if sha256_file(self.model_path) != EXPECTED_MODEL_SHA256:
            raise ValueError("Patch Proposed checkpoint SHA256 mismatch")
        self.device = resolve_device(device_name)
        self._inference_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._request_count = 0
        self._model_load_count = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device.index if self.device.index is not None else 0)
        self.model, self.checkpoint, self.load_audit, self.model_load_seconds = load_export_model(
            self.model_path, self.device
        )
        self._model_load_count += 1
        self.transform = Phase2Transform(training=False)
        self.preprocessing = self.transform.preprocessing_config()
        self.model_sha256 = EXPECTED_MODEL_SHA256

    @property
    def model_load_count(self) -> int:
        return self._model_load_count

    @property
    def inference_request_count(self) -> int:
        with self._counter_lock:
            return self._request_count

    def health(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "architecture": self.checkpoint["architecture"],
            "initialization": self.checkpoint["initialization_description"],
            "class_mapping": CLASS_MAPPING,
            "preprocessing": self.preprocessing,
            "strict_load": self.load_audit["strict_load"],
            "missing_keys": self.load_audit["missing_keys"],
            "unexpected_keys": self.load_audit["unexpected_keys"],
            "state_dict_key_count": self.load_audit["state_dict_key_count"],
            "model_eval": not self.model.training,
            "trainable_parameter_count": sum(
                int(parameter.requires_grad) for parameter in self.model.parameters()
            ),
            "model_load_count": self.model_load_count,
            "inference_request_count": self.inference_request_count,
            "device": str(self.device),
            "model_load_seconds": self.model_load_seconds,
            "optimizer_created": False,
            "backward_called": False,
            "test_images_read_count": 0,
        }

    def predict(self, image_value: Any, ground_truth_class_id: int | None = None) -> dict[str, Any]:
        if ground_truth_class_id is not None and ground_truth_class_id not in CLASS_MAPPING:
            raise ValueError("Ground Truth class must be one of 0, 1, 2, 3, or 4")
        image, image_audit = validate_roi_image(image_value)
        preprocessing_started = time.perf_counter()
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        if list(input_tensor.shape) != [1, 3, 224, 224]:
            raise RuntimeError(f"Unexpected input tensor shape: {list(input_tensor.shape)}")

        inference_started = time.perf_counter()
        with self._inference_lock:
            self.model.eval()
            with torch.inference_mode():
                logits = self.model(input_tensor)
                probabilities_tensor = torch.softmax(logits.float(), dim=1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - inference_started
        if list(logits.shape) != [1, 5] or list(probabilities_tensor.shape) != [1, 5]:
            raise RuntimeError("Unexpected logits or probabilities shape")
        nan_count = int(torch.isnan(logits).sum().item()) + int(
            torch.isnan(probabilities_tensor).sum().item()
        )
        inf_count = int(torch.isinf(logits).sum().item()) + int(
            torch.isinf(probabilities_tensor).sum().item()
        )
        if nan_count or inf_count:
            raise RuntimeError(f"Model output contains NaN={nan_count}, Inf={inf_count}")
        probability_sum = float(probabilities_tensor.sum().item())
        if abs(probability_sum - 1.0) > 1e-6:
            raise RuntimeError(f"Softmax probabilities do not sum to one: {probability_sum}")
        probabilities = [float(value) for value in probabilities_tensor[0].cpu().tolist()]
        predicted_class_id = int(torch.argmax(probabilities_tensor, dim=1).item())
        confidence = probabilities[predicted_class_id]
        with self._counter_lock:
            self._request_count += 1
            request_number = self._request_count
        gpu_memory = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        if self.device.type == "cuda":
            gpu_memory = {
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
        return {
            "request_number": request_number,
            "architecture": self.checkpoint["architecture"],
            "initialization": self.checkpoint["initialization_description"],
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "class_mapping": CLASS_MAPPING,
            "preprocessing": self.preprocessing,
            "image": image,
            "image_filename": image_audit["filename"],
            "image_sha256": image_audit["image_sha256"],
            "original_width": image_audit["width"],
            "original_height": image_audit["height"],
            "original_mode": image_audit["mode"],
            "warnings": image_audit["warnings"],
            "input_tensor_shape": list(input_tensor.shape),
            "input_tensor_dtype": str(input_tensor.dtype),
            "input_tensor_min": float(input_tensor.min().item()),
            "input_tensor_max": float(input_tensor.max().item()),
            "input_tensor_mean": float(input_tensor.mean().item()),
            "logits_shape": list(logits.shape),
            "probabilities_shape": list(probabilities_tensor.shape),
            "probability_sum": probability_sum,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "probabilities": probabilities,
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": CLASS_MAPPING[predicted_class_id],
            "confidence": confidence,
            "ground_truth_class_id": ground_truth_class_id,
            "ground_truth_class_name": (
                CLASS_MAPPING[ground_truth_class_id] if ground_truth_class_id is not None else None
            ),
            "true_class_probability": (
                probabilities[ground_truth_class_id] if ground_truth_class_id is not None else None
            ),
            "is_correct": (
                predicted_class_id == ground_truth_class_id
                if ground_truth_class_id is not None
                else None
            ),
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "device": str(self.device),
            "gpu_memory": gpu_memory,
            "strict_load": self.load_audit["strict_load"],
            "missing_keys": self.load_audit["missing_keys"],
            "unexpected_keys": self.load_audit["unexpected_keys"],
            "model_load_count": self.model_load_count,
            "optimizer_created": False,
            "backward_called": False,
            "threshold_used": False,
            "test_images_read_count": 0,
        }


def get_inference_service(model_path: Path, device_name: str = "auto") -> PatchROIInferenceService:
    global _singleton
    requested_path = model_path.expanduser().resolve()
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = PatchROIInferenceService(requested_path, device_name)
    if _singleton.model_path != requested_path:
        raise RuntimeError("Inference singleton is already bound to a different model path")
    if device_name != "auto" and str(_singleton.device) != str(resolve_device(device_name)):
        raise RuntimeError("Inference singleton is already bound to a different device")
    return _singleton
