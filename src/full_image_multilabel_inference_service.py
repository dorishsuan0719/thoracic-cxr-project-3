"""Thread-safe singleton inference service for the formal Full-image multilabel model."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from full_image_multilabel_report_prompt import (
    CLASS_MAPPING_EN,
    CLASS_MAPPING_ZH,
    DISCLAIMER,
    NO_POSITIVE_MESSAGE,
)
from infer_full_image_224_multilabel_single import (
    FullImageMultilabelConvNeXt,
    FullImageTransform,
    load_thresholds,
    sample_metrics,
    sha256_file,
    validate_checkpoint,
)


EXPECTED_FORMAL_MODEL_SHA256 = "0287fe36d3623ccdb5aa43857db1168a1598788071ebdecbc43324a6953f426f"
EXPECTED_THRESHOLD_SHA256 = "73a54c9b6a3de2b2f63479b0bd918836cedb577a29255b0f6e0b30dac310e9d5"
DEFAULT_MODEL_RELATIVE = Path(
    "outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/"
    "checkpoints/full_image_multilabel_patch_transfer.pt"
)
DEFAULT_THRESHOLDS_RELATIVE = Path(
    "outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/"
    "validation_selected_thresholds.json"
)
DEFAULT_CATALOG_RELATIVE = Path(
    "outputs/full_image_ground_truth_catalog/full_image_ground_truth_manifest.csv"
)
PREPROCESSING_DESCRIPTION = (
    "完整胸腔 X 光轉為 RGB，整張直接以 BILINEAR antialias resize 至 224x224，"
    "ToTensor 後使用 ImageNet mean/std normalization；不使用 BBox、ROI 或裁切。"
)


_singleton_lock = threading.Lock()
_instances: dict[tuple[str, str, str, str], "FullImageMultilabelInferenceService"] = {}


def _resolve_device(requested: str) -> torch.device:
    if requested.strip().lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def _hash_pil(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _names(ids: list[int], mapping: dict[int, str]) -> list[str]:
    return [mapping[class_id] for class_id in ids]


class FullImageMultilabelInferenceService:
    def __init__(
        self,
        model_path: Path,
        threshold_path: Path,
        device: str,
        catalog_path: Path | None,
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        self.threshold_path = threshold_path.expanduser().resolve()
        self.catalog_path = catalog_path.expanduser().resolve() if catalog_path else None
        self.device = _resolve_device(device)
        self._inference_lock = threading.Lock()
        self._model_load_count = 0
        self._inference_count = 0
        self._optimizer_created = False
        self._backward_executed = False
        self._catalog_by_sha: dict[str, dict[str, str]] = {}
        self._catalog_by_image_id: dict[str, dict[str, str]] = {}
        self._catalog_duplicate_sha: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Formal Full-image checkpoint is missing: {self.model_path}")
        if not self.threshold_path.is_file():
            raise FileNotFoundError(f"Validation threshold JSON is missing: {self.threshold_path}")
        self.model_sha256 = sha256_file(self.model_path)
        self.threshold_sha256 = sha256_file(self.threshold_path)
        if self.model_sha256 != EXPECTED_FORMAL_MODEL_SHA256:
            raise RuntimeError(
                f"Formal Full-image checkpoint SHA256 mismatch: {self.model_sha256}"
            )
        if self.threshold_sha256 != EXPECTED_THRESHOLD_SHA256:
            raise RuntimeError(f"Validation threshold SHA256 mismatch: {self.threshold_sha256}")

        self.thresholds, self.threshold_payload = load_thresholds(self.threshold_path)
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state = validate_checkpoint(checkpoint, self.thresholds)
        self.architecture = checkpoint["architecture"]
        self.initialization = checkpoint["experiment"]
        self.checkpoint_kind = checkpoint["checkpoint_kind"]
        self.checkpoint_test_evaluation_count = int(checkpoint["test_evaluation_count"])
        self.model = FullImageMultilabelConvNeXt()
        incompatible = self.model.load_state_dict(state, strict=True)
        self.missing_keys = list(incompatible.missing_keys)
        self.unexpected_keys = list(incompatible.unexpected_keys)
        if self.missing_keys or self.unexpected_keys:
            raise RuntimeError(
                f"Strict load failed: missing={self.missing_keys}, unexpected={self.unexpected_keys}"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("At least one model parameter remains trainable")
        self.model.to(self.device)
        self.transform = FullImageTransform(224)
        self._model_load_count += 1
        self._load_catalog()

    def _load_catalog(self) -> None:
        if self.catalog_path is None or not self.catalog_path.is_file():
            return
        sha_rows: dict[str, list[dict[str, str]]] = {}
        with self.catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            image_id = row.get("image_id", "").strip()
            image_sha = row.get("image_sha256", "").strip().lower()
            if not image_id or not image_sha:
                raise RuntimeError("Ground Truth catalog contains an empty image_id or SHA256")
            if image_id in self._catalog_by_image_id:
                raise RuntimeError(f"Ground Truth catalog duplicate image_id: {image_id}")
            self._catalog_by_image_id[image_id] = row
            sha_rows.setdefault(image_sha, []).append(row)
        self._catalog_duplicate_sha = sorted(sha for sha, matches in sha_rows.items() if len(matches) > 1)
        self._catalog_by_sha = {
            sha: matches[0] for sha, matches in sha_rows.items() if len(matches) == 1
        }
        if len(rows) != 590:
            raise RuntimeError(f"Ground Truth catalog must contain 590 rows, got {len(rows)}")

    @property
    def model_load_count(self) -> int:
        return self._model_load_count

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def health(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "model_loaded": True,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "threshold_path": str(self.threshold_path),
            "threshold_sha256": self.threshold_sha256,
            "thresholds": self.thresholds,
            "architecture": self.architecture,
            "initialization": self.initialization,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_test_evaluation_count": self.checkpoint_test_evaluation_count,
            "strict_load": True,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "model_eval": not self.model.training,
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in self.model.parameters()
            ),
            "device": str(self.device),
            "model_load_count": self.model_load_count,
            "singleton": True,
            "inference_lock": isinstance(self._inference_lock, type(threading.Lock())),
            "catalog_available": bool(self._catalog_by_image_id),
            "catalog_path": str(self.catalog_path) if self.catalog_path else None,
            "catalog_rows": len(self._catalog_by_image_id),
            "catalog_duplicate_sha256": len(self._catalog_duplicate_sha),
            "uses_bbox": False,
            "uses_roi_crop": False,
            "uses_yolo": False,
            "uses_softmax": False,
            "uses_sigmoid": True,
            "optimizer_created": self._optimizer_created,
            "backward_executed": self._backward_executed,
            "test_images_read_count": 0,
        }

    def _open_input(self, image_input: Any) -> tuple[Image.Image, dict[str, Any], Path | None]:
        source_path: Path | None = None
        if isinstance(image_input, (str, Path)):
            source_path = Path(image_input).expanduser().resolve()
            if not source_path.is_file() or source_path.stat().st_size == 0:
                raise ValueError("上傳圖片不存在或為空檔案。")
            if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError("僅支援 PNG、JPG 或 JPEG 完整胸腔 X 光。")
            try:
                with Image.open(source_path) as opened:
                    opened.load()
                    image = opened.copy()
            except Exception as exc:
                raise ValueError("無法讀取上傳圖片，請確認檔案格式與內容。") from exc
            image_sha = sha256_file(source_path)
            filename = source_path.name
        elif isinstance(image_input, Image.Image):
            image = image_input.copy()
            image_sha = _hash_pil(image)
            filename = "uploaded_image.png"
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
            image_sha = _hash_pil(image)
            filename = "uploaded_image.png"
        else:
            raise ValueError("請先上傳完整胸腔 X 光圖片。")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("圖片尺寸無效。")
        metadata = {
            "image_filename": filename,
            "image_path": str(source_path) if source_path else None,
            "image_sha256": image_sha,
            "image_sha256_short": image_sha[:12],
            "original_width": width,
            "original_height": height,
            "original_mode": image.mode,
            "model_input_width": 224,
            "model_input_height": 224,
        }
        return image, metadata, source_path

    def _lookup_ground_truth(
        self, image_sha: str, filename: str
    ) -> tuple[dict[str, str] | None, str | None]:
        row = self._catalog_by_sha.get(image_sha.lower())
        if row is not None:
            return row, "sha256"
        image_id = Path(filename).stem
        row = self._catalog_by_image_id.get(image_id)
        if row is not None:
            return row, "image_id"
        return None, None

    def predict(self, image_input: Any) -> dict[str, Any]:
        total_started = time.perf_counter()
        image, metadata, source_path = self._open_input(image_input)
        source_hash_before = metadata["image_sha256"]
        preprocess_started = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0)
        preprocessing_seconds = time.perf_counter() - preprocess_started
        if list(tensor.shape) != [1, 3, 224, 224] or not torch.isfinite(tensor).all():
            raise RuntimeError("Full-image preprocessing did not produce finite [1,3,224,224] input")

        batch = tensor.to(self.device)
        with self._inference_lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(batch)
                probability_tensor = torch.sigmoid(logits)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            classification_seconds = time.perf_counter() - inference_started
            self._inference_count += 1

        if list(logits.shape) != [1, 5] or list(probability_tensor.shape) != [1, 5]:
            raise RuntimeError(
                f"Unexpected output shapes: logits={list(logits.shape)}, probabilities={list(probability_tensor.shape)}"
            )
        if not torch.isfinite(probability_tensor).all():
            raise RuntimeError("Model probability output contains NaN or Inf")
        probabilities = [float(value) for value in probability_tensor[0].cpu().tolist()]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise RuntimeError("A model probability is outside [0,1]")
        predicted_vector = [
            int(probabilities[index] >= self.thresholds[index]) for index in range(5)
        ]
        predicted_ids = [index for index, value in enumerate(predicted_vector) if value]

        catalog_row, match_method = self._lookup_ground_truth(
            metadata["image_sha256"], metadata["image_filename"]
        )
        if catalog_row is None:
            truth_vector = None
            truth_ids: list[int] | None = None
        else:
            truth_vector = [
                int(catalog_row[f"label_{index}_{slug}"])
                for index, slug in enumerate(
                    ("aortic_enlargement", "cardiomegaly", "pleural_thickening", "pulmonary_fibrosis", "pleural_effusion")
                )
            ]
            truth_ids = [index for index, value in enumerate(truth_vector) if value]
        metrics = sample_metrics(predicted_vector, truth_vector)

        if source_path is not None and sha256_file(source_path) != source_hash_before:
            raise RuntimeError("Uploaded source image changed during inference")
        total_seconds = time.perf_counter() - total_started
        probability_rows = [
            {
                "class_id": index,
                "class_name_en": CLASS_MAPPING_EN[index],
                "class_name_zh": CLASS_MAPPING_ZH[index],
                "probability": probabilities[index],
                "threshold": self.thresholds[index],
                "decision": "Positive" if predicted_vector[index] else "Negative",
            }
            for index in range(5)
        ]
        return {
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            **metadata,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "threshold_path": str(self.threshold_path),
            "threshold_sha256": self.threshold_sha256,
            "architecture": self.architecture,
            "initialization": self.initialization,
            "device": str(self.device),
            "preprocessing_description": PREPROCESSING_DESCRIPTION,
            "input_tensor_shape": [1, 3, 224, 224],
            "logits_shape": [1, 5],
            "probabilities_shape": [1, 5],
            "predicted_vector_shape": [1, 5],
            "class_probabilities": probabilities,
            "class_thresholds": list(self.thresholds),
            "probability_rows": probability_rows,
            "predicted_label_vector": predicted_vector,
            "predicted_class_ids": predicted_ids,
            "predicted_class_names_en": _names(predicted_ids, CLASS_MAPPING_EN),
            "predicted_class_names_zh": _names(predicted_ids, CLASS_MAPPING_ZH),
            "no_positive_message": NO_POSITIVE_MESSAGE if not predicted_ids else None,
            "ground_truth_catalog_match": catalog_row is not None,
            "ground_truth_match_method": match_method,
            "ground_truth_image_id": catalog_row.get("image_id") if catalog_row else None,
            "ground_truth_label_vector": truth_vector,
            "ground_truth_class_ids": truth_ids,
            "ground_truth_class_names_en": _names(truth_ids, CLASS_MAPPING_EN) if truth_ids is not None else None,
            "ground_truth_class_names_zh": _names(truth_ids, CLASS_MAPPING_ZH) if truth_ids is not None else None,
            "correctly_detected_class_ids": metrics["correctly_detected_ids"],
            "correctly_detected_labels": _names(metrics["correctly_detected_ids"], CLASS_MAPPING_EN) if metrics["correctly_detected_ids"] is not None else None,
            "missed_class_ids": metrics["missed_ids"],
            "missed_labels": _names(metrics["missed_ids"], CLASS_MAPPING_EN) if metrics["missed_ids"] is not None else None,
            "extra_class_ids": metrics["extra_ids"],
            "extra_predicted_labels": _names(metrics["extra_ids"], CLASS_MAPPING_EN) if metrics["extra_ids"] is not None else None,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "exact_match": metrics["exact_match"],
            "sample_precision": metrics["sample_precision"],
            "sample_recall": metrics["sample_recall"],
            "sample_f1": metrics["sample_f1"],
            "preprocessing_seconds": preprocessing_seconds,
            "classification_seconds": classification_seconds,
            "total_model_seconds": total_seconds,
            "model_load_count": self.model_load_count,
            "inference_count": self.inference_count,
            "uses_bbox": False,
            "uses_roi_crop": False,
            "uses_yolo": False,
            "uses_softmax": False,
            "uses_sigmoid": True,
            "optimizer_created": False,
            "backward_executed": False,
            "test_images_read_count": 0,
            "source_image_unchanged": True,
            "disclaimer": DISCLAIMER,
        }

    def ollama_payload(self, prediction: dict[str, Any]) -> dict[str, Any]:
        return {
            "class_probabilities": list(prediction["class_probabilities"]),
            "class_thresholds": list(prediction["class_thresholds"]),
            "predicted_class_ids": list(prediction["predicted_class_ids"]),
            "predicted_class_names_en": list(prediction["predicted_class_names_en"]),
            "predicted_class_names_zh": list(prediction["predicted_class_names_zh"]),
            "preprocessing_description": PREPROCESSING_DESCRIPTION,
            "model_name": "Full-image ConvNeXt-Tiny multilabel classifier",
            "disclaimer": DISCLAIMER,
        }


def get_inference_service(
    model_path: Path,
    threshold_path: Path,
    device: str = "auto",
    catalog_path: Path | None = None,
) -> FullImageMultilabelInferenceService:
    key = (
        str(model_path.expanduser().resolve()).casefold(),
        str(threshold_path.expanduser().resolve()).casefold(),
        device.strip().lower(),
        str(catalog_path.expanduser().resolve()).casefold() if catalog_path else "",
    )
    with _singleton_lock:
        if key not in _instances:
            _instances[key] = FullImageMultilabelInferenceService(
                model_path, threshold_path, device, catalog_path
            )
        return _instances[key]


def create_probability_figure(prediction: dict[str, Any]) -> plt.Figure:
    probabilities = prediction["class_probabilities"]
    thresholds = prediction["class_thresholds"]
    predicted = set(prediction["predicted_class_ids"])
    figure, axis = plt.subplots(figsize=(9.2, 4.8), facecolor="white")
    x = np.arange(5)
    colors = ["#16866f" if index in predicted else "#66839b" for index in range(5)]
    bars = axis.bar(x, probabilities, width=0.64, color=colors)
    axis.scatter(x, thresholds, marker="_", s=760, linewidths=3, color="#d1495b", zorder=4)
    for index, bar in enumerate(bars):
        near_top = probabilities[index] > 0.9
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            probabilities[index] - 0.025 if near_top else probabilities[index] + 0.025,
            f"{probabilities[index]:.4f}\n{'Positive' if index in predicted else 'Negative'}",
            ha="center",
            va="top" if near_top else "bottom",
            fontsize=9,
            color="white" if near_top else "black",
            fontweight="bold" if index in predicted else "normal",
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Independent Sigmoid probability")
    axis.set_xticks(x, ["0 Aortic", "1 Cardio", "2 Pleural T.", "3 Fibrosis", "4 Effusion"])
    axis.set_title("Five-class probabilities and Validation thresholds")
    axis.grid(axis="y", alpha=0.2)
    axis.plot([], [], color="#d1495b", linewidth=3, label="Validation threshold")
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure
