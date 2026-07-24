from __future__ import annotations

import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS = ("train", "val", "test")
CLASS_ORDER = [
    ("Aortic enlargement", 0),
    ("Cardiomegaly", 1),
    ("Pleural thickening", 2),
    ("Pulmonary fibrosis", 3),
    ("Pleural effusion", 4),
]
CLASS_ID_TO_NAME = {class_id: class_name for class_name, class_id in CLASS_ORDER}
CLASS_DIR_NAMES = {
    0: "0_aortic_enlargement",
    1: "1_cardiomegaly",
    2: "2_pleural_thickening",
    3: "3_pulmonary_fibrosis",
    4: "4_pleural_effusion",
}
CLASS_COLORS = {
    0: (255, 80, 80),
    1: (80, 190, 255),
    2: (255, 190, 60),
    3: (90, 220, 130),
    4: (190, 120, 255),
}
SUPPORTED_EXTENSIONS = {".dicom", ".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ""}
IMAGE_SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "images",
]


def combined_annotation_path() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "annotations" / "annotations.csv"


def raw_annotation_path(split: str) -> Path:
    split_path = PROJECT_ROOT / "data" / "raw" / "annotations" / f"{split}_annotations.csv"
    return split_path if split_path.exists() else combined_annotation_path()


def source_annotation_path(split: str) -> Path:
    return raw_annotation_path(split)


def raw_image_dir() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "images"


def metadata_dir() -> Path:
    return PROJECT_ROOT / "data" / "metadata"


def overlay_root() -> Path:
    return PROJECT_ROOT / "data" / "processed" / "bbox_overlay"


def crop_root() -> Path:
    return PROJECT_ROOT / "data" / "processed" / "bbox_crops"


def reports_dir() -> Path:
    return PROJECT_ROOT / "outputs" / "reports"


def image_manifest_path() -> Path:
    return metadata_dir() / "image_manifest.csv"


def crop_manifest_path() -> Path:
    return metadata_dir() / "crop_manifest.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_source_annotation(row: dict[str, str], split: str, original_row_index: int) -> dict[str, Any]:
    class_id = parse_int(row.get("class_id"))
    return {
        "image_id": row.get("image_id", ""),
        "class_id": class_id,
        "class_name": row.get("class_name", CLASS_ID_TO_NAME.get(class_id, "")),
        "x_min": parse_float(row.get("x_min")),
        "y_min": parse_float(row.get("y_min")),
        "x_max": parse_float(row.get("x_max")),
        "y_max": parse_float(row.get("y_max")),
        "rad_id": row.get("rad_id", ""),
        "split": split,
        "original_row_index": original_row_index,
        "annotation_index": original_row_index,
    }


def _normalize_raw_annotation_row(row: dict[str, str], split: str) -> dict[str, Any]:
    class_id = parse_int(row.get("class_id"))
    return {
        "image_id": row.get("image_id", ""),
        "class_id": class_id,
        "class_name": row.get("class_name", CLASS_ID_TO_NAME.get(class_id, "")),
        "x_min": parse_float(row.get("x_min")),
        "y_min": parse_float(row.get("y_min")),
        "x_max": parse_float(row.get("x_max")),
        "y_max": parse_float(row.get("y_max")),
        "rad_id": row.get("rad_id", ""),
        "split": row.get("split", split),
        "original_row_index": parse_int(row.get("original_row_index")),
        "annotation_index": parse_int(row.get("annotation_index")),
    }


def load_raw_annotations() -> list[dict[str, Any]]:
    split_paths = [raw_annotation_path(split) for split in SPLITS]
    if len({path.resolve() for path in split_paths}) == 1:
        path = split_paths[0]
        return [_normalize_raw_annotation_row(row, row.get("split", "")) for row in read_csv_rows(path)]

    rows: list[dict[str, Any]] = []
    for split, path in zip(SPLITS, split_paths):
        for row in read_csv_rows(path):
            rows.append(_normalize_raw_annotation_row(row, split))
    return rows


def build_source_image_index() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    index: dict[str, Path] = {}
    source_rows: list[dict[str, Any]] = []
    for order, root in enumerate(IMAGE_SEARCH_DIRS, start=1):
        exists = root.exists()
        indexed = 0
        if exists:
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    path = Path(dirpath) / filename
                    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    indexed += 1
                    index.setdefault(path.stem, path)
        source_rows.append({"search_order": order, "path": str(root), "exists": exists, "indexed_files": indexed})
    return index, source_rows


def load_image_manifest() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(image_manifest_path())
    return {row["source_image_id"]: row for row in rows}


def dicom_to_uint8(path: Path):
    try:
        import numpy as np
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as exc:
        raise RuntimeError("DICOM loading requires numpy and pydicom.") from exc

    ds = pydicom.dcmread(str(path))
    pixel_array = ds.pixel_array
    try:
        array = apply_voi_lut(pixel_array, ds).astype(np.float32)
    except Exception:
        array = pixel_array.astype(np.float32)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] not in (3, 4):
        array = array[0]

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        array = np.max(array) - array

    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape[:2], dtype=np.uint8)

    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if max_value <= min_value:
        return np.zeros(array.shape[:2], dtype=np.uint8)

    scaled = (array - min_value) / (max_value - min_value) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def load_image(path: Path):
    from PIL import Image

    if path.suffix.lower() in {".dicom", ".dcm"}:
        return Image.fromarray(dicom_to_uint8(path)).convert("RGB")
    return Image.open(path).convert("RGB")


def copy_or_convert_source_image(source_path: Path, destination_path: Path) -> tuple[int, int, str]:
    from PIL import Image

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() in {".dicom", ".dcm"}:
        image = load_image(source_path)
        image.save(destination_path.with_suffix(".png"), format="PNG")
        return image.size[0], image.size[1], str(destination_path.with_suffix(".png"))

    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        destination = destination_path.with_suffix(".png")
        rgb.save(destination, format="PNG")
        return rgb.size[0], rgb.size[1], str(destination)


def duplicate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["image_id"], row["class_id"], row["x_min"], row["y_min"], row["x_max"], row["y_max"])


def clamp_bbox(row: dict[str, Any], width: int, height: int, margin_ratio: float = 0.0) -> dict[str, Any]:
    x_min = float(row["x_min"])
    y_min = float(row["y_min"])
    x_max = float(row["x_max"])
    y_max = float(row["y_max"])
    margin_x = (x_max - x_min) * margin_ratio
    margin_y = (y_max - y_min) * margin_ratio

    expanded_x_min = x_min - margin_x
    expanded_y_min = y_min - margin_y
    expanded_x_max = x_max + margin_x
    expanded_y_max = y_max + margin_y

    clamped_x_min = max(0.0, min(expanded_x_min, float(width - 1)))
    clamped_y_min = max(0.0, min(expanded_y_min, float(height - 1)))
    clamped_x_max = max(1.0, min(expanded_x_max, float(width)))
    clamped_y_max = max(1.0, min(expanded_y_max, float(height)))

    left = int(math.floor(clamped_x_min))
    top = int(math.floor(clamped_y_min))
    right = int(math.ceil(clamped_x_max))
    bottom = int(math.ceil(clamped_y_max))
    return {
        "clamped_x_min": left,
        "clamped_y_min": top,
        "clamped_x_max": right,
        "clamped_y_max": bottom,
        "crop_width": max(0, right - left),
        "crop_height": max(0, bottom - top),
    }


def validate_bbox(row: dict[str, Any], width: int, height: int, margin_ratio: float = 0.0) -> tuple[bool, str, bool, dict[str, Any]]:
    values = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
        return False, "nan_or_non_finite_coordinate", False, {}

    x_min = float(row["x_min"])
    y_min = float(row["y_min"])
    x_max = float(row["x_max"])
    y_max = float(row["y_max"])
    if x_max <= x_min or y_max <= y_min:
        return False, "non_positive_bbox_area", False, {}
    if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
        return False, "bbox_completely_outside_image", True, {}

    crop_box = clamp_bbox(row, width, height, margin_ratio=margin_ratio)
    if crop_box["crop_width"] <= 0 or crop_box["crop_height"] <= 0:
        return False, "empty_crop_after_clamp", True, crop_box

    needs_clamp = x_min < 0 or y_min < 0 or x_max > width or y_max > height
    return True, "", needs_clamp, crop_box


def ensure_class_dirs(root: Path) -> None:
    for split in SPLITS:
        for class_id, dirname in CLASS_DIR_NAMES.items():
            (root / split / dirname).mkdir(parents=True, exist_ok=True)


def summarize_split_class(rows: Iterable[dict[str, Any]], image_key: str, row_label: str) -> list[dict[str, Any]]:
    rows = list(rows)
    image_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        key = (row["split"], int(row["class_id"]))
        image_sets[key].add(str(row[image_key]))
        counts[key] += 1

    summary = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            key = (split, class_id)
            summary.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image_count": len(image_sets[key]),
                    row_label: counts[key],
                }
            )
    return summary

