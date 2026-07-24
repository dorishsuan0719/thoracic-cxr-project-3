#!/usr/bin/env python
"""Build the deterministic full-image five-label Master and 8:1:1 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image


CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
EXPECTED_MODEL_SHA256 = "8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5"
LABEL_FIELDS = [
    "label_0_aortic_enlargement",
    "label_1_cardiomegaly",
    "label_2_pleural_thickening",
    "label_3_pulmonary_fibrosis",
    "label_4_pleural_effusion",
]
MASTER_FIELDS = [
    "master_index",
    "image_id",
    "source_image_id",
    "full_image_path",
    "image_filename",
    "image_sha256",
    "original_width",
    "original_height",
    "original_mode",
    *LABEL_FIELDS,
    "label_vector",
    "positive_class_ids",
    "positive_class_names",
    "num_positive_labels",
]
SPLIT_FIELDS = ["split", *MASTER_FIELDS]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_atomic(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    replace_atomic(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    replace_atomic(temporary, path)


def read_annotations(path: Path) -> tuple[list[dict[str, str]], dict[str, set[int]], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"image_id", "class_id", "class_name"}
        if not required.issubset(fields):
            raise ValueError(f"Annotation columns are missing: {sorted(required - fields)}")
        rows = list(reader)
    labels: dict[str, set[int]] = defaultdict(set)
    invalid = []
    for row_number, row in enumerate(rows, start=2):
        image_id = str(row.get("image_id") or "").strip()
        try:
            class_id = int(row.get("class_id", ""))
        except ValueError:
            class_id = -1
        class_name = str(row.get("class_name") or "").strip()
        reason = ""
        if not image_id:
            reason = "missing_image_id"
        elif class_id not in CLASS_MAPPING:
            reason = "class_id_not_in_0_to_4"
        elif class_name != CLASS_MAPPING[class_id]:
            reason = "class_name_mapping_mismatch"
        if reason:
            invalid.append(
                {
                    "row_number": row_number,
                    "image_id": image_id,
                    "class_id": row.get("class_id", ""),
                    "class_name": class_name,
                    "reason": reason,
                }
            )
        else:
            labels[image_id].add(class_id)
    return rows, labels, invalid


def index_images(images_dir: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            grouped[path.stem].append(path.resolve())
    duplicates = []
    index = {}
    for image_id, paths in grouped.items():
        if len(paths) == 1:
            index[image_id] = paths[0]
        else:
            duplicates.append(
                {"image_id": image_id, "paths": [str(path) for path in paths]}
            )
    return index, duplicates


def inspect_image(path: Path) -> dict[str, Any]:
    if path.stat().st_size <= 0:
        raise ValueError("empty image file")
    with Image.open(path) as probe:
        probe.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    if width <= 0 or height <= 0:
        raise ValueError("non-positive image dimensions")
    return {
        "original_width": width,
        "original_height": height,
        "original_mode": mode,
        "image_sha256": sha256_file(path),
    }


def build_master(
    labels: dict[str, set[int]], image_index: dict[str, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    master = []
    missing = []
    unreadable = []
    for image_id in sorted(labels):
        path = image_index.get(image_id)
        if path is None:
            missing.append({"image_id": image_id, "reason": "annotation_without_image"})
            continue
        try:
            properties = inspect_image(path)
        except Exception as exc:
            unreadable.append(
                {"image_id": image_id, "full_image_path": str(path), "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        vector = [int(class_id in labels[image_id]) for class_id in CLASS_MAPPING]
        positives = [index for index, value in enumerate(vector) if value]
        row = {
            "master_index": len(master),
            "image_id": image_id,
            "source_image_id": image_id,
            "full_image_path": str(path),
            "image_filename": path.name,
            **properties,
            **{field: vector[index] for index, field in enumerate(LABEL_FIELDS)},
            "label_vector": "[" + ",".join(map(str, vector)) + "]",
            "positive_class_ids": "|".join(map(str, positives)),
            "positive_class_names": "|".join(CLASS_MAPPING[index] for index in positives),
            "num_positive_labels": len(positives),
        }
        master.append(row)
    return master, missing, unreadable


def iterative_split(master: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    x = np.arange(len(master)).reshape(-1, 1)
    y = np.asarray(
        [[int(row[field]) for field in LABEL_FIELDS] for row in master], dtype=np.int64
    )
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, temporary_indices = next(first.split(x, y))
    temporary_y = y[temporary_indices]
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_local, test_local = next(second.split(x[temporary_indices], temporary_y))
    split_indices = {
        "train": train_indices,
        "val": temporary_indices[val_local],
        "test": temporary_indices[test_local],
    }
    splits = {
        name: [master[int(index)] for index in sorted(indices)]
        for name, indices in split_indices.items()
    }
    return repair_split_sizes(splits, {"train": 472, "val": 59, "test": 59})


def repair_split_sizes(
    splits: dict[str, list[dict[str, Any]]], targets: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically repair iterative-stratification rounding to exact row counts."""
    global_positive = np.asarray(
        [
            sum(int(row[field]) for rows in splits.values() for row in rows)
            for field in LABEL_FIELDS
        ],
        dtype=np.float64,
    )
    total = float(sum(targets.values()))

    def objective(candidate: dict[str, Any], source: str, destination: str) -> float:
        score = 0.0
        vector = np.asarray([int(candidate[field]) for field in LABEL_FIELDS], dtype=np.float64)
        for name, rows in splits.items():
            counts = np.asarray(
                [sum(int(row[field]) for row in rows) for field in LABEL_FIELDS],
                dtype=np.float64,
            )
            if name == source:
                counts -= vector
            elif name == destination:
                counts += vector
            ideal = global_positive * (targets[name] / total)
            score += float(np.square(counts - ideal).sum())
        return score

    while any(len(splits[name]) != targets[name] for name in targets):
        sources = [name for name in targets if len(splits[name]) > targets[name]]
        destinations = [name for name in targets if len(splits[name]) < targets[name]]
        if not sources or not destinations:
            raise RuntimeError("Unable to repair split sizes")
        choices = []
        for source in sorted(sources):
            for destination in sorted(destinations):
                for row in splits[source]:
                    choices.append(
                        (
                            objective(row, source, destination),
                            int(row["num_positive_labels"]),
                            row["image_id"],
                            source,
                            destination,
                            row,
                        )
                    )
        _, _, _, source, destination, selected = min(choices, key=lambda item: item[:5])
        splits[source].remove(selected)
        splits[destination].append(selected)
    for rows in splits.values():
        rows.sort(key=lambda row: int(row["master_index"]))
    return splits


def audit_splits(master: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected = {"train": 472, "val": 59, "test": 59}
    split_sets = {
        name: {
            "image_id": {row["image_id"] for row in rows},
            "sha256": {row["image_sha256"] for row in rows},
        }
        for name, rows in splits.items()
    }
    intersections = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        intersections[f"{left}_{right}_image_id"] = sorted(
            split_sets[left]["image_id"] & split_sets[right]["image_id"]
        )
        intersections[f"{left}_{right}_sha256"] = sorted(
            split_sets[left]["sha256"] & split_sets[right]["sha256"]
        )
    class_counts = {}
    positive_negative = {}
    for name, rows in splits.items():
        class_counts[name] = {
            str(index): sum(int(row[LABEL_FIELDS[index]]) for row in rows)
            for index in CLASS_MAPPING
        }
        positive_negative[name] = {
            str(index): {
                "positive": class_counts[name][str(index)],
                "negative": len(rows) - class_counts[name][str(index)],
            }
            for index in CLASS_MAPPING
        }
    duplicate_paths = [
        path for path, count in Counter(row["full_image_path"] for row in master).items() if count > 1
    ]
    duplicate_hashes = [
        digest for digest, count in Counter(row["image_sha256"] for row in master).items() if count > 1
    ]
    failures = []
    if len(master) != 590:
        failures.append("master_rows")
    if any(len(splits[name]) != count for name, count in expected.items()):
        failures.append("split_rows")
    if any(intersections.values()):
        failures.append("split_leakage")
    if duplicate_paths:
        failures.append("duplicate_paths")
    if duplicate_hashes:
        failures.append("duplicate_content")
    for split_name, values in positive_negative.items():
        for class_id, counts in values.items():
            if counts["positive"] <= 0 or counts["negative"] <= 0:
                failures.append(f"{split_name}_class_{class_id}_positive_negative")
    return {
        "status": "PASS" if not failures else "FAIL",
        "master_rows": len(master),
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "expected_split_rows": expected,
        "intersections": intersections,
        "class_positive_counts": class_counts,
        "positive_negative_counts": positive_negative,
        "duplicate_full_image_paths": duplicate_paths,
        "duplicate_sha256": duplicate_hashes,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    annotations = args.annotations.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not project_root.is_dir() or not images_dir.is_dir() or not annotations.is_file():
        raise FileNotFoundError("Project, images, or annotations source is missing")
    forbidden_sources = [
        project_root / "data" / "processed" / "bbox_crops",
        project_root / "data" / "processed" / "bbox_crops_224",
        project_root / "outputs" / "roi_balanced_224",
    ]
    if any(source == images_dir or source in images_dir.parents for source in forbidden_sources):
        raise ValueError("ROI-derived images cannot be used as full-image source")
    if args.image_size != 224:
        raise ValueError("This locked experiment requires --image-size 224")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    annotation_rows, labels, invalid_labels = read_annotations(annotations)
    image_index, duplicate_ids = index_images(images_dir)
    master, missing, unreadable = build_master(labels, image_index)
    image_without_annotation = sorted(set(image_index) - set(labels))
    splits = iterative_split(master, args.seed) if len(master) == 590 else {"train": [], "val": [], "test": []}
    split_audit = audit_splits(master, splits) if len(master) == 590 else {"status": "FAIL"}
    positive_counts = {
        str(index): sum(int(row[LABEL_FIELDS[index]]) for row in master)
        for index in CLASS_MAPPING
    }
    failures = []
    expected_positive = {str(index): 350 for index in CLASS_MAPPING}
    if len(annotation_rows) != 4546:
        failures.append("annotation_rows_not_4546")
    if len(labels) != 590 or len(image_index) != 590 or len(master) != 590:
        failures.append("full_image_or_master_rows_not_590")
    if positive_counts != expected_positive:
        failures.append("positive_counts_not_350_each")
    if invalid_labels:
        failures.append("invalid_labels")
    if duplicate_ids:
        failures.append("duplicate_image_ids")
    if missing or unreadable or image_without_annotation:
        failures.append("image_annotation_integrity")
    if split_audit.get("status") != "PASS":
        failures.append("split_audit")
    dry_run_audit = {
        "status": "PASS" if not failures else "FAIL",
        "dry_run": args.dry_run,
        "created_at": utc_now(),
        "project_root": str(project_root),
        "full_image_source": str(images_dir),
        "annotations": str(annotations),
        "annotation_rows": len(annotation_rows),
        "unique_annotation_image_ids": len(labels),
        "image_files": len(image_index),
        "master_rows": len(master),
        "positive_image_counts": positive_counts,
        "invalid_label_rows": len(invalid_labels),
        "duplicate_image_ids": len(duplicate_ids),
        "missing_images": len(missing),
        "unreadable_images": len(unreadable),
        "images_without_annotation": len(image_without_annotation),
        "split_audit": split_audit,
        "seed": args.seed,
        "image_size": args.image_size,
        "uses_bbox": False,
        "uses_roi_crop": False,
        "center_crop": False,
        "augmentation": False,
        "model_inference": False,
        "failures": failures,
    }
    print(json.dumps(dry_run_audit, ensure_ascii=False, indent=2))
    if failures:
        return 1
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_csv(output_dir / "full_image_multilabel_master_manifest.csv", master, MASTER_FIELDS)
    for split_name in ("train", "val", "test"):
        split_rows = [{"split": split_name, **row} for row in splits[split_name]]
        atomic_write_csv(output_dir / f"{split_name}_manifest.csv", split_rows, SPLIT_FIELDS)
    preprocessing = {
        "task": "full-image multilabel five-class classification",
        "steps": [
            "PIL open complete chest X-ray",
            "Convert RGB",
            "Resize complete image directly to 224x224",
            "ToTensor",
            "ImageNet mean/std normalization",
        ],
        "resize": {"size": [224, 224], "interpolation": "BILINEAR", "antialias": True},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "expected_tensor_shape": [3, 224, 224],
        "center_crop": False,
        "bbox": False,
        "roi_crop": False,
        "augmentation": False,
        "raw_images_modified": False,
    }
    atomic_write_json(output_dir / "preprocessing_spec.json", preprocessing)
    atomic_write_json(output_dir / "dataset_integrity_audit.json", dry_run_audit)
    atomic_write_json(output_dir / "split_leakage_audit.json", split_audit)
    dataset_metadata = {
        "status": "PASS",
        "created_at": utc_now(),
        "seed": args.seed,
        "master_manifest_sha256": sha256_file(output_dir / "full_image_multilabel_master_manifest.csv"),
        "split_manifest_sha256": {
            name: sha256_file(output_dir / f"{name}_manifest.csv")
            for name in ("train", "val", "test")
        },
        "class_mapping": CLASS_MAPPING,
        "label_fields": LABEL_FIELDS,
        "split_method": "two-stage MultilabelStratifiedShuffleSplit",
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "positive_image_counts": positive_counts,
    }
    atomic_write_json(output_dir / "dataset_metadata.json", dataset_metadata)
    residual = list(output_dir.rglob("*.tmp")) + list(output_dir.rglob("*.writing"))
    if residual:
        raise RuntimeError(f"Temporary output files remain: {residual}")
    print(f"Created full-image multilabel dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
