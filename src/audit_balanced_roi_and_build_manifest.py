
#!/usr/bin/env python
"""Audit balanced 224x224 ROI images and build a deterministic manifest.

Step 1 for RAD-DINO feature cache preparation. This script does not run any
model forward pass and does not modify source image folders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import PIL

CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}

CLASS_FOLDERS = {
    0: "0_aortic_enlargement",
    1: "1_cardiomegaly",
    2: "2_pleural_thickening",
    3: "3_pulmonary_fibrosis",
    4: "4_pleural_effusion",
}

EXPECTED_PER_CLASS = 945
EXPECTED_TOTAL = EXPECTED_PER_CLASS * 5
EXPECTED_SIZE = (224, 224)
AUG_MARKER = "__aug_brightness_"
FILENAME_RE = re.compile(r"^(?P<source_image_id>.+)_class(?P<class_id>[0-4])_rad(?P<rad_id>[^_]+)_bbox(?P<bbox_index>\d{4})(?P<aug>__aug_brightness_\d{4}_f\d{3})?\.png$")

MANIFEST_FIELDS = [
    "feature_index",
    "image_path",
    "relative_path",
    "filename",
    "class_id",
    "class_name",
    "source_image_id",
    "original_roi_id",
    "original_roi_path",
    "is_brightness_augmented",
    "file_size_bytes",
    "image_width",
    "image_height",
    "image_mode",
    "image_sha256",
]

UNRESOLVED_FIELDS = [
    "image_path",
    "relative_path",
    "filename",
    "class_id",
    "class_name",
    "source_image_id",
    "original_roi_id",
    "is_brightness_augmented",
    "error_reason",
]

DUP_FIELDS = ["image_sha256", "duplicate_count", "image_paths"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text, encoding="utf-8")


def atomic_write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def get_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def parse_filename(path: Path, class_id_from_folder: int) -> Tuple[dict, str | None]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {}, "filename_parse_failed"
    parsed = match.groupdict()
    parsed_class_id = int(parsed["class_id"])
    if parsed_class_id != class_id_from_folder:
        return parsed, "class_id_folder_filename_mismatch"
    stem = path.stem
    is_aug = AUG_MARKER in stem
    original_roi_id = stem.split(AUG_MARKER, 1)[0] if is_aug else stem
    parsed["class_id"] = parsed_class_id
    parsed["is_brightness_augmented"] = is_aug
    parsed["original_roi_id"] = original_roi_id
    return parsed, None


def collect_source_metadata(project_root: Path) -> dict:
    metadata_dir = project_root / "data" / "metadata"
    audit_path = metadata_dir / "full_image_dataset_audit.csv"
    manifest_path = metadata_dir / "image_manifest.csv"
    sources_path = metadata_dir / "image_sources.csv"
    metadata = {
        "audit_path": str(audit_path),
        "image_manifest_path": str(manifest_path),
        "image_sources_path": str(sources_path),
        "audit_rows": 0,
        "image_manifest_rows": 0,
        "image_sources_rows": 0,
        "audit_image_ids": set(),
        "manifest_image_ids": set(),
    }
    if audit_path.is_file():
        audit_rows = read_csv_rows(audit_path)
        metadata["audit_rows"] = len(audit_rows)
        metadata["audit_image_ids"] = {row.get("image_id", "").strip() for row in audit_rows if row.get("image_id")}
    if manifest_path.is_file():
        image_manifest_rows = read_csv_rows(manifest_path)
        metadata["image_manifest_rows"] = len(image_manifest_rows)
        metadata["manifest_image_ids"] = {row.get("source_image_id", "").strip() for row in image_manifest_rows if row.get("source_image_id")}
    if sources_path.is_file():
        metadata["image_sources_rows"] = len(read_csv_rows(sources_path))
    return metadata


def audit_and_build(project_root: Path) -> dict:
    balanced_root = project_root / "outputs" / "roi_balanced_224" / "balanced_945_seed42"
    original_roi_root = project_root / "data" / "processed" / "bbox_crops_224"
    output_root = project_root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42"

    if not balanced_root.is_dir():
        raise FileNotFoundError(f"Balanced ROI dataset not found: {balanced_root}")
    if not original_roi_root.is_dir():
        raise FileNotFoundError(f"Original ROI folder not found: {original_roi_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_metadata = collect_source_metadata(project_root)
    audit_image_ids = source_metadata["audit_image_ids"]
    manifest_image_ids = source_metadata["manifest_image_ids"]

    records: List[dict] = []
    unresolved: List[dict] = []
    non_png_file_count = 0
    zero_byte_file_count = 0
    unreadable_image_count = 0
    wrong_size_count = 0
    wrong_mode_count = 0
    missing_class_id_count = 0
    invalid_class_id_count = 0
    missing_source_image_id_count = 0
    missing_original_roi_id_count = 0
    augmented_without_source_count = 0
    source_image_id_not_in_metadata_count = 0

    all_abs_paths: List[str] = []
    all_rel_paths: List[str] = []
    filename_within_class: Dict[int, List[str]] = defaultdict(list)
    sha_to_paths: Dict[str, List[str]] = defaultdict(list)
    per_class_count = Counter()
    original_image_count = 0
    augmented_image_count = 0

    for class_id in sorted(CLASS_FOLDERS):
        folder = CLASS_FOLDERS[class_id]
        class_dir = balanced_root / folder
        if not class_dir.is_dir():
            unresolved.append({
                "image_path": str(class_dir),
                "relative_path": folder,
                "filename": "",
                "class_id": class_id,
                "class_name": CLASS_MAPPING[class_id],
                "source_image_id": "",
                "original_roi_id": "",
                "is_brightness_augmented": "",
                "error_reason": "missing_class_folder",
            })
            continue

        for child in sorted(class_dir.iterdir(), key=lambda p: p.name):
            if not child.is_file():
                continue
            if child.suffix.lower() != ".png":
                non_png_file_count += 1
                unresolved.append({
                    "image_path": str(child),
                    "relative_path": child.relative_to(balanced_root).as_posix(),
                    "filename": child.name,
                    "class_id": class_id,
                    "class_name": CLASS_MAPPING[class_id],
                    "source_image_id": "",
                    "original_roi_id": "",
                    "is_brightness_augmented": "",
                    "error_reason": "non_png_file",
                })
                continue

            parsed, parse_error = parse_filename(child, class_id)
            rel_path = child.relative_to(balanced_root).as_posix()
            abs_path = str(child.resolve())
            class_name = CLASS_MAPPING[class_id]
            source_image_id = parsed.get("source_image_id", "") if parsed else ""
            original_roi_id = parsed.get("original_roi_id", "") if parsed else ""
            is_aug = bool(parsed.get("is_brightness_augmented", False)) if parsed else False
            original_roi_path = original_roi_root / folder / f"{original_roi_id}.png" if original_roi_id else Path("")

            errors: List[str] = []
            if parse_error:
                errors.append(parse_error)
            if class_id not in CLASS_MAPPING:
                invalid_class_id_count += 1
                errors.append("invalid_class_id")
            if parsed and parsed.get("class_id") is None:
                missing_class_id_count += 1
                errors.append("missing_class_id")
            if not source_image_id:
                missing_source_image_id_count += 1
                errors.append("missing_source_image_id")
            if not original_roi_id:
                missing_original_roi_id_count += 1
                errors.append("missing_original_roi_id")
            if source_image_id and audit_image_ids and source_image_id not in audit_image_ids:
                source_image_id_not_in_metadata_count += 1
                errors.append("source_image_id_not_in_full_image_dataset_audit")
            if source_image_id and manifest_image_ids and source_image_id not in manifest_image_ids:
                source_image_id_not_in_metadata_count += 1
                errors.append("source_image_id_not_in_image_manifest")
            if is_aug and (not original_roi_path.is_file()):
                augmented_without_source_count += 1
                errors.append("augmented_without_source_original_roi")
            if (not is_aug) and original_roi_id and (not original_roi_path.is_file()):
                errors.append("original_roi_path_missing")

            file_size = child.stat().st_size
            if file_size == 0:
                zero_byte_file_count += 1
                errors.append("zero_byte_file")

            width = height = None
            mode = ""
            digest = ""
            try:
                with Image.open(child) as img:
                    img.load()
                    width, height = img.size
                    mode = img.mode
                if (width, height) != EXPECTED_SIZE:
                    wrong_size_count += 1
                    errors.append("wrong_size")
                if mode != "L":
                    wrong_mode_count += 1
                    errors.append("wrong_mode")
                digest = sha256_file(child)
            except Exception as exc:  # noqa: BLE001 - report exact unreadable image failures.
                unreadable_image_count += 1
                errors.append(f"unreadable_image:{exc}")

            if errors:
                unresolved.append({
                    "image_path": str(child),
                    "relative_path": rel_path,
                    "filename": child.name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image_id": source_image_id,
                    "original_roi_id": original_roi_id,
                    "is_brightness_augmented": str(is_aug),
                    "error_reason": ";".join(errors),
                })

            record = {
                "feature_index": -1,
                "image_path": str(child),
                "relative_path": rel_path,
                "filename": child.name,
                "class_id": class_id,
                "class_name": class_name,
                "source_image_id": source_image_id,
                "original_roi_id": original_roi_id,
                "original_roi_path": str(original_roi_path) if original_roi_id else "",
                "is_brightness_augmented": str(is_aug),
                "file_size_bytes": file_size,
                "image_width": width if width is not None else "",
                "image_height": height if height is not None else "",
                "image_mode": mode,
                "image_sha256": digest,
            }
            records.append(record)
            per_class_count[class_id] += 1
            filename_within_class[class_id].append(child.name)
            all_abs_paths.append(abs_path.lower())
            all_rel_paths.append(rel_path.lower())
            if digest:
                sha_to_paths[digest].append(str(child))
            if is_aug:
                augmented_image_count += 1
            else:
                original_image_count += 1

    records.sort(key=lambda row: (int(row["class_id"]), row["filename"]))
    for idx, row in enumerate(records):
        row["feature_index"] = idx

    duplicate_absolute_path_count = len(all_abs_paths) - len(set(all_abs_paths))
    duplicate_relative_path_count = len(all_rel_paths) - len(set(all_rel_paths))
    duplicate_filename_within_class_count = 0
    for names in filename_within_class.values():
        counts = Counter(names)
        duplicate_filename_within_class_count += sum(count - 1 for count in counts.values() if count > 1)

    duplicate_groups = [
        {
            "image_sha256": digest,
            "duplicate_count": len(paths),
            "image_paths": "|".join(sorted(paths)),
        }
        for digest, paths in sorted(sha_to_paths.items())
        if len(paths) > 1
    ]
    duplicate_sha256_group_count = len(duplicate_groups)
    duplicate_sha256_image_count = sum(row["duplicate_count"] for row in duplicate_groups)

    total_png_count = len(records)
    feature_indices = [row["feature_index"] for row in records]
    feature_index_is_contiguous = feature_indices == list(range(len(records)))
    all_classes_equal_expected = all(per_class_count[class_id] == EXPECTED_PER_CLASS for class_id in CLASS_MAPPING)

    summary = {
        "dataset_name": "balanced_945_seed42",
        "source_dataset_path": str(balanced_root),
        "output_root": str(output_root),
        "total_png_count": total_png_count,
        "manifest_row_count": len(records),
        "per_class_count": {str(cid): per_class_count[cid] for cid in sorted(CLASS_MAPPING)},
        "unreadable_image_count": unreadable_image_count,
        "zero_byte_file_count": zero_byte_file_count,
        "wrong_size_count": wrong_size_count,
        "wrong_mode_count": wrong_mode_count,
        "non_png_file_count": non_png_file_count,
        "duplicate_absolute_path_count": duplicate_absolute_path_count,
        "duplicate_relative_path_count": duplicate_relative_path_count,
        "duplicate_filename_within_class_count": duplicate_filename_within_class_count,
        "missing_class_id_count": missing_class_id_count,
        "invalid_class_id_count": invalid_class_id_count,
        "missing_source_image_id_count": missing_source_image_id_count,
        "missing_original_roi_id_count": missing_original_roi_id_count,
        "source_image_id_not_in_metadata_count": source_image_id_not_in_metadata_count,
        "augmented_image_count": augmented_image_count,
        "original_image_count": original_image_count,
        "augmented_without_source_count": augmented_without_source_count,
        "duplicate_sha256_group_count": duplicate_sha256_group_count,
        "duplicate_sha256_image_count": duplicate_sha256_image_count,
        "unresolved_record_count": len(unresolved),
        "feature_index_min": min(feature_indices) if feature_indices else None,
        "feature_index_max": max(feature_indices) if feature_indices else None,
        "feature_index_is_contiguous": feature_index_is_contiguous,
        "all_classes_equal_945": all_classes_equal_expected,
        "teacher_feature_expected_shape": [EXPECTED_TOTAL, 768],
        "status": "PASS",
    }

    failure_keys = [
        "unreadable_image_count",
        "zero_byte_file_count",
        "wrong_size_count",
        "wrong_mode_count",
        "non_png_file_count",
        "duplicate_absolute_path_count",
        "duplicate_relative_path_count",
        "duplicate_filename_within_class_count",
        "missing_class_id_count",
        "invalid_class_id_count",
        "missing_source_image_id_count",
        "missing_original_roi_id_count",
        "source_image_id_not_in_metadata_count",
        "augmented_without_source_count",
        "unresolved_record_count",
    ]
    if total_png_count != EXPECTED_TOTAL or len(records) != EXPECTED_TOTAL:
        summary["status"] = "FAIL"
    if not all_classes_equal_expected or not feature_index_is_contiguous:
        summary["status"] = "FAIL"
    if any(summary[key] != 0 for key in failure_keys):
        summary["status"] = "FAIL"

    manifest_path = output_root / "roi_manifest.csv"
    audit_summary_path = output_root / "audit_summary.json"
    audit_report_path = output_root / "audit_report.txt"
    duplicate_path = output_root / "duplicate_sha256_groups.csv"
    unresolved_path = output_root / "unresolved_records.csv"
    metadata_path = output_root / "manifest_metadata.json"

    atomic_write_csv(manifest_path, MANIFEST_FIELDS, records)
    manifest_sha256 = sha256_file(manifest_path)
    summary["manifest_sha256"] = manifest_sha256
    atomic_write_json(audit_summary_path, summary)
    atomic_write_csv(duplicate_path, DUP_FIELDS, duplicate_groups)
    atomic_write_csv(unresolved_path, UNRESOLVED_FIELDS, unresolved)

    manifest_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "balanced_945_seed42",
        "project_root": str(project_root),
        "source_dataset_path": str(balanced_root),
        "output_root": str(output_root),
        "total_records": len(records),
        "class_mapping": {str(cid): name for cid, name in CLASS_MAPPING.items()},
        "sorting_rule": "class_id ascending, filename ascending within class; feature_index assigned after sorting",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pillow_version": PIL.__version__,
        "git_commit": get_git_commit(project_root),
        "source_metadata": {
            "full_image_dataset_audit_csv": source_metadata["audit_path"],
            "full_image_dataset_audit_rows": source_metadata["audit_rows"],
            "image_manifest_csv": source_metadata["image_manifest_path"],
            "image_manifest_rows": source_metadata["image_manifest_rows"],
            "image_sources_csv": source_metadata["image_sources_path"],
            "image_sources_rows": source_metadata["image_sources_rows"],
        },
        "teacher_feature_expected_shape": [EXPECTED_TOTAL, 768],
        "feature_cache_created": False,
        "model_forward_executed": False,
        "normalization_applied": False,
        "train_val_test_split_created": False,
        "audit_status": summary["status"],
    }
    atomic_write_json(metadata_path, manifest_metadata)

    report_lines = [
        "Balanced ROI 224 Step 1 Audit Report",
        f"Status: {summary['status']}",
        f"Project root: {project_root}",
        f"Source dataset: {balanced_root}",
        f"Output root: {output_root}",
        f"Total PNG: {total_png_count}",
        f"Manifest rows: {len(records)}",
        "Per-class counts:",
    ]
    for cid in sorted(CLASS_MAPPING):
        report_lines.append(f"  class {cid} {CLASS_MAPPING[cid]}: {per_class_count[cid]}")
    report_lines.extend([
        f"Original images: {original_image_count}",
        f"Brightness augmented images: {augmented_image_count}",
        f"Unreadable: {unreadable_image_count}",
        f"Zero byte: {zero_byte_file_count}",
        f"Wrong size: {wrong_size_count}",
        f"Wrong mode: {wrong_mode_count}",
        f"Non-PNG: {non_png_file_count}",
        f"Duplicate absolute path: {duplicate_absolute_path_count}",
        f"Duplicate relative path: {duplicate_relative_path_count}",
        f"Duplicate filename within class: {duplicate_filename_within_class_count}",
        f"Duplicate SHA256 groups: {duplicate_sha256_group_count}",
        f"Duplicate SHA256 images: {duplicate_sha256_image_count}",
        f"Missing source_image_id: {missing_source_image_id_count}",
        f"Missing original_roi_id: {missing_original_roi_id_count}",
        f"Augmented without source: {augmented_without_source_count}",
        f"Unresolved records: {len(unresolved)}",
        f"Feature index contiguous 0..{EXPECTED_TOTAL - 1}: {feature_index_is_contiguous}",
        f"Manifest SHA256: {manifest_sha256}",
        "RAD-DINO forward executed: False",
        "Teacher features created: False",
        "Train/val/test split created: False",
        "Normalization applied: False",
    ])
    atomic_write_text(audit_report_path, "\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "paths": {
            "roi_manifest": str(manifest_path),
            "audit_summary": str(audit_summary_path),
            "audit_report": str(audit_report_path),
            "duplicate_sha256_groups": str(duplicate_path),
            "unresolved_records": str(unresolved_path),
            "manifest_metadata": str(metadata_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit balanced 224x224 ROI dataset and build deterministic manifest.")
    parser.add_argument("--project-root", required=True, type=Path, help="Project root, e.g. C:\\Users\\09688\\thoracic-cxr-project-3")
    args = parser.parse_args()

    result = audit_and_build(args.project_root.resolve())
    summary = result["summary"]
    print("STEP1_BALANCED_ROI_MANIFEST_DONE")
    print(f"status={summary['status']}")
    print(f"total_png_count={summary['total_png_count']}")
    print(f"manifest_row_count={summary['manifest_row_count']}")
    print("per_class_count=" + ";".join(f"{cid}:{summary['per_class_count'][cid]}" for cid in sorted(summary["per_class_count"], key=int)))
    print(f"original_image_count={summary['original_image_count']}")
    print(f"augmented_image_count={summary['augmented_image_count']}")
    print(f"missing_source_image_id_count={summary['missing_source_image_id_count']}")
    print(f"unreadable_image_count={summary['unreadable_image_count']}")
    print(f"wrong_size_count={summary['wrong_size_count']}")
    print(f"wrong_mode_count={summary['wrong_mode_count']}")
    print(f"duplicate_absolute_path_count={summary['duplicate_absolute_path_count']}")
    print(f"duplicate_relative_path_count={summary['duplicate_relative_path_count']}")
    print(f"duplicate_sha256_group_count={summary['duplicate_sha256_group_count']}")
    print(f"duplicate_sha256_image_count={summary['duplicate_sha256_image_count']}")
    print(f"augmented_without_source_count={summary['augmented_without_source_count']}")
    print(f"unresolved_record_count={summary['unresolved_record_count']}")
    print(f"manifest_sha256={summary['manifest_sha256']}")
    print(f"feature_index_min={summary['feature_index_min']}")
    print(f"feature_index_max={summary['feature_index_max']}")
    print(f"feature_index_is_contiguous={summary['feature_index_is_contiguous']}")
    for name, path in result["paths"].items():
        print(f"{name}={path}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
