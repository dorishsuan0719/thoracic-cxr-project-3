from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_DIR_NAMES, PROJECT_ROOT, metadata_dir, reports_dir, write_csv_rows

EXPECTED_TOTAL = 2343
EXPECTED_CLASS_COUNTS = {0: 386, 1: 417, 2: 355, 3: 471, 4: 714}
FORBIDDEN_SPLIT_DIRS = {"train", "val", "test"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_rows(path, fieldnames, rows)


def norm_path(value: str) -> str:
    return str(Path(value).resolve()).casefold() if value else ""


def error_row(path: str, error_type: str, detail: str = "") -> dict[str, Any]:
    return {"path": path, "error_type": error_type, "error_detail": detail}


def validate_folder(folder: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv_rows(manifest_path)
    errors: list[dict[str, Any]] = []
    expected_dirs = set(CLASS_DIR_NAMES.values())
    actual_dirs = {path.name for path in folder.iterdir() if path.is_dir()} if folder.exists() else set()

    if not folder.exists():
        errors.append(error_row(str(folder), "folder_missing"))
    unexpected_dirs = sorted(actual_dirs - expected_dirs)
    missing_dirs = sorted(expected_dirs - actual_dirs)
    forbidden_dirs = sorted(actual_dirs & FORBIDDEN_SPLIT_DIRS)
    for name in unexpected_dirs:
        errors.append(error_row(str(folder / name), "unexpected_class_folder"))
    for name in missing_dirs:
        errors.append(error_row(str(folder / name), "missing_class_folder"))
    for name in forbidden_dirs:
        errors.append(error_row(str(folder / name), "forbidden_split_folder"))

    png_paths = list(folder.rglob("*.png")) if folder.exists() else []
    duplicate_manifest_paths = 0
    duplicate_source_annotation = 0
    path_counts = Counter(norm_path(row["master_image_path"]) for row in rows)
    source_annotation_counts = Counter((row["source_image_id"], row["annotation_index"]) for row in rows)
    duplicate_manifest_paths = sum(1 for count in path_counts.values() if count > 1)
    duplicate_source_annotation = sum(1 for count in source_annotation_counts.values() if count > 1)
    if duplicate_manifest_paths:
        errors.append(error_row(str(manifest_path), "duplicate_manifest_image_path", str(duplicate_manifest_paths)))
    if duplicate_source_annotation:
        errors.append(error_row(str(manifest_path), "duplicate_source_image_annotation_index", str(duplicate_source_annotation)))

    missing = unreadable = empty = wrong_size = 0
    class_counts: Counter[int] = Counter()
    for row in rows:
        image_path = Path(row["master_image_path"])
        class_id = int(row["class_id"])
        class_counts[class_id] += 1
        if not image_path.exists():
            missing += 1
            errors.append(error_row(str(image_path), "missing_image"))
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                gray = image.convert("L")
                if gray.size != (224, 224):
                    wrong_size += 1
                    errors.append(error_row(str(image_path), "wrong_size", f"{gray.size[0]}x{gray.size[1]}"))
                if gray.getextrema() == (0, 0):
                    empty += 1
                    errors.append(error_row(str(image_path), "empty_all_zero"))
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            errors.append(error_row(str(image_path), "unreadable_image", str(exc)))

    for class_id, expected in EXPECTED_CLASS_COUNTS.items():
        if class_counts[class_id] != expected:
            errors.append(error_row(str(manifest_path), "class_count_mismatch", f"class{class_id}: {class_counts[class_id]} != {expected}"))

    metrics = {
        "manifest_rows": len(rows),
        "png_count": len(png_paths),
        "missing": missing,
        "unreadable": unreadable,
        "empty": empty,
        "wrong_size": wrong_size,
        "duplicate_image_path": duplicate_manifest_paths,
        "duplicate_source_image_annotation_index": duplicate_source_annotation,
        "actual_top_level_dirs": ",".join(sorted(actual_dirs)),
        "forbidden_split_dirs_present": ",".join(forbidden_dirs),
        "class0_count": class_counts[0],
        "class1_count": class_counts[1],
        "class2_count": class_counts[2],
        "class3_count": class_counts[3],
        "class4_count": class_counts[4],
    }
    return metrics, errors


def choose_backup_path(processed_dir: Path) -> Path:
    base = processed_dir / "bbox_crops_224_legacy_split_backup"
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return processed_dir / f"bbox_crops_224_legacy_split_backup_{stamp}"


def update_manifest_paths(rows: list[dict[str, str]], old_root: Path, new_root: Path) -> list[dict[str, str]]:
    updated = []
    old_text = str(old_root)
    new_text = str(new_root)
    for row in rows:
        new_row = row.copy()
        new_row["master_image_path"] = new_row["master_image_path"].replace(old_text, new_text)
        updated.append(new_row)
    return updated


def update_version_json(version_path: Path) -> None:
    data = json.loads(version_path.read_text(encoding="utf-8"))
    data["official_image_folder"] = "data/processed/bbox_crops_224"
    data["folder_structure"] = "class_only"
    data["preprocessing_split_applied"] = False
    data["official_training_split_created"] = False
    version_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_validation_report(metrics: dict[str, Any], errors: list[dict[str, Any]], backup_path: Path | None) -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    with (reports_dir() / "roi_224_final_folder_validation.txt").open("w", encoding="utf-8") as f:
        f.write("ROI 224 final folder validation\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"error_rows: {len(errors)}\n")
        f.write(f"legacy_backup_path: {backup_path or ''}\n")
    write_csv_rows(reports_dir() / "roi_224_final_folder_errors.csv", ["path", "error_type", "error_detail"], errors)


def main() -> int:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    legacy_dir = processed_dir / "bbox_crops_224"
    master_dir = processed_dir / "bbox_crops_224_master"
    official_dir = processed_dir / "bbox_crops_224"
    master_manifest_path = metadata_dir() / "roi_224_master_manifest.csv"
    official_manifest_path = metadata_dir() / "roi_224_manifest.csv"
    version_path = metadata_dir() / "dataset_version_roi_224_master_v1.json"

    pre_metrics, pre_errors = validate_folder(master_dir, master_manifest_path)
    if (
        pre_errors
        or pre_metrics["manifest_rows"] != EXPECTED_TOTAL
        or pre_metrics["png_count"] != EXPECTED_TOTAL
        or pre_metrics["forbidden_split_dirs_present"]
    ):
        write_validation_report(pre_metrics, pre_errors, None)
        print("Pre-rename validation failed. No folders were renamed.")
        print(f"error_rows: {len(pre_errors)}")
        return 1

    backup_path = choose_backup_path(processed_dir)
    if legacy_dir.exists():
        legacy_dir.rename(backup_path)
    else:
        backup_path = None
    master_dir.rename(official_dir)

    original_rows = read_csv_rows(master_manifest_path)
    updated_rows = update_manifest_paths(original_rows, master_dir, official_dir)
    write_csv(master_manifest_path, updated_rows)
    write_csv(official_manifest_path, updated_rows)
    update_version_json(version_path)

    final_metrics, final_errors = validate_folder(official_dir, official_manifest_path)
    write_validation_report(final_metrics, final_errors, backup_path)

    print("ROI 224 folder naming finalized.")
    print(f"legacy_backup_path: {backup_path}")
    print(f"official_folder: {official_dir}")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")
    print(f"error_rows: {len(final_errors)}")
    return 0 if not final_errors and final_metrics["manifest_rows"] == EXPECTED_TOTAL and final_metrics["png_count"] == EXPECTED_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())

