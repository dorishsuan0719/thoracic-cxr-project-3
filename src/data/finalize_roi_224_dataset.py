from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_ORDER, SPLITS, metadata_dir, reports_dir, write_csv_rows

REQUIRED_COLUMNS = [
    "output_224_path",
    "source_crop_path",
    "source_image_id",
    "split",
    "class_id",
    "class_name",
    "annotation_index",
    "manual_crop_review_status",
    "previous_technical_issue",
    "review_status",
    "review_note",
]
ALLOWED_STATUS = {"pass", "review", "fail"}


def read_csv_with_fallback(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp950", "mbcs"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader), list(reader.fieldnames or [])
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode CSV")


def norm_path(value: str) -> str:
    return str(Path(value).resolve()).casefold() if value else ""


def add_error(errors: list[dict[str, Any]], row_number: int, row: dict[str, str], error_type: str, detail: str) -> None:
    errors.append(
        {
            "row_number": row_number,
            "output_224_path": row.get("output_224_path", ""),
            "source_crop_path": row.get("source_crop_path", ""),
            "source_image_id": row.get("source_image_id", ""),
            "split": row.get("split", ""),
            "class_id": row.get("class_id", ""),
            "class_name": row.get("class_name", ""),
            "annotation_index": row.get("annotation_index", ""),
            "review_status": row.get("review_status", ""),
            "review_note": row.get("review_note", ""),
            "error_type": error_type,
            "error_detail": detail,
        }
    )


def validate_manual_224(rows: list[dict[str, str]], fieldnames: list[str], manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    errors: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        add_error(errors, 0, {}, "missing_required_columns", ";".join(missing_columns))

    manifest_by_output = {norm_path(row["output_224_path"]): row for row in manifest_rows}
    path_counts = Counter(norm_path(row.get("output_224_path", "")) for row in rows)

    if len(rows) != 30:
        add_error(errors, 0, {}, "row_count_not_30", f"rows={len(rows)}")
    for output_path, count in path_counts.items():
        if output_path and count > 1:
            for idx, row in enumerate(rows, start=2):
                if norm_path(row.get("output_224_path", "")) == output_path:
                    add_error(errors, idx, row, "duplicate_output_224_path", f"count={count}")

    for idx, row in enumerate(rows, start=2):
        status = str(row.get("review_status", "") or "").strip().lower()
        status_counts[status] += 1
        if not status:
            add_error(errors, idx, row, "missing_review_status", "review_status is blank or the column is missing")
        elif status not in ALLOWED_STATUS:
            add_error(errors, idx, row, "invalid_review_status", status)
        if not str(row.get("review_note", "") or "").strip():
            add_error(errors, idx, row, "missing_review_note", "review_note is blank")

        output_path = Path(row.get("output_224_path", ""))
        if not output_path.exists():
            add_error(errors, idx, row, "missing_output_224_path", str(output_path))
            continue

        manifest = manifest_by_output.get(norm_path(str(output_path)))
        if manifest is None:
            add_error(errors, idx, row, "output_path_missing_from_manifest", str(output_path))
        else:
            for field in ["annotation_index", "split", "class_id", "class_name"]:
                if str(row.get(field, "")) != str(manifest.get(field, "")):
                    add_error(errors, idx, row, "manifest_field_mismatch", f"{field}: manual={row.get(field, '')}, manifest={manifest.get(field, '')}")

        try:
            with Image.open(output_path) as image:
                image.verify()
            with Image.open(output_path) as image:
                if image.size != (224, 224):
                    add_error(errors, idx, row, "wrong_image_size", f"{image.size[0]}x{image.size[1]}")
        except Exception as exc:  # noqa: BLE001
            add_error(errors, idx, row, "unreadable_image", str(exc))

    return errors, status_counts


def write_manual_reports(rows: list[dict[str, str]], status_counts: Counter[str], errors: list[dict[str, Any]]) -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for status in ("pass", "review", "fail", ""):
        summary_rows.append({"summary_type": "review_status", "name": status or "(blank)", "count": status_counts[status]})
    for error_type, count in sorted(Counter(row["error_type"] for row in errors).items()):
        summary_rows.append({"summary_type": "validation_error", "name": error_type, "count": count})

    write_csv_rows(reports_dir() / "manual_224_review_summary.csv", ["summary_type", "name", "count"], summary_rows)
    write_csv_rows(
        reports_dir() / "manual_224_review_errors.csv",
        [
            "row_number",
            "output_224_path",
            "source_crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "review_status",
            "review_note",
            "error_type",
            "error_detail",
        ],
        errors,
    )
    with (reports_dir() / "manual_224_review_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Manual 224 review summary\n")
        f.write(f"manual_224_review_rows: {len(rows)}\n")
        f.write(f"validation_error_rows: {len(errors)}\n")
        for status in ("pass", "review", "fail", ""):
            f.write(f"{status or '(blank)'}: {status_counts[status]}\n")
        if errors:
            f.write("\nvalidation_error_counts\n")
            for error_type, count in sorted(Counter(row["error_type"] for row in errors).items()):
                f.write(f"{error_type}: {count}\n")


def create_final_reports(status_counts: Counter[str]) -> None:
    manifest_rows, _ = read_csv_with_fallback(metadata_dir() / "model_input_224_manifest.csv")
    final_crops_rows, _ = read_csv_with_fallback(metadata_dir() / "final_crops_for_model.csv")
    excluded_rows, _ = read_csv_with_fallback(metadata_dir() / "excluded_annotations.csv")
    audit_lines = (reports_dir() / "model_input_224_audit.txt").read_text(encoding="utf-8").splitlines()
    audit = {}
    for line in audit_lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            audit[key] = value

    included = [row for row in final_crops_rows if str(row.get("include_for_model", "")).upper() == "TRUE"]
    split_class_counts = Counter((row["split"], row["class_id"], row["class_name"]) for row in manifest_rows if row.get("conversion_status") == "success")
    split_counts = Counter(row["split"] for row in manifest_rows if row.get("conversion_status") == "success")
    manual_crop_counts = Counter(row.get("manual_review_status", "") for row in final_crops_rows)
    excluded_counts = Counter(row.get("exclusion_reason", "") for row in excluded_rows)

    summary_rows = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image_count": split_class_counts[(split, str(class_id), class_name)],
                }
            )
    write_csv_rows(
        reports_dir() / "roi_classification_dataset_final_summary_by_split_class.csv",
        ["split", "class_id", "class_name", "image_count"],
        summary_rows,
    )

    with (reports_dir() / "roi_classification_dataset_final_summary.txt").open("w", encoding="utf-8") as f:
        f.write("ROI classification dataset final summary\n")
        f.write("dataset_name: thoracic_cxr_roi_classification\n")
        f.write("version: roi_224_v1\n")
        f.write("num_classes: 5\n")
        f.write(f"total_images: {len(included)}\n")
        f.write(f"train_images: {split_counts['train']}\n")
        f.write(f"val_images: {split_counts['val']}\n")
        f.write(f"test_images: {split_counts['test']}\n")
        f.write("image_size: 224x224\n")
        f.write("all_images_224x224: True\n")
        f.write(f"missing_images: {audit.get('missing_output', '0')}\n")
        f.write(f"unreadable_images: {audit.get('unreadable_output', '0')}\n")
        f.write(f"empty_images: {audit.get('empty_image_all_zero', '0')}\n")
        f.write(f"duplicate_output_path_groups: {audit.get('duplicate_output_path_groups', '0')}\n")
        f.write(f"train_vs_val_leakage: {audit.get('train_vs_val_leakage', '0')}\n")
        f.write(f"train_vs_test_leakage: {audit.get('train_vs_test_leakage', '0')}\n")
        f.write(f"val_vs_test_leakage: {audit.get('val_vs_test_leakage', '0')}\n")
        f.write("\nmanual_crop_review_counts\n")
        for key in ("pass", "review", "fail", "not_sampled"):
            f.write(f"{key}: {manual_crop_counts[key]}\n")
        f.write("\nmanual_224_review_counts\n")
        for key in ("pass", "review", "fail"):
            f.write(f"{key}: {status_counts[key]}\n")
        f.write("\nexcluded_annotation_counts\n")
        f.write(f"duplicate_annotation: {excluded_counts['duplicate_annotation']}\n")
        f.write(f"same_bbox_different_class: {excluded_counts['same_bbox_different_class']}\n")
        f.write("\nprocessing\n")
        f.write("roi_source: Ground Truth BBox ROI crop\n")
        f.write("margin_ratio: 0\n")
        f.write("resize_method: keep aspect ratio letterbox\n")
        f.write("padding: black, value 0\n")
        f.write("output_size: 224x224\n")
        f.write("augmentation_applied: false\n")
        f.write("normalization_applied: false\n")

    version = {
        "dataset_name": "thoracic_cxr_roi_classification",
        "version": "roi_224_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_classes": 5,
        "class_mapping": {str(class_id): class_name for class_name, class_id in CLASS_ORDER},
        "total_images": len(included),
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "test_images": split_counts["test"],
        "image_size": [224, 224],
        "resize_method": "keep_aspect_ratio_letterbox",
        "padding_value": 0,
        "augmentation_applied": False,
        "normalization_applied": False,
        "source_manifest": str(metadata_dir() / "model_input_224_manifest.csv"),
        "final_model_csv": str(metadata_dir() / "final_crops_for_model.csv"),
        "leakage_check": {
            "train_vs_val": int(audit.get("train_vs_val_leakage", "0")),
            "train_vs_test": int(audit.get("train_vs_test_leakage", "0")),
            "val_vs_test": int(audit.get("val_vs_test_leakage", "0")),
        },
        "manual_review_summary": {
            "manual_crop_review": dict(manual_crop_counts),
            "manual_224_review": {key: status_counts[key] for key in ("pass", "review", "fail")},
        },
    }
    (metadata_dir() / "dataset_version_roi_224_v1.json").write_text(
        json.dumps(version, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    manual_rows, manual_fields = read_csv_with_fallback(metadata_dir() / "manual_224_review.csv")
    manifest_rows, _ = read_csv_with_fallback(metadata_dir() / "model_input_224_manifest.csv")
    errors, status_counts = validate_manual_224(manual_rows, manual_fields, manifest_rows)
    write_manual_reports(manual_rows, status_counts, errors)

    print("Manual 224 review validation completed.")
    print(f"manual_224_review_rows: {len(manual_rows)}")
    print(f"validation_error_rows: {len(errors)}")
    for status in ("pass", "review", "fail", ""):
        print(f"{status or '(blank)'}: {status_counts[status]}")

    if errors or status_counts["fail"] > 0:
        print("Stopped before final dataset summary because validation errors or fail reviews exist.")
        return 1

    create_final_reports(status_counts)
    print("Final dataset reports created.")
    print(f"Final summary: {reports_dir() / 'roi_classification_dataset_final_summary.txt'}")
    print(f"Dataset version: {metadata_dir() / 'dataset_version_roi_224_v1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

