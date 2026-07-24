from __future__ import annotations

import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import metadata_dir, write_csv_rows

REQUIRED_COLUMNS = [
    "crop_path",
    "source_image_id",
    "split",
    "class_id",
    "class_name",
    "annotation_index",
    "review_status",
    "technical_issue",
    "review_note",
]
ALLOWED_STATUS = {"pass", "review", "fail"}
BBOX_PATTERN = re.compile(r"_bbox(\d+)\.png$", re.IGNORECASE)


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


def bbox_index(path: str) -> int | None:
    match = BBOX_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else None


def add_error(errors: list[dict[str, Any]], row_number: int, row: dict[str, str], error_type: str, detail: str) -> None:
    errors.append(
        {
            "row_number": row_number,
            "crop_path": row.get("crop_path", ""),
            "source_image_id": row.get("source_image_id", ""),
            "split": row.get("split", ""),
            "class_id": row.get("class_id", ""),
            "class_name": row.get("class_name", ""),
            "annotation_index": row.get("annotation_index", ""),
            "review_status": row.get("review_status", ""),
            "technical_issue": row.get("technical_issue", ""),
            "error_type": error_type,
            "error_detail": detail,
        }
    )


def main() -> int:
    repaired_path = metadata_dir() / "manual_crop_review_repaired.csv"
    manual_path = metadata_dir() / "manual_crop_review.csv"
    backup_path = metadata_dir() / "manual_crop_review_backup_before_repair.csv"
    manifest_path = metadata_dir() / "crop_manifest.csv"
    errors_path = metadata_dir() / "manual_crop_review_repaired_validation_errors.csv"
    summary_path = metadata_dir() / "manual_crop_review_repaired_validation_summary.txt"

    repaired_rows, fieldnames = read_csv_with_fallback(repaired_path)
    manifest_rows, _ = read_csv_with_fallback(manifest_path)
    manifest_by_path = {norm_path(row["crop_path"]): row for row in manifest_rows}

    errors: list[dict[str, Any]] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        add_error(errors, 0, {}, "missing_required_columns", ";".join(missing_columns))

    crop_path_counts = Counter(norm_path(row.get("crop_path", "")) for row in repaired_rows)
    for key, count in crop_path_counts.items():
        if key and count > 1:
            for index, row in enumerate(repaired_rows, start=2):
                if norm_path(row.get("crop_path", "")) == key:
                    add_error(errors, index, row, "duplicate_crop_path", f"count={count}")

    if len(repaired_rows) != 150:
        add_error(errors, 0, {}, "row_count_not_150", f"rows={len(repaired_rows)}")
    if len(crop_path_counts) != 150:
        add_error(errors, 0, {}, "unique_crop_path_not_150", f"unique_crop_path={len(crop_path_counts)}")

    for index, row in enumerate(repaired_rows, start=2):
        status = str(row.get("review_status", "")).strip().lower()
        issue = str(row.get("technical_issue", "")).strip()
        if not status:
            add_error(errors, index, row, "missing_review_status", "review_status is blank")
        elif status not in ALLOWED_STATUS:
            add_error(errors, index, row, "invalid_review_status", status)
        if not issue:
            add_error(errors, index, row, "missing_technical_issue", "technical_issue is blank")

        crop_path = row.get("crop_path", "")
        if not Path(crop_path).exists():
            add_error(errors, index, row, "crop_path_missing_on_disk", crop_path)

        manifest = manifest_by_path.get(norm_path(crop_path))
        if manifest is None:
            add_error(errors, index, row, "crop_path_missing_from_manifest", crop_path)
        else:
            for column in ["source_image_id", "split", "class_id", "class_name", "annotation_index"]:
                if str(row.get(column, "")) != str(manifest.get(column, "")):
                    add_error(
                        errors,
                        index,
                        row,
                        "manifest_field_mismatch",
                        f"{column}: repaired={row.get(column, '')}, manifest={manifest.get(column, '')}",
                    )

        file_bbox = bbox_index(crop_path)
        try:
            annotation_index = int(str(row.get("annotation_index", "")).strip())
        except ValueError:
            annotation_index = None
        if annotation_index is None or file_bbox is None or annotation_index != file_bbox:
            add_error(errors, index, row, "annotation_index_filename_mismatch", f"annotation_index={annotation_index}, filename_bbox={file_bbox}")

    write_csv_rows(
        errors_path,
        [
            "row_number",
            "crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "review_status",
            "technical_issue",
            "error_type",
            "error_detail",
        ],
        errors,
    )

    status_counts = Counter(str(row.get("review_status", "")).strip().lower() for row in repaired_rows)
    issue_counts = Counter(str(row.get("technical_issue", "")).strip() for row in repaired_rows)
    summary = {
        "repaired_rows": len(repaired_rows),
        "repaired_unique_crop_paths": len(crop_path_counts),
        "duplicate_crop_path_count": sum(1 for count in crop_path_counts.values() if count > 1),
        "validation_error_count": len(errors),
        "pass_count": status_counts["pass"],
        "review_count": status_counts["review"],
        "fail_count": status_counts["fail"],
        "blank_review_status_count": status_counts[""],
        "blank_technical_issue_count": issue_counts[""],
        "backup_exists": backup_path.exists(),
        "applied_to_manual_crop_review": False,
    }

    if errors:
        with summary_path.open("w", encoding="utf-8") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        print("Repaired manual crop review validation failed.")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"Errors: {errors_path}")
        return 1

    try:
        shutil.copy2(repaired_path, manual_path)
        summary["applied_to_manual_crop_review"] = True
    except PermissionError as exc:
        summary["apply_error"] = str(exc)
        with summary_path.open("w", encoding="utf-8") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        print("Repaired manual crop review validation passed, but applying it failed.")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"Summary: {summary_path}")
        return 1
    with summary_path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("Repaired manual crop review validation passed.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Applied repaired file to: {manual_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
