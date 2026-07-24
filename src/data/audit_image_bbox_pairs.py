from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    CLASS_ORDER,
    SPLITS,
    duplicate_key,
    load_image,
    load_image_manifest,
    load_raw_annotations,
    metadata_dir,
    validate_bbox,
    write_csv_rows,
)

AUDIT_FIELDS = [
    "split",
    "image_id",
    "class_id",
    "class_name",
    "rad_id",
    "original_row_index",
    "annotation_index",
    "source_image_path",
    "raw_image_path",
    "original_width",
    "original_height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "is_duplicate",
    "duplicate_of_annotation_index",
    "is_valid",
    "needs_clamp",
    "error_reason",
]


def main() -> None:
    metadata_dir().mkdir(parents=True, exist_ok=True)
    annotations = load_raw_annotations()
    image_manifest = load_image_manifest()

    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    audited_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    clamp_rows: list[dict[str, Any]] = []

    for row in annotations:
        image_info = image_manifest.get(row["image_id"], {})
        raw_path = image_info.get("raw_image_path", "")
        source_path = image_info.get("source_image_path", "")
        width = int(image_info["original_width"]) if str(image_info.get("original_width", "")).isdigit() else 0
        height = int(image_info["original_height"]) if str(image_info.get("original_height", "")).isdigit() else 0

        key = duplicate_key(row)
        duplicate_of = seen.get(key)
        is_duplicate = duplicate_of is not None
        if not is_duplicate:
            seen[key] = row

        audited = {
            **row,
            "source_image_path": source_path,
            "raw_image_path": raw_path,
            "original_width": width or "",
            "original_height": height or "",
            "is_duplicate": is_duplicate,
            "duplicate_of_annotation_index": duplicate_of["annotation_index"] if duplicate_of else "",
            "is_valid": False,
            "needs_clamp": False,
            "error_reason": "",
        }

        if is_duplicate:
            audited["error_reason"] = "duplicate_annotation"
            duplicate_rows.append(audited.copy())
            audited_rows.append(audited)
            continue

        if not raw_path or not Path(raw_path).exists():
            audited["error_reason"] = "missing_image"
            missing_rows.append(audited.copy())
            audited_rows.append(audited)
            continue

        try:
            with load_image(Path(raw_path)) as image:
                actual_width, actual_height = image.size
        except Exception as exc:  # noqa: BLE001
            audited["error_reason"] = f"image_read_error:{exc}"
            invalid_rows.append(audited.copy())
            audited_rows.append(audited)
            continue

        audited["original_width"] = actual_width
        audited["original_height"] = actual_height
        valid, reason, needs_clamp, _ = validate_bbox(row, actual_width, actual_height, margin_ratio=0.0)
        audited["is_valid"] = valid
        audited["needs_clamp"] = needs_clamp
        audited["error_reason"] = reason

        if valid and needs_clamp:
            clamp_rows.append(audited.copy())
        elif not valid:
            invalid_rows.append(audited.copy())
        audited_rows.append(audited)

    valid_rows = [row for row in audited_rows if row["is_valid"] is True and row["is_duplicate"] is False]
    write_csv_rows(metadata_dir() / "audited_annotations.csv", AUDIT_FIELDS, audited_rows)
    write_csv_rows(metadata_dir() / "valid_annotations.csv", AUDIT_FIELDS, valid_rows)
    write_csv_rows(metadata_dir() / "missing_images.csv", AUDIT_FIELDS, missing_rows)
    write_csv_rows(metadata_dir() / "invalid_bboxes.csv", AUDIT_FIELDS, invalid_rows)
    write_csv_rows(metadata_dir() / "duplicate_annotations.csv", AUDIT_FIELDS, duplicate_rows)
    write_csv_rows(metadata_dir() / "bbox_needing_clamp.csv", AUDIT_FIELDS, clamp_rows)

    summary_rows = []
    by_group = defaultdict(lambda: {"source_images": set(), "raw_annotations": 0, "valid_bboxes": 0, "missing": 0, "invalid": 0, "duplicates": 0})
    for row in audited_rows:
        key = (row["split"], int(row["class_id"]))
        group = by_group[key]
        group["raw_annotations"] += 1
        if row["is_duplicate"] is True:
            group["duplicates"] += 1
        elif row["error_reason"] == "missing_image":
            group["missing"] += 1
        elif row["is_valid"] is True:
            group["valid_bboxes"] += 1
            group["source_images"].add(row["image_id"])
        else:
            group["invalid"] += 1

    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            group = by_group[(split, class_id)]
            summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image_count": len(group["source_images"]),
                    "annotation_count": group["raw_annotations"],
                    "valid_bbox_count": group["valid_bboxes"],
                    "missing_image_count": group["missing"],
                    "invalid_bbox_count": group["invalid"],
                    "duplicate_removed_count": group["duplicates"],
                }
            )
    write_csv_rows(
        metadata_dir() / "image_bbox_audit_summary.csv",
        ["split", "class_id", "class_name", "source_image_count", "annotation_count", "valid_bbox_count", "missing_image_count", "invalid_bbox_count", "duplicate_removed_count"],
        summary_rows,
    )

    totals = {
        "annotation_rows": len(audited_rows),
        "valid_bbox_rows": len(valid_rows),
        "missing_images": len(missing_rows),
        "invalid_bboxes": len(invalid_rows),
        "duplicate_annotations_removed": len(duplicate_rows),
        "bbox_rows_needing_clamp": len(clamp_rows),
    }
    with (metadata_dir() / "image_bbox_audit_summary.txt").open("w", encoding="utf-8") as f:
        for key, value in totals.items():
            f.write(f"{key}: {value}\n")

    print("Image/BBox audit completed.")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"Metadata output: {metadata_dir()}")


if __name__ == "__main__":
    main()

