from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    CLASS_ORDER,
    IMAGE_SEARCH_DIRS,
    PROJECT_ROOT,
    SPLITS,
    build_source_image_index,
    copy_or_convert_source_image,
    metadata_dir,
    normalize_source_annotation,
    raw_annotation_path,
    raw_image_dir,
    read_csv_rows,
    source_annotation_path,
    write_csv_rows,
)

ANNOTATION_FIELDS = [
    "image_id",
    "class_id",
    "class_name",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "rad_id",
    "split",
    "original_row_index",
    "annotation_index",
]
IMAGE_MANIFEST_FIELDS = [
    "source_image_id",
    "source_image_path",
    "raw_image_path",
    "file_extension",
    "original_width",
    "original_height",
    "copy_status",
    "error_reason",
]


def copy_annotation_csvs() -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        source_path = source_annotation_path(split)
        rows = [
            normalize_source_annotation(row, split=split, original_row_index=idx)
            for idx, row in enumerate(read_csv_rows(source_path))
        ]
        write_csv_rows(raw_annotation_path(split), ANNOTATION_FIELDS, rows)
        all_rows.extend(rows)
    return all_rows


def main() -> None:
    raw_image_dir().mkdir(parents=True, exist_ok=True)
    metadata_dir().mkdir(parents=True, exist_ok=True)

    annotations = copy_annotation_csvs()
    unique_image_ids = sorted({row["image_id"] for row in annotations if row["image_id"]})
    source_index, image_source_rows = build_source_image_index()

    manifest_rows: list[dict[str, Any]] = []
    for image_id in unique_image_ids:
        source_path = source_index.get(image_id)
        row = {
            "source_image_id": image_id,
            "source_image_path": str(source_path) if source_path else "",
            "raw_image_path": "",
            "file_extension": source_path.suffix.lower() if source_path else "",
            "original_width": "",
            "original_height": "",
            "copy_status": "missing",
            "error_reason": "missing_source_image",
        }
        if source_path is not None:
            try:
                destination_base = raw_image_dir() / image_id
                width, height, raw_path = copy_or_convert_source_image(source_path, destination_base)
                row.update(
                    {
                        "raw_image_path": raw_path,
                        "original_width": width,
                        "original_height": height,
                        "copy_status": "copied",
                        "error_reason": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["copy_status"] = "error"
                row["error_reason"] = str(exc)
        manifest_rows.append(row)

    write_csv_rows(metadata_dir() / "image_sources.csv", ["search_order", "path", "exists", "indexed_files"], image_source_rows)
    write_csv_rows(metadata_dir() / "image_manifest.csv", IMAGE_MANIFEST_FIELDS, manifest_rows)

    split_counts = defaultdict(set)
    for row in annotations:
        split_counts[row["split"]].add(row["image_id"])

    copied = sum(1 for row in manifest_rows if row["copy_status"] == "copied")
    missing = sum(1 for row in manifest_rows if row["copy_status"] == "missing")
    errors = sum(1 for row in manifest_rows if row["copy_status"] == "error")
    print("Raw image collection completed.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Image search dirs:")
    for path in IMAGE_SEARCH_DIRS:
        print(f"  {path}")
    print(f"Unique source images: {len(unique_image_ids)}")
    for split in SPLITS:
        print(f"{split} source images: {len(split_counts[split])}")
    print(f"Copied/converted images: {copied}")
    print(f"Missing images: {missing}")
    print(f"Image copy/convert errors: {errors}")
    print(f"Image manifest: {metadata_dir() / 'image_manifest.csv'}")


if __name__ == "__main__":
    main()

