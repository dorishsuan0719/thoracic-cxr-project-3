from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import CLASS_DIR_NAMES, bool_from_csv, clamp_bbox, crop_manifest_path, crop_root, load_image, metadata_dir, write_csv_rows

CROP_MANIFEST_FIELDS = [
    "crop_id",
    "split",
    "source_image_id",
    "source_image_path",
    "raw_image_path",
    "crop_path",
    "class_id",
    "class_name",
    "rad_id",
    "original_row_index",
    "annotation_index",
    "original_width",
    "original_height",
    "original_x_min",
    "original_y_min",
    "original_x_max",
    "original_y_max",
    "clamped_x_min",
    "clamped_y_min",
    "clamped_x_max",
    "clamped_y_max",
    "crop_width",
    "crop_height",
    "margin_ratio",
    "is_valid",
    "error_reason",
]


def read_valid_rows() -> list[dict[str, Any]]:
    with (metadata_dir() / "final_annotations_for_crop.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop full-resolution BBox ROI PNGs.")
    parser.add_argument("--margin-ratio", type=float, default=0.0)
    parser.add_argument("--allow-out-of-bounds-clamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.margin_ratio < 0:
        raise SystemExit("--margin-ratio must be >= 0")

    rows = read_valid_rows()
    needs_clamp = [row for row in rows if bool_from_csv(row.get("needs_clamp"))]
    if needs_clamp and not args.allow_out_of_bounds_clamp:
        raise SystemExit(
            f"Stopped before formal crop: {len(needs_clamp)} valid audited BBox rows need boundary clamp. "
            f"Inspect {metadata_dir() / 'bbox_needing_clamp.csv'} and overlays before rerunning."
        )

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        split = row["split"]
        class_id = int(row["class_id"])
        class_name = row["class_name"]
        image_id = row["image_id"]
        annotation_index = int(row["annotation_index"])
        crop_id = f"{split}_{image_id}_class{class_id}_bbox{annotation_index:04d}"
        crop_dir = crop_root() / split / CLASS_DIR_NAMES[class_id]
        crop_path = crop_dir / f"{image_id}_class{class_id}_bbox{annotation_index:04d}.png"
        crop_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "crop_id": crop_id,
            "split": split,
            "source_image_id": image_id,
            "source_image_path": row["source_image_path"],
            "raw_image_path": row["raw_image_path"],
            "crop_path": str(crop_path),
            "class_id": class_id,
            "class_name": class_name,
            "rad_id": row["rad_id"],
            "original_row_index": row["original_row_index"],
            "annotation_index": annotation_index,
            "original_width": row["original_width"],
            "original_height": row["original_height"],
            "original_x_min": row["x_min"],
            "original_y_min": row["y_min"],
            "original_x_max": row["x_max"],
            "original_y_max": row["y_max"],
            "clamped_x_min": "",
            "clamped_y_min": "",
            "clamped_x_max": "",
            "clamped_y_max": "",
            "crop_width": "",
            "crop_height": "",
            "margin_ratio": args.margin_ratio,
            "is_valid": False,
            "error_reason": "",
        }

        try:
            raw_path = Path(row["raw_image_path"])
            with load_image(raw_path) as image:
                width, height = image.size
                bbox = {
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                }
                crop_box = clamp_bbox(bbox, width, height, margin_ratio=args.margin_ratio)
                crop = image.crop(
                    (
                        crop_box["clamped_x_min"],
                        crop_box["clamped_y_min"],
                        crop_box["clamped_x_max"],
                        crop_box["clamped_y_max"],
                    )
                )
                if crop.size[0] <= 0 or crop.size[1] <= 0:
                    manifest["error_reason"] = "empty_crop"
                else:
                    crop.save(crop_path, format="PNG")
                    manifest.update(crop_box)
                    manifest["crop_width"] = crop.size[0]
                    manifest["crop_height"] = crop.size[1]
                    manifest["is_valid"] = True
        except Exception as exc:  # noqa: BLE001
            manifest["error_reason"] = f"crop_error:{exc}"
        manifest_rows.append(manifest)

    write_csv_rows(crop_manifest_path(), CROP_MANIFEST_FIELDS, manifest_rows)
    valid = sum(1 for row in manifest_rows if row["is_valid"] is True)
    print("ROI crop completed.")
    print(f"Valid crops: {valid}")
    print(f"Failed crops: {len(manifest_rows) - valid}")
    print(f"Crop manifest: {crop_manifest_path()}")
    print(f"Crop root: {crop_root()}")


if __name__ == "__main__":
    main()
