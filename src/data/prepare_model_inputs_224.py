from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_DIR_NAMES, PROJECT_ROOT, metadata_dir, write_csv_rows

MANIFEST_FIELDS = [
    "source_crop_path",
    "output_224_path",
    "source_image_id",
    "split",
    "class_id",
    "class_name",
    "annotation_index",
    "manual_review_status",
    "technical_issue",
    "review_note",
    "source_width",
    "source_height",
    "resized_width",
    "resized_height",
    "pad_left",
    "pad_top",
    "pad_right",
    "pad_bottom",
    "output_width",
    "output_height",
    "conversion_status",
    "error_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 224x224 grayscale letterboxed model inputs from ROI crops.")
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "final_crops_for_model.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "bbox_crops_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding-value", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing 224 PNG outputs.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def bool_true(value: Any) -> bool:
    return str(value).strip().upper() == "TRUE"


def output_path_for(row: dict[str, str], output_dir: Path) -> Path:
    class_id = int(row["class_id"])
    return output_dir / row["split"] / CLASS_DIR_NAMES[class_id] / Path(row["crop_path"]).name


def letterbox_grayscale(image: Image.Image, image_size: int, padding_value: int) -> tuple[Image.Image, dict[str, int]]:
    gray = image.convert("L")
    source_width, source_height = gray.size
    scale = min(image_size / source_width, image_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC
    resized = gray.resize((resized_width, resized_height), resample=resample)

    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2
    pad_right = image_size - resized_width - pad_left
    pad_bottom = image_size - resized_height - pad_top
    canvas = Image.new("L", (image_size, image_size), color=padding_value)
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, {
        "source_width": source_width,
        "source_height": source_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "output_width": image_size,
        "output_height": image_size,
    }


def main() -> int:
    args = parse_args()
    if args.image_size <= 0:
        raise SystemExit("--image-size must be > 0")
    if not 0 <= args.padding_value <= 255:
        raise SystemExit("--padding-value must be in [0, 255]")

    input_rows = [row for row in read_rows(args.input_csv) if bool_true(row.get("include_for_model"))]
    output_paths = [str(output_path_for(row, args.output_dir).resolve()).casefold() for row in input_rows]
    duplicate_paths = {path for path, count in Counter(output_paths).items() if count > 1}
    if duplicate_paths:
        print(f"Stopped: duplicate output paths would overwrite files: {len(duplicate_paths)}")
        return 1

    manifest_rows: list[dict[str, Any]] = []
    for row in input_rows:
        source_crop_path = Path(row["crop_path"])
        output_path = output_path_for(row, args.output_dir)
        manifest = {
            "source_crop_path": str(source_crop_path),
            "output_224_path": str(output_path),
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "annotation_index": row["annotation_index"],
            "manual_review_status": row.get("manual_review_status", ""),
            "technical_issue": row.get("technical_issue", ""),
            "review_note": row.get("review_note", ""),
            "source_width": "",
            "source_height": "",
            "resized_width": "",
            "resized_height": "",
            "pad_left": "",
            "pad_top": "",
            "pad_right": "",
            "pad_bottom": "",
            "output_width": "",
            "output_height": "",
            "conversion_status": "failed",
            "error_reason": "",
        }
        try:
            if not source_crop_path.exists():
                raise FileNotFoundError(f"source crop not found: {source_crop_path}")
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"output exists and --overwrite was not set: {output_path}")

            with Image.open(source_crop_path) as image:
                converted, metrics = letterbox_grayscale(image, args.image_size, args.padding_value)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            converted.save(output_path, format="PNG")
            manifest.update(metrics)
            manifest["conversion_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            manifest["error_reason"] = str(exc)
        manifest_rows.append(manifest)

    manifest_path = metadata_dir() / "model_input_224_manifest.csv"
    write_csv_rows(manifest_path, MANIFEST_FIELDS, manifest_rows)
    success_count = sum(1 for row in manifest_rows if row["conversion_status"] == "success")
    failed_count = len(manifest_rows) - success_count
    print("224 model input conversion completed.")
    print(f"Input rows included: {len(input_rows)}")
    print(f"Converted successfully: {success_count}")
    print(f"Failed conversions: {failed_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {args.output_dir}")
    return 0 if failed_count == 0 and success_count == len(input_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
