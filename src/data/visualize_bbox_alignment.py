from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import ImageDraw, ImageFont

from common import CLASS_COLORS, CLASS_ORDER, SPLITS, load_image, metadata_dir, overlay_root, write_csv_rows

RANDOM_SEED = 42
MAX_IMAGES_PER_SPLIT_CLASS = 10


def read_valid_rows() -> list[dict[str, Any]]:
    with (metadata_dir() / "valid_annotations.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_rows(image, rows: list[dict[str, Any]]):
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    width, height = rendered.size

    title_lines = [f"image_id: {rows[0]['image_id']}", f"split: {rows[0]['split']}", f"original_size: {width}x{height}"]
    for row in rows[:8]:
        title_lines.append(
            "class{class_id} {class_name} bbox=({x_min},{y_min},{x_max},{y_max}) rad_id={rad_id} idx={annotation_index}".format(**row)
        )
    line_h = 14
    panel_h = line_h * len(title_lines) + 8
    draw.rectangle([0, 0, width, panel_h], fill=(0, 0, 0))
    for idx, line in enumerate(title_lines):
        draw.text((6, 4 + idx * line_h), line, fill=(255, 255, 255), font=font)

    for row in rows:
        class_id = int(row["class_id"])
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        box = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])]
        line_width = max(3, round(min(width, height) / 800))
        draw.rectangle(box, outline=color, width=line_width)
        label = f"class{class_id} {row['class_name']}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_x = max(0, min(int(box[0]), width - text_w - 4))
        label_y = max(panel_h, int(box[1]) - text_h - 6)
        draw.rectangle([label_x, label_y, label_x + text_w + 4, label_y + text_h + 4], fill=color)
        draw.text((label_x + 2, label_y + 2), label, fill=(0, 0, 0), font=font)
    return rendered


def main() -> None:
    rows = read_valid_rows()
    rows_by_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    image_ids_by_split_class: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        row["class_id"] = int(row["class_id"])
        rows_by_image[(row["split"], row["image_id"])].append(row)
        image_ids_by_split_class[(row["split"], row["class_id"])].add(row["image_id"])

    rng = random.Random(RANDOM_SEED)
    summary_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            image_ids = sorted(image_ids_by_split_class[(split, class_id)])
            rng.shuffle(image_ids)
            selected = sorted(image_ids[:MAX_IMAGES_PER_SPLIT_CLASS])
            out_dir = overlay_root() / split / f"{class_id}_{class_name.lower().replace(' ', '_')}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for image_id in selected:
                image_rows = rows_by_image[(split, image_id)]
                raw_path = Path(image_rows[0]["raw_image_path"])
                with load_image(raw_path) as image:
                    rendered = draw_rows(image, image_rows)
                    out_path = out_dir / f"{split}_class{class_id}_{image_id}.png"
                    rendered.save(out_path, format="PNG")
                file_rows.append(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": class_name,
                        "image_id": image_id,
                        "raw_image_path": str(raw_path),
                        "overlay_path": str(out_path),
                    }
                )

            summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "available_source_images": len(image_ids),
                    "requested_samples": MAX_IMAGES_PER_SPLIT_CLASS,
                    "rendered_samples": len(selected),
                    "output_dir": str(out_dir),
                }
            )

    write_csv_rows(metadata_dir() / "bbox_overlay_summary.csv", ["split", "class_id", "class_name", "available_source_images", "requested_samples", "rendered_samples", "output_dir"], summary_rows)
    write_csv_rows(metadata_dir() / "bbox_overlay_files.csv", ["split", "class_id", "class_name", "image_id", "raw_image_path", "overlay_path"], file_rows)

    print("BBox overlay visualization completed.")
    print(f"Rendered overlays: {len(file_rows)}")
    print(f"Overlay root: {overlay_root()}")
    print("No ROI crop was performed.")


if __name__ == "__main__":
    main()

