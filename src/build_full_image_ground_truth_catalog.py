from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_INFO = {
    0: ("Aortic enlargement", "主動脈擴大", "aortic_enlargement"),
    1: ("Cardiomegaly", "心臟擴大", "cardiomegaly"),
    2: ("Pleural thickening", "胸膜增厚", "pleural_thickening"),
    3: ("Pulmonary fibrosis", "肺纖維化", "pulmonary_fibrosis"),
    4: ("Pleural effusion", "胸腔積液", "pleural_effusion"),
}

MANIFEST_FIELDS = [
    "catalog_index",
    "image_id",
    "full_image_path",
    "image_filename",
    "image_sha256",
    "original_width",
    "original_height",
    "original_mode",
    "label_0_aortic_enlargement",
    "label_1_cardiomegaly",
    "label_2_pleural_thickening",
    "label_3_pulmonary_fibrosis",
    "label_4_pleural_effusion",
    "label_vector",
    "label_code",
    "positive_class_ids",
    "positive_class_names_en",
    "positive_class_names_zh",
    "num_positive_labels",
    "annotation_row_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full-image multilabel Ground Truth catalog and fixed Demo set."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-count", type=int, default=30)
    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument("--copy-demo-images", dest="copy_demo_images", action="store_true")
    copy_group.add_argument("--no-copy-demo-images", dest="copy_demo_images", action="store_false")
    parser.set_defaults(copy_demo_images=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_source_sha(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["image_id"]):
        digest.update(record["image_filename"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["image_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv_bom(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_label_vector(labels: tuple[int, ...]) -> list[int]:
    return [int(class_id in labels) for class_id in CLASS_INFO]


def label_code(labels: tuple[int, ...]) -> str:
    return "labels_" + "-".join(str(class_id) for class_id in labels)


def inspect_sources(images_dir: Path, annotations_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not images_dir.is_dir():
        raise RuntimeError(f"Images directory does not exist: {images_dir}")
    if not annotations_path.is_file() or annotations_path.stat().st_size == 0:
        raise RuntimeError(f"Annotation CSV is missing or empty: {annotations_path}")

    image_paths = sorted(
        (path.resolve() for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name.lower(),
    )
    all_files = [path for path in images_dir.iterdir() if path.is_file()]
    non_png = sorted(path.name for path in all_files if path.suffix.lower() != ".png")
    image_id_paths: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        image_id_paths[path.stem].append(path)
    duplicate_image_ids = sorted(image_id for image_id, paths in image_id_paths.items() if len(paths) > 1)

    required_columns = {"image_id", "class_id", "class_name"}
    annotations_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    annotation_rows: list[dict[str, str]] = []
    invalid_labels: list[dict[str, Any]] = []
    exact_annotation_counter: Counter[tuple[tuple[str, str], ...]] = Counter()
    with annotations_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            raise RuntimeError(f"Missing required annotation columns: {missing_columns}")
        for row_number, row in enumerate(reader, start=2):
            normalized = {key: (value or "").strip() for key, value in row.items()}
            annotation_rows.append(normalized)
            exact_annotation_counter[tuple(sorted(normalized.items()))] += 1
            image_id = normalized["image_id"]
            try:
                class_id = int(normalized["class_id"])
            except ValueError:
                invalid_labels.append({"row": row_number, "reason": "non_integer_class_id", "row_data": normalized})
                continue
            expected_name = CLASS_INFO.get(class_id, (None, None, None))[0]
            if expected_name is None or normalized["class_name"] != expected_name:
                invalid_labels.append({
                    "row": row_number,
                    "reason": "invalid_class_id_or_name",
                    "class_id": normalized["class_id"],
                    "class_name": normalized["class_name"],
                })
                continue
            if not image_id:
                invalid_labels.append({"row": row_number, "reason": "empty_image_id"})
                continue
            annotations_by_image[image_id].append(normalized)

    image_ids = set(image_id_paths)
    annotation_ids = set(annotations_by_image)
    missing_images = sorted(annotation_ids - image_ids)
    images_without_annotations = sorted(image_ids - annotation_ids)

    records: list[dict[str, Any]] = []
    unreadable_images: list[dict[str, str]] = []
    sha_to_image_ids: dict[str, list[str]] = defaultdict(list)
    for catalog_index, image_id in enumerate(sorted(image_ids)):
        path = image_id_paths[image_id][0]
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
        except Exception as exc:
            unreadable_images.append({"image_id": image_id, "path": str(path), "error": str(exc)})
            continue

        digest = sha256_file(path)
        sha_to_image_ids[digest].append(image_id)
        rows = annotations_by_image.get(image_id, [])
        labels = tuple(sorted({int(row["class_id"]) for row in rows}))
        vector = format_label_vector(labels)
        names_en = [CLASS_INFO[class_id][0] for class_id in labels]
        names_zh = [CLASS_INFO[class_id][1] for class_id in labels]
        record = {
            "catalog_index": catalog_index,
            "image_id": image_id,
            "full_image_path": str(path),
            "image_filename": path.name,
            "image_sha256": digest,
            "original_width": width,
            "original_height": height,
            "original_mode": mode,
            **{f"label_{class_id}_{CLASS_INFO[class_id][2]}": vector[class_id] for class_id in CLASS_INFO},
            "label_vector": json.dumps(vector, separators=(",", ":")),
            "label_vector_list": vector,
            "label_code": label_code(labels),
            "positive_class_ids": "|".join(str(class_id) for class_id in labels),
            "positive_class_ids_list": list(labels),
            "positive_class_names_en": " | ".join(names_en),
            "positive_class_names_en_list": names_en,
            "positive_class_names_zh": " | ".join(names_zh),
            "positive_class_names_zh_list": names_zh,
            "num_positive_labels": len(labels),
            "annotation_row_count": len(rows),
        }
        records.append(record)

    duplicate_sha_groups = {
        digest: ids for digest, ids in sorted(sha_to_image_ids.items()) if len(ids) > 1
    }
    duplicate_annotation_rows = sum(count - 1 for count in exact_annotation_counter.values() if count > 1)
    audit = {
        "raw_image_count": len(image_paths),
        "raw_non_png_file_count": len(non_png),
        "raw_non_png_files": non_png,
        "annotation_rows": len(annotation_rows),
        "unique_annotation_image_id_count": len(annotation_ids),
        "manifest_rows": len(records),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "image_without_annotation_count": len(images_without_annotations),
        "images_without_annotations": images_without_annotations,
        "unreadable_image_count": len(unreadable_images),
        "unreadable_images": unreadable_images,
        "invalid_label_count": len(invalid_labels),
        "invalid_labels": invalid_labels,
        "duplicate_image_id_count": len(duplicate_image_ids),
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_sha256_group_count": len(duplicate_sha_groups),
        "duplicate_sha256_groups": duplicate_sha_groups,
        "duplicate_annotation_row_count": duplicate_annotation_rows,
    }
    return records, audit


def distribution_data(records: list[dict[str, Any]]) -> tuple[Counter[int], Counter[str], Counter[int]]:
    cardinality = Counter(record["num_positive_labels"] for record in records)
    combinations = Counter(record["label_code"] for record in records)
    positives = Counter(
        class_id for record in records for class_id in record["positive_class_ids_list"]
    )
    return cardinality, combinations, positives


def select_demo_records(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(records):
        raise RuntimeError(f"demo-count must be between 1 and {len(records)}")
    cardinalities = sorted({record["num_positive_labels"] for record in records})
    if not cardinalities:
        raise RuntimeError("No label cardinalities are available for Demo selection")

    base, remainder = divmod(count, len(cardinalities))
    quotas = {cardinality: base + int(index < remainder) for index, cardinality in enumerate(cardinalities)}
    available = Counter(record["num_positive_labels"] for record in records)
    if any(quotas[cardinality] > available[cardinality] for cardinality in cardinalities):
        raise RuntimeError(f"Insufficient records for cardinality quotas: quotas={quotas}, available={dict(available)}")

    rng = random.Random(seed)
    shuffled_ids = [record["image_id"] for record in sorted(records, key=lambda item: item["image_id"])]
    rng.shuffle(shuffled_ids)
    random_rank = {image_id: rank for rank, image_id in enumerate(shuffled_ids)}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    class_coverage = Counter()
    cardinality_coverage = Counter()
    minimum_class_coverage = min(8, count)

    while len(selected) < count:
        candidates = [
            record
            for record in records
            if record["image_id"] not in selected_ids
            and cardinality_coverage[record["num_positive_labels"]] < quotas[record["num_positive_labels"]]
        ]
        if not candidates:
            raise RuntimeError("Deterministic Demo selection ran out of eligible candidates")

        def candidate_score(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
            labels = record["positive_class_ids_list"]
            deficit_sum = sum(max(0, minimum_class_coverage - class_coverage[class_id]) for class_id in labels)
            deficit_classes = sum(class_coverage[class_id] < minimum_class_coverage for class_id in labels)
            quota_remaining = quotas[record["num_positive_labels"]] - cardinality_coverage[record["num_positive_labels"]]
            current_coverage = sum(class_coverage[class_id] for class_id in labels)
            return deficit_sum, deficit_classes, quota_remaining, -current_coverage, -random_rank[record["image_id"]]

        chosen = max(candidates, key=candidate_score)
        chosen = dict(chosen)
        deficit_labels = [
            str(class_id)
            for class_id in chosen["positive_class_ids_list"]
            if class_coverage[class_id] < minimum_class_coverage
        ]
        chosen["selection_reason"] = (
            f"fixed_seed_{seed};cardinality_{chosen['num_positive_labels']}_quota;"
            f"class_coverage_{'-'.join(deficit_labels) if deficit_labels else 'balanced'}"
        )
        selected.append(chosen)
        selected_ids.add(chosen["image_id"])
        cardinality_coverage[chosen["num_positive_labels"]] += 1
        class_coverage.update(chosen["positive_class_ids_list"])

    if len(selected_ids) != count:
        raise RuntimeError("Demo selection contains duplicate image_id values")
    missing_classes = [class_id for class_id in CLASS_INFO if class_coverage[class_id] < minimum_class_coverage]
    if missing_classes:
        raise RuntimeError(
            f"Demo selection does not meet minimum class coverage {minimum_class_coverage}: {missing_classes}"
        )
    return selected


def manifest_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: record[field] for field in MANIFEST_FIELDS} for record in records]


def create_lookup(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        record["image_id"]: {
            "image_path": record["full_image_path"],
            "label_vector": record["label_vector_list"],
            "positive_class_ids": record["positive_class_ids_list"],
            "positive_class_names_en": record["positive_class_names_en_list"],
            "positive_class_names_zh": record["positive_class_names_zh_list"],
            "num_positive_labels": record["num_positive_labels"],
            "label_code": record["label_code"],
        }
        for record in records
    }


def create_distribution_rows(
    records: list[dict[str, Any]], cardinality: Counter[int], combinations: Counter[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_code = {record["label_code"]: record for record in records}
    cardinality_rows = [
        {"num_positive_labels": key, "image_count": cardinality[key]}
        for key in sorted(cardinality)
    ]
    combination_rows = []
    for code in sorted(combinations, key=lambda item: (by_code[item]["num_positive_labels"], item)):
        record = by_code[code]
        combination_rows.append({
            "label_code": code,
            "label_vector": record["label_vector"],
            "positive_class_ids": record["positive_class_ids"],
            "positive_class_names_en": record["positive_class_names_en"],
            "positive_class_names_zh": record["positive_class_names_zh"],
            "num_positive_labels": record["num_positive_labels"],
            "image_count": combinations[code],
        })
    return cardinality_rows, combination_rows


def create_class_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for class_id in record["positive_class_ids_list"]:
            rows.append({
                "class_id": class_id,
                "class_name_en": CLASS_INFO[class_id][0],
                "class_name_zh": CLASS_INFO[class_id][1],
                "image_id": record["image_id"],
                "full_image_path": record["full_image_path"],
                "label_code": record["label_code"],
                "all_positive_class_ids": record["positive_class_ids"],
                "all_positive_class_names": record["positive_class_names_en"],
            })
    return sorted(rows, key=lambda item: (item["class_id"], item["image_id"]))


def demo_filename(order: int, record: dict[str, Any]) -> str:
    return f"{order:02d}__{record['image_id']}__GT_{record['label_code']}.png"


def copy_demo_set(
    selected: list[dict[str, Any]],
    output_dir: Path,
    public_output_dir: Path,
    annotation_source: Path,
    copy_images: bool,
) -> list[dict[str, Any]]:
    demo_dir = output_dir / "demo_images"
    sidecar_dir = demo_dir / "sidecar_json"
    demo_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for order, record in enumerate(selected, start=1):
        filename = demo_filename(order, record)
        destination = demo_dir / filename
        public_destination = public_output_dir / "demo_images" / filename
        if copy_images:
            shutil.copy2(record["full_image_path"], destination)
            if sha256_file(destination) != record["image_sha256"]:
                raise RuntimeError(f"Copied Demo SHA256 mismatch: {destination}")
        sidecar = {
            "image_id": record["image_id"],
            "source_image_path": record["full_image_path"],
            "copied_demo_path": str(public_destination.resolve()) if copy_images else "",
            "image_sha256": record["image_sha256"],
            "ground_truth_label_vector": record["label_vector_list"],
            "positive_class_ids": record["positive_class_ids_list"],
            "positive_class_names_en": record["positive_class_names_en_list"],
            "positive_class_names_zh": record["positive_class_names_zh_list"],
            "num_positive_labels": record["num_positive_labels"],
            "annotation_source": str(annotation_source.resolve()),
            "disclaimer": "此 Ground Truth 來自原始資料集標註，用於研究模型輸出比對，不代表新的臨床判讀。",
        }
        write_json(sidecar_dir / f"{Path(filename).stem}.json", sidecar)
        rows.append({
            "demo_order": order,
            "image_id": record["image_id"],
            "source_full_image_path": record["full_image_path"],
            "copied_demo_path": str(public_destination.resolve()) if copy_images else "",
            "image_sha256": record["image_sha256"],
            "label_vector": record["label_vector"],
            "label_code": record["label_code"],
            "positive_class_ids": record["positive_class_ids"],
            "positive_class_names_en": record["positive_class_names_en"],
            "positive_class_names_zh": record["positive_class_names_zh"],
            "num_positive_labels": record["num_positive_labels"],
            "selection_seed": record["selection_seed"],
            "selection_reason": record["selection_reason"],
        })
    return rows


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def create_preview(selected: list[dict[str, Any]], output_dir: Path) -> Path:
    preview_records = selected[:20]
    columns, rows = 4, 5
    tile_width, tile_height = 430, 500
    header_height = 80
    canvas = Image.new("RGB", (columns * tile_width, header_height + rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(27)
    body_font = font(15)
    small_font = font(13)
    draw.text((20, 18), "Full-image Ground Truth Demo Preview", fill="black", font=title_font)
    draw.text((20, 51), "Complete X-rays only; no BBox, ROI crop, prediction, or Ollama output", fill="#444444", font=small_font)

    for index, record in enumerate(preview_records):
        column = index % columns
        row = index // columns
        x0 = column * tile_width
        y0 = header_height + row * tile_height
        draw.rectangle((x0 + 5, y0 + 5, x0 + tile_width - 5, y0 + tile_height - 5), outline="#bbbbbb", width=1)
        with Image.open(record["full_image_path"]) as source:
            source.load()
            image = source.convert("RGB")
            fitted = ImageOps.contain(image, (tile_width - 24, 330), method=Image.Resampling.LANCZOS)
        image_x = x0 + (tile_width - fitted.width) // 2
        image_y = y0 + 12
        canvas.paste(fitted, (image_x, image_y))
        text_y = y0 + 355
        draw.text((x0 + 12, text_y), f"{index + 1:02d}  {record['image_id']}", fill="black", font=body_font)
        label_text = f"GT: {record['positive_class_names_en']}"
        words = label_text.split()
        wrapped: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textlength(candidate, font=small_font) > tile_width - 24:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
        for line_number, line in enumerate(wrapped[:3]):
            draw.text((x0 + 12, text_y + 24 + 18 * line_number), line, fill="#1f4b75", font=small_font)
        vector_y = text_y + 30 + 18 * min(len(wrapped), 3)
        draw.text((x0 + 12, vector_y), f"Vector: {record['label_vector']}", fill="#333333", font=small_font)

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "ground_truth_catalog_preview.png"
    canvas.save(preview_path, format="PNG", optimize=True)
    return preview_path


def image_src_for_gallery(image_path: str, output_dir: Path) -> str:
    relative = os.path.relpath(Path(image_path), output_dir)
    return Path(relative).as_posix()


def create_gallery(records: list[dict[str, Any]], selected: list[dict[str, Any]], output_dir: Path) -> None:
    selected_order = {record["image_id"]: index for index, record in enumerate(selected, start=1)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["label_code"]].append(record)

    parts = ["""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full-image Ground Truth Catalog</title>
<style>
body{margin:0;font-family:Segoe UI,"Microsoft JhengHei",sans-serif;color:#202428;background:#f4f6f7}header{background:#123047;color:white;padding:24px 30px}main{padding:22px 28px}.notice{background:#fff;border-left:5px solid #2b7a78;padding:14px;margin-bottom:22px}.toc a{display:inline-block;margin:4px 8px 4px 0;color:#0b5a7a}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.card{background:white;border:1px solid #ccd3d7;padding:10px}.card.demo{border:3px solid #c17b00}.card img{width:100%;height:245px;object-fit:contain;background:#050505}.id{font-weight:700;overflow-wrap:anywhere;margin:8px 0 4px}.labels{color:#154f72;font-weight:600}.meta,.path{font-size:12px;color:#555;overflow-wrap:anywhere}.badge{display:inline-block;background:#c17b00;color:white;padding:2px 7px;margin-bottom:5px}h2{margin-top:34px;border-bottom:2px solid #78909c;padding-bottom:7px}
</style>
</head>
<body><header><h1>Full-image Multilabel Ground Truth Catalog</h1><p>590 complete chest X-rays indexed from annotations.csv. No model predictions, BBox drawings, or ROI crops.</p></header><main>
<div class="notice"><strong>Ground Truth:</strong> image-level multi-hot labels derived from all annotation rows for each image_id. Demo selection is fixed with seed 42 and is independent of model performance.</div>
<div class="toc"><strong>Label combinations:</strong> """]
    for code in sorted(grouped, key=lambda item: (grouped[item][0]["num_positive_labels"], item)):
        parts.append(f'<a href="#{html.escape(code)}">{html.escape(code)} ({len(grouped[code])})</a>')
    parts.append("</div><h2>Fixed Demo Set (30)</h2><div class=\"grid\">")

    def card(record: dict[str, Any], force_demo: bool = False) -> str:
        demo_order = selected_order.get(record["image_id"])
        is_demo = force_demo or demo_order is not None
        if is_demo and demo_order is not None:
            src = f"demo_images/{html.escape(demo_filename(demo_order, record))}"
            badge = f'<div class="badge">Demo {demo_order:02d}</div>'
        else:
            src = html.escape(image_src_for_gallery(record["full_image_path"], output_dir))
            badge = ""
        return (
            f'<article class="card{" demo" if is_demo else ""}">{badge}'
            f'<img loading="lazy" src="{src}" alt="Full chest X-ray {html.escape(record["image_id"])}">'
            f'<div class="id">{html.escape(record["image_id"])}</div>'
            f'<div class="labels">{html.escape(record["positive_class_names_en"])}</div>'
            f'<div class="meta">Vector {html.escape(record["label_vector"])} | {record["num_positive_labels"]} positive label(s)</div>'
            f'<div class="path">{html.escape(record["full_image_path"])}</div></article>'
        )

    for record in selected:
        parts.append(card(record, force_demo=True))
    parts.append("</div>")
    for code in sorted(grouped, key=lambda item: (grouped[item][0]["num_positive_labels"], item)):
        parts.append(f'<h2 id="{html.escape(code)}">{html.escape(code)} ({len(grouped[code])} images)</h2><div class="grid">')
        for record in grouped[code]:
            parts.append(card(record))
        parts.append("</div>")
    parts.append("</main></body></html>\n")
    (output_dir / "gallery.html").write_text("".join(parts), encoding="utf-8")


def create_readme(output_dir: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    content = f"""# Full-image Multilabel Ground Truth Catalog

Generated from `{args.annotations.resolve()}` and {len(records)} complete raw chest X-rays in `{args.images_dir.resolve()}`.

## Ground Truth

Each `image_id` has one five-element multi-hot Ground Truth vector. Multiple BBox rows for the same class remain one positive image-level label. No image is assigned a single primary class, and no BBox or ROI crop is used.

## Future Gradio Comparison

Compare `predicted_label_vector` with `ground_truth_label_vector` independently for all five classes:

- Ground Truth 1, Prediction 1: TP
- Ground Truth 0, Prediction 1: FP
- Ground Truth 1, Prediction 0: FN
- Ground Truth 0, Prediction 0: TN

For every image, display Ground Truth labels, predicted labels, correctly detected labels, missed labels, extra predicted labels, exact match, sample precision, sample recall, and sample F1. Exact match is Yes only when all five binary labels agree. Top-1 correctness is not a valid evaluation for this multilabel task.

## Ollama Role

Ollama does not receive images and does not validate model correctness. The intended flow is:

`Full image -> Full-image classifier -> five probabilities -> Validation thresholds -> predicted labels -> Ground Truth comparison (dataset images only) -> structured result to Ollama -> text explanation`

Ollama must not modify model predictions, invent diseases, determine Ground Truth, claim a diagnosis, or replace formal metrics.

## Fixed Demo Set

The 30-image Demo set was selected deterministically with seed {args.seed}, without model inference. It covers all five classes and available label cardinalities. Demo files preserve source pixels, resolution, and complete X-ray content. Labels are encoded only in filenames and sidecar JSON; nothing is drawn on the image.

## Research Disclaimer

此 Ground Truth 來自原始資料集標註，用於研究模型輸出比對，不代表新的臨床判讀。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def validate_expected(records: list[dict[str, Any]], source_audit: dict[str, Any], demo: list[dict[str, Any]]) -> None:
    cardinality, _, positives = distribution_data(records)
    failures: list[str] = []
    checks = {
        "raw_image_count": (source_audit["raw_image_count"], 590),
        "annotation_rows": (source_audit["annotation_rows"], 4546),
        "unique_annotation_image_id_count": (source_audit["unique_annotation_image_id_count"], 590),
        "manifest_rows": (len(records), 590),
        "demo_count": (len(demo), 30),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
    for class_id in CLASS_INFO:
        if positives[class_id] != 350:
            failures.append(f"class {class_id} positive images: expected 350, got {positives[class_id]}")
    for field in (
        "missing_image_count",
        "image_without_annotation_count",
        "unreadable_image_count",
        "invalid_label_count",
        "duplicate_image_id_count",
        "duplicate_sha256_group_count",
    ):
        if source_audit[field] != 0:
            failures.append(f"{field}: expected 0, got {source_audit[field]}")
    if any(record["num_positive_labels"] == 0 for record in records):
        failures.append("At least one image has no positive target label")
    demo_classes = Counter(class_id for record in demo for class_id in record["positive_class_ids_list"])
    for class_id in CLASS_INFO:
        if demo_classes[class_id] < 8:
            failures.append(f"Demo class {class_id} coverage is below 8: {demo_classes[class_id]}")
    if set(cardinality) != set(range(1, 6)):
        failures.append(f"Expected label cardinalities 1..5, got {sorted(cardinality)}")
    if {record["num_positive_labels"] for record in demo} != set(range(1, 6)):
        failures.append("Demo does not cover label cardinalities 1..5")
    if failures:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(failures))


def build_outputs(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    source_audit: dict[str, Any],
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    cardinality, combinations, positives = distribution_data(records)
    cardinality_rows, combination_rows = create_distribution_rows(records, cardinality, combinations)

    manifest_path = output_dir / "full_image_ground_truth_manifest.csv"
    write_csv_bom(manifest_path, MANIFEST_FIELDS, manifest_rows(records))
    write_json(output_dir / "image_label_lookup.json", create_lookup(records))
    write_csv_bom(
        output_dir / "label_combination_distribution.csv",
        ["label_code", "label_vector", "positive_class_ids", "positive_class_names_en", "positive_class_names_zh", "num_positive_labels", "image_count"],
        combination_rows,
    )
    write_csv_bom(
        output_dir / "label_cardinality_distribution.csv",
        ["num_positive_labels", "image_count"],
        cardinality_rows,
    )
    class_index = create_class_index(records)
    write_csv_bom(
        output_dir / "class_positive_image_index.csv",
        ["class_id", "class_name_en", "class_name_zh", "image_id", "full_image_path", "label_code", "all_positive_class_ids", "all_positive_class_names"],
        class_index,
    )

    for record in selected:
        record["selection_seed"] = args.seed
    demo_rows = copy_demo_set(
        selected, output_dir, args.output_dir, args.annotations, args.copy_demo_images
    )
    write_csv_bom(
        output_dir / "demo_selection.csv",
        ["demo_order", "image_id", "source_full_image_path", "copied_demo_path", "image_sha256", "label_vector", "label_code", "positive_class_ids", "positive_class_names_en", "positive_class_names_zh", "num_positive_labels", "selection_seed", "selection_reason"],
        demo_rows,
    )
    create_gallery(records, selected, output_dir)
    preview_path = create_preview(selected, output_dir)
    create_readme(output_dir, args, records)

    demo_class_coverage = Counter(class_id for record in selected for class_id in record["positive_class_ids_list"])
    demo_cardinality = Counter(record["num_positive_labels"] for record in selected)
    source_before = aggregate_source_sha(records)
    source_after_records = []
    for record in records:
        updated = dict(record)
        updated["image_sha256"] = sha256_file(Path(record["full_image_path"]))
        source_after_records.append(updated)
    source_after = aggregate_source_sha(source_after_records)
    changed_sources = [
        before["image_id"]
        for before, after in zip(records, source_after_records)
        if before["image_sha256"] != after["image_sha256"]
    ]

    copied_demo_count = len(list((output_dir / "demo_images").glob("*.png")))
    sidecar_count = len(list((output_dir / "demo_images" / "sidecar_json").glob("*.json")))
    audit = {
        "status": "PASS",
        "created_at": utc_now(),
        **source_audit,
        "manifest_rows": len(records),
        "per_class_positive_image_count": {str(class_id): positives[class_id] for class_id in CLASS_INFO},
        "label_cardinality_distribution": {str(key): cardinality[key] for key in sorted(cardinality)},
        "label_combination_count": len(combinations),
        "label_combination_distribution": dict(sorted(combinations.items())),
        "manifest_sha256": sha256_file(manifest_path),
        "demo_selection_count": len(selected),
        "demo_unique_image_id_count": len({record["image_id"] for record in selected}),
        "demo_per_class_coverage": {str(class_id): demo_class_coverage[class_id] for class_id in CLASS_INFO},
        "demo_label_cardinality_distribution": {str(key): demo_cardinality[key] for key in sorted(demo_cardinality)},
        "copied_demo_images_count": copied_demo_count,
        "sidecar_json_count": sidecar_count,
        "preview_image_count": 20,
        "preview_path": str((args.output_dir / "preview" / preview_path.name).resolve()),
        "source_raw_aggregate_sha256_before": source_before,
        "source_raw_aggregate_sha256_after": source_after,
        "source_raw_sha256_unchanged": source_before == source_after and not changed_sources,
        "changed_source_images": changed_sources,
        "used_roi": False,
        "used_bbox_drawing": False,
        "test_inference_count": 0,
        "model_inference_count": 0,
        "ollama_calls": 0,
    }
    if not audit["source_raw_sha256_unchanged"]:
        raise RuntimeError(f"Raw source SHA256 changed: {changed_sources}")
    if copied_demo_count != len(selected) or sidecar_count != len(selected):
        raise RuntimeError(
            f"Demo output mismatch: copied={copied_demo_count}, sidecars={sidecar_count}, expected={len(selected)}"
        )
    write_json(output_dir / "ground_truth_audit.json", audit)
    return audit


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.images_dir = args.images_dir.resolve()
    args.annotations = args.annotations.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.project_root.is_dir():
        raise RuntimeError(f"Project root does not exist: {args.project_root}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        contents = sorted(path.name for path in args.output_dir.iterdir())
        raise RuntimeError(f"Output directory exists and is non-empty; refusing to overwrite: {contents}")

    records, source_audit = inspect_sources(args.images_dir, args.annotations)
    selected = select_demo_records(records, args.demo_count, args.seed)
    validate_expected(records, source_audit, selected)
    cardinality, combinations, positives = distribution_data(records)
    demo_classes = Counter(class_id for record in selected for class_id in record["positive_class_ids_list"])
    demo_cardinality = Counter(record["num_positive_labels"] for record in selected)
    dry_summary = {
        "status": "PASS",
        "dry_run": args.dry_run,
        "raw_image_count": source_audit["raw_image_count"],
        "annotation_rows": source_audit["annotation_rows"],
        "manifest_rows": len(records),
        "per_class_positive_image_count": {str(class_id): positives[class_id] for class_id in CLASS_INFO},
        "label_cardinality_distribution": {str(key): cardinality[key] for key in sorted(cardinality)},
        "label_combination_count": len(combinations),
        "demo_selection_count": len(selected),
        "demo_per_class_coverage": {str(class_id): demo_classes[class_id] for class_id in CLASS_INFO},
        "demo_label_cardinality_distribution": {str(key): demo_cardinality[key] for key in sorted(demo_cardinality)},
        "output_created": False,
        "model_inference_count": 0,
        "ollama_calls": 0,
    }
    if args.dry_run:
        print(json.dumps(dry_summary, ensure_ascii=False, indent=2))
        return 0

    if args.output_dir.exists():
        # The requested output may exist only if it is empty.
        args.output_dir.rmdir()
    staging = args.output_dir.with_name(f"{args.output_dir.name}.writing-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"Staging path already exists: {staging}")
    try:
        audit = build_outputs(args, records, source_audit, selected, staging)
        staging.rename(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    remnants = sorted(
        str(path) for path in args.output_dir.rglob("*") if path.name.endswith((".tmp", ".writing"))
    )
    if remnants:
        raise RuntimeError(f"Temporary remnants found: {remnants}")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(args.output_dir),
        "manifest_rows": audit["manifest_rows"],
        "demo_selection_count": audit["demo_selection_count"],
        "manifest_sha256": audit["manifest_sha256"],
        "source_raw_sha256_unchanged": audit["source_raw_sha256_unchanged"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
