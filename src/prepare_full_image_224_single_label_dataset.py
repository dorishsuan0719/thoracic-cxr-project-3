#!/usr/bin/env python
"""Audit and prepare a full-image 224 single-label five-class dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps


CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_ID_COLUMNS = ("image_id", "source_image_id")
PRIMARY_ID_COLUMNS = (
    "primary_class_id",
    "main_class_id",
    "major_class_id",
    "main_label_id",
)
PRIMARY_NAME_COLUMNS = (
    "primary_class",
    "primary_class_name",
    "main_class",
    "main_class_name",
    "major_class",
    "major_class_name",
    "main_label",
)
CANDIDATE_FIELDS = [
    "image_id",
    "full_image_path",
    "image_filename",
    "image_sha256",
    "source_annotation_file",
    "discovered_class_ids",
    "discovered_class_names",
    "discovered_class_count",
    "primary_class_id",
    "primary_class_name",
    "primary_label_source",
    "is_single_label",
    "conflict_status",
    "include_in_master",
    "exclusion_reason",
]
MASTER_FIELDS = [
    "master_index",
    "image_id",
    "source_image_id",
    "full_image_path",
    "image_filename",
    "image_sha256",
    "class_id",
    "class_name",
    "primary_label_source",
    "original_width",
    "original_height",
    "original_mode",
    "source_annotation_file",
]
SPLIT_FIELDS = [
    "split",
    "image_id",
    "source_image_id",
    "full_image_path",
    "image_sha256",
    "class_id",
    "class_name",
    "master_index",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized(value: Any) -> str:
    return str(value or "").strip()


def bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], []
        return list(reader.fieldnames), list(reader)


def parse_class_id(value: Any) -> int | None:
    text = normalized(value)
    if not text:
        return None
    try:
        number = float(text)
        if not number.is_integer():
            return None
        class_id = int(number)
    except ValueError:
        return None
    return class_id if class_id in CLASS_MAPPING else None


def parse_primary_label(row: dict[str, str], headers: list[str]) -> tuple[int | None, str]:
    lower_to_actual = {header.casefold(): header for header in headers}
    class_id = None
    source_column = ""
    for alias in PRIMARY_ID_COLUMNS:
        actual = lower_to_actual.get(alias.casefold())
        if actual and normalized(row.get(actual)):
            class_id = parse_class_id(row.get(actual))
            source_column = actual
            break
    for alias in PRIMARY_NAME_COLUMNS:
        actual = lower_to_actual.get(alias.casefold())
        if not actual or not normalized(row.get(actual)):
            continue
        value = normalized(row.get(actual))
        by_name = {
            name.casefold(): key for key, name in CLASS_MAPPING.items()
        }.get(value.casefold())
        by_id = parse_class_id(value)
        name_id = by_name if by_name is not None else by_id
        if class_id is not None and name_id is not None and class_id != name_id:
            return None, f"conflicting_primary_columns:{source_column},{actual}"
        if class_id is None:
            class_id = name_id
            source_column = actual
        break
    return class_id, source_column


def discover_files(
    images_dir: Path, annotations_dir: Path, metadata_dir: Path
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    image_files = sorted(
        path.resolve()
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    table_files = sorted(
        {
            path.resolve()
            for root in (annotations_dir, metadata_dir)
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".yaml", ".yml"}
        }
    )
    inventory = []
    for path in table_files:
        headers: list[str] = []
        row_count: int | str = ""
        error = ""
        if path.suffix.lower() == ".csv":
            try:
                headers, rows = read_csv(path)
                row_count = len(rows)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        inventory.append(
            {
                "path": str(path),
                "file_type": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": row_count,
                "columns": "|".join(headers),
                "read_error": error,
            }
        )
    inventory.append(
        {
            "path": str(images_dir.resolve()),
            "file_type": "image_directory",
            "size_bytes": sum(path.stat().st_size for path in image_files),
            "sha256": "",
            "row_count": len(image_files),
            "columns": "",
            "read_error": "",
        }
    )
    return image_files, table_files, inventory


def find_annotation_source(table_files: list[Path]) -> tuple[Path, list[str], list[dict[str, str]]]:
    matches = []
    for path in table_files:
        if path.suffix.lower() != ".csv":
            continue
        headers, rows = read_csv(path)
        names = {header.casefold() for header in headers}
        required = {"image_id", "class_id", "class_name"}
        if required.issubset(names):
            matches.append((path, headers, rows))
    preferred = [item for item in matches if item[0].name.casefold() == "annotations.csv"]
    if len(preferred) == 1:
        return preferred[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError("No CSV with image_id, class_id, and class_name was found")
    raise RuntimeError(f"Ambiguous annotation CSV files: {[str(item[0]) for item in matches]}")


def discover_primary_labels(
    table_files: list[Path], annotation_path: Path
) -> tuple[dict[str, set[tuple[int, str]]], list[dict[str, Any]], list[dict[str, Any]]]:
    claims: dict[str, set[tuple[int, str]]] = defaultdict(set)
    schemas = []
    invalid = []
    for path in table_files:
        if path.suffix.lower() != ".csv":
            continue
        headers, rows = read_csv(path)
        lower_to_actual = {header.casefold(): header for header in headers}
        id_column = next(
            (lower_to_actual[name.casefold()] for name in IMAGE_ID_COLUMNS if name.casefold() in lower_to_actual),
            None,
        )
        primary_columns = [
            lower_to_actual[name.casefold()]
            for name in (*PRIMARY_ID_COLUMNS, *PRIMARY_NAME_COLUMNS)
            if name.casefold() in lower_to_actual
        ]
        schemas.append(
            {
                "path": str(path),
                "columns": headers,
                "row_count": len(rows),
                "image_id_column": id_column,
                "primary_label_columns": primary_columns,
                "used_as_annotation_source": path == annotation_path,
            }
        )
        if not id_column or not primary_columns:
            continue
        for index, row in enumerate(rows, start=2):
            image_id = normalized(row.get(id_column))
            if not image_id:
                continue
            class_id, source_column = parse_primary_label(row, headers)
            if class_id is None:
                if any(normalized(row.get(column)) for column in primary_columns):
                    invalid.append(
                        {
                            "source_file": str(path),
                            "row_number": index,
                            "image_id": image_id,
                            "class_id": "",
                            "class_name": "",
                            "reason": f"invalid_or_conflicting_primary_label:{source_column}",
                        }
                    )
                continue
            claims[image_id].add((class_id, f"{path}:{source_column}"))
    return claims, schemas, invalid


def index_images(image_files: list[Path]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in image_files:
        grouped[path.stem].append(path)
    duplicates = []
    unique = {}
    for image_id, paths in sorted(grouped.items()):
        if len(paths) == 1:
            unique[image_id] = paths[0]
        else:
            for path in paths:
                duplicates.append(
                    {
                        "image_id": image_id,
                        "image_path": str(path),
                        "duplicate_filename_count": len(paths),
                        "reason": "multiple_files_share_image_id_stem",
                    }
                )
    return unique, duplicates


def inspect_images(
    image_files: list[Path], known_full_image_ids: set[str]
) -> tuple[dict[Path, dict[str, Any]], list[dict[str, Any]], dict[str, list[Path]]]:
    properties: dict[Path, dict[str, Any]] = {}
    unreadable = []
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in image_files:
        try:
            file_size = path.stat().st_size
            if file_size <= 0:
                raise ValueError("empty_file")
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format or path.suffix.lstrip(".").upper()
                if width <= 0 or height <= 0:
                    raise ValueError("non_positive_dimensions")
                image_hash = sha256_file(path)
                warning_parts = []
                if path.stem not in known_full_image_ids:
                    warning_parts.append("filename_not_in_annotation_ids")
                lowered = str(path).casefold()
                if any(token in lowered for token in ("bbox_crop", "bbox_crops", "_bbox")):
                    warning_parts.append("path_or_filename_contains_bbox_crop_token")
                if (width, height) == (224, 224):
                    warning_parts.append("source_is_already_224_square_verify_full_image")
                aspect_ratio = width / height
                if max(aspect_ratio, 1.0 / aspect_ratio) > 3.0:
                    warning_parts.append("extreme_aspect_ratio_verify_full_image")
                record = {
                    "image_id": path.stem,
                    "full_image_path": str(path),
                    "original_width": width,
                    "original_height": height,
                    "original_mode": mode,
                    "original_format": image_format,
                    "is_grayscale": mode in {"1", "L", "I", "I;16", "F"},
                    "is_rgb": mode == "RGB",
                    "is_224x224": width == 224 and height == 224,
                    "aspect_ratio": f"{aspect_ratio:.8f}",
                    "file_size_bytes": file_size,
                    "image_sha256": image_hash,
                    "duplicate_sha256_group": "",
                    "warning": "|".join(warning_parts),
                }
                properties[path] = record
                hashes[image_hash].append(path)
        except Exception as exc:
            unreadable.append(
                {
                    "image_id": path.stem,
                    "full_image_path": str(path),
                    "error_type": type(exc).__name__,
                    "error_reason": str(exc),
                }
            )
    duplicate_groups = {
        digest: paths for digest, paths in hashes.items() if len(paths) > 1
    }
    for group_number, (digest, paths) in enumerate(sorted(duplicate_groups.items()), start=1):
        group_id = f"sha256_group_{group_number:04d}"
        for path in paths:
            properties[path]["duplicate_sha256_group"] = group_id
    return properties, unreadable, duplicate_groups


def annotation_labels(
    rows: list[dict[str, str]], annotation_path: Path
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    by_image: dict[str, set[int]] = defaultdict(set)
    invalid = []
    seen_rows = set()
    for row_number, row in enumerate(rows, start=2):
        image_id = normalized(row.get("image_id"))
        class_id = parse_class_id(row.get("class_id"))
        class_name = normalized(row.get("class_name"))
        signature = tuple(sorted((key, normalized(value)) for key, value in row.items()))
        if signature in seen_rows:
            invalid.append(
                {
                    "source_file": str(annotation_path),
                    "row_number": row_number,
                    "image_id": image_id,
                    "class_id": row.get("class_id", ""),
                    "class_name": class_name,
                    "reason": "duplicate_annotation_row",
                }
            )
            continue
        seen_rows.add(signature)
        reason = ""
        if not image_id:
            reason = "missing_image_id"
        elif class_id is None:
            reason = "class_id_not_in_0_to_4"
        elif class_name != CLASS_MAPPING[class_id]:
            reason = "class_name_mapping_mismatch"
        if reason:
            invalid.append(
                {
                    "source_file": str(annotation_path),
                    "row_number": row_number,
                    "image_id": image_id,
                    "class_id": row.get("class_id", ""),
                    "class_name": class_name,
                    "reason": reason,
                }
            )
            continue
        by_image[image_id].add(class_id)
    return by_image, invalid


def choose_primary(
    image_id: str,
    discovered: set[int],
    claims: dict[str, set[tuple[int, str]]],
    annotation_path: Path,
) -> tuple[int | None, str, str]:
    if len(discovered) == 1:
        class_id = next(iter(discovered))
        return class_id, f"single_unique_annotation:{annotation_path}", "SINGLE_LABEL_ANNOTATION"
    image_claims = claims.get(image_id, set())
    claimed_ids = {class_id for class_id, _ in image_claims}
    if len(claimed_ids) == 1:
        class_id = next(iter(claimed_ids))
        if class_id not in discovered:
            return None, "|".join(sorted(source for _, source in image_claims)), "INVALID_PRIMARY_LABEL"
        return (
            class_id,
            "|".join(sorted(source for _, source in image_claims)),
            "RESOLVED_BY_PRIMARY_LABEL",
        )
    if len(claimed_ids) > 1:
        return None, "|".join(sorted(source for _, source in image_claims)), "CONFLICTING_PRIMARY_LABELS"
    return None, "", "NEEDS_LABEL_POLICY"


def build_candidates(
    labels: dict[str, set[int]],
    image_index: dict[str, Path],
    properties: dict[Path, dict[str, Any]],
    claims: dict[str, set[tuple[int, str]]],
    annotation_path: Path,
    duplicate_ids: set[str],
    unreadable_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    conflicts = []
    missing = []
    for image_id in sorted(set(labels) | set(image_index)):
        discovered = labels.get(image_id, set())
        path = image_index.get(image_id)
        primary_id, primary_source, conflict_status = choose_primary(
            image_id, discovered, claims, annotation_path
        ) if discovered else (None, "", "NO_VALID_TARGET_LABEL")
        exclusion = []
        if path is None:
            exclusion.append("missing_image")
            missing.append(
                {
                    "image_id": image_id,
                    "expected_image_filename": image_id,
                    "source_annotation_file": str(annotation_path),
                    "error_reason": "no_matching_full_image_file",
                }
            )
        if image_id in duplicate_ids:
            exclusion.append("duplicate_image_id")
        if image_id in unreadable_ids:
            exclusion.append("unreadable_image")
        if not discovered:
            exclusion.append("no_valid_target_label")
        if primary_id is None:
            exclusion.append(conflict_status.casefold())
        if conflict_status == "NEEDS_LABEL_POLICY":
            conflicts.append(
                {
                    "image_id": image_id,
                    "full_image_path": str(path) if path else "",
                    "discovered_class_ids": "|".join(map(str, sorted(discovered))),
                    "discovered_class_names": "|".join(CLASS_MAPPING[key] for key in sorted(discovered)),
                    "class_combination": "+".join(map(str, sorted(discovered))),
                    "discovered_class_count": len(discovered),
                    "primary_label_source": "",
                    "conflict_status": conflict_status,
                    "required_action": "provide_formal_primary_label_policy",
                }
            )
        include = not exclusion
        image_hash = properties.get(path, {}).get("image_sha256", "") if path else ""
        candidates.append(
            {
                "image_id": image_id,
                "full_image_path": str(path) if path else "",
                "image_filename": path.name if path else "",
                "image_sha256": image_hash,
                "source_annotation_file": str(annotation_path),
                "discovered_class_ids": "|".join(map(str, sorted(discovered))),
                "discovered_class_names": "|".join(CLASS_MAPPING[key] for key in sorted(discovered)),
                "discovered_class_count": len(discovered),
                "primary_class_id": primary_id if primary_id is not None else "",
                "primary_class_name": CLASS_MAPPING[primary_id] if primary_id is not None else "",
                "primary_label_source": primary_source,
                "is_single_label": bool_text(len(discovered) == 1),
                "conflict_status": conflict_status,
                "include_in_master": bool_text(include),
                "exclusion_reason": "|".join(exclusion),
            }
        )
    return candidates, conflicts, missing


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def stratified_split(
    master_rows: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[dict[str, Any]]]:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in master_rows:
        by_class[int(row["class_id"])].append(row)
    output = {"train": [], "val": [], "test": []}
    names = ("train", "val", "test")
    for class_id in sorted(by_class):
        rows = sorted(by_class[class_id], key=lambda row: row["image_id"])
        random.Random(seed + class_id).shuffle(rows)
        counts = allocate_counts(len(rows), ratios)
        start = 0
        for name, count in zip(names, counts):
            output[name].extend(rows[start : start + count])
            start += count
    for name in names:
        output[name].sort(key=lambda row: int(row["master_index"]))
    return output


def create_previews(
    output_dir: Path, master_rows: list[dict[str, Any]], image_size: int
) -> list[str]:
    per_class_dir = output_dir / "preview" / "per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for class_id in CLASS_MAPPING:
        rows = [row for row in master_rows if int(row["class_id"]) == class_id][:5]
        if len(rows) < 5:
            raise RuntimeError(f"Class {class_id} has fewer than five images for preview")
        selected.extend(rows)
    tile_width, tile_height = 280, 294
    canvas = Image.new("RGB", (tile_width * 5, tile_height * 5), "white")
    outputs = []
    for index, row in enumerate(selected):
        with Image.open(row["full_image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        class_id = int(row["class_id"])
        name = f"class{class_id}_{row['image_id']}.png"
        destination = per_class_dir / name
        preview = Image.new("RGB", (tile_width, tile_height), "white")
        preview.paste(resized, ((tile_width - image_size) // 2, 4))
        draw = ImageDraw.Draw(preview)
        draw.text((5, image_size + 10), f"{row['image_id'][:18]} | class {class_id}", fill="black")
        draw.text((5, image_size + 28), CLASS_MAPPING[class_id], fill="black")
        draw.text(
            (5, image_size + 46),
            f"original {row['original_width']}x{row['original_height']} | {Path(row['full_image_path']).stem[:18]}",
            fill="black",
        )
        preview.save(destination, format="PNG")
        canvas.paste(preview, ((index % 5) * tile_width, (index // 5) * tile_height))
        outputs.append(str(destination.resolve()))
    overview = output_dir / "preview" / "full_image_224_preview.png"
    canvas.save(overview, format="PNG")
    outputs.insert(0, str(overview.resolve()))
    return outputs


def leakage_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sets = {
        name: {
            "image_id": {row["image_id"] for row in rows},
            "image_sha256": {row["image_sha256"] for row in rows},
        }
        for name, rows in splits.items()
    }
    intersections = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for field in ("image_id", "image_sha256"):
            key = f"{left}_intersection_{right}_{field}"
            intersections[key] = sorted(sets[left][field] & sets[right][field])
    all_rows = [row for rows in splits.values() for row in rows]
    duplicate_paths = [
        value for value, count in Counter(row["full_image_path"] for row in all_rows).items() if count > 1
    ]
    duplicate_indices = [
        value for value, count in Counter(str(row["master_index"]) for row in all_rows).items() if count > 1
    ]
    image_classes: dict[str, set[int]] = defaultdict(set)
    for row in all_rows:
        image_classes[row["image_id"]].add(int(row["class_id"]))
    conflicting = sorted(image_id for image_id, values in image_classes.items() if len(values) > 1)
    leak_count = sum(len(value) for value in intersections.values())
    return {
        "status": "PASS" if not leak_count and not duplicate_paths and not duplicate_indices and not conflicting else "FAIL",
        **intersections,
        "duplicate_full_image_path": duplicate_paths,
        "duplicate_master_index": duplicate_indices,
        "conflicting_class_labels": conflicting,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    annotations_dir = args.annotations_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata_dir = project_root / "data" / "metadata"
    if not project_root.is_dir() or not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError("Project root, images directory, or annotations directory is missing")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train, val, and test ratios must be positive and sum to 1")
    if args.image_size <= 0:
        raise ValueError("image-size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"STOP: output directory exists and is non-empty: {output_dir}")
        for path in sorted(output_dir.iterdir()):
            print(path)
        return 3
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files, table_files, inventory = discover_files(images_dir, annotations_dir, metadata_dir)
    annotation_path, annotation_headers, annotation_rows = find_annotation_source(table_files)
    primary_claims, schemas, primary_invalid = discover_primary_labels(table_files, annotation_path)
    labels, annotation_invalid = annotation_labels(annotation_rows, annotation_path)
    image_index, duplicate_id_rows = index_images(image_files)
    properties, unreadable_rows, duplicate_hash_groups = inspect_images(image_files, set(labels))
    duplicate_ids = {row["image_id"] for row in duplicate_id_rows}
    unreadable_ids = {row["image_id"] for row in unreadable_rows}
    candidates, conflicts, missing_rows = build_candidates(
        labels,
        image_index,
        properties,
        primary_claims,
        annotation_path,
        duplicate_ids,
        unreadable_ids,
    )
    invalid_rows = annotation_invalid + primary_invalid
    unresolved_count = len(conflicts)
    potential_master = [row for row in candidates if row["include_in_master"] == "TRUE"]
    status = "NEEDS_LABEL_POLICY" if unresolved_count else "PASS"
    if invalid_rows or duplicate_id_rows:
        status = "FAIL"

    inventory_fields = [
        "path", "file_type", "size_bytes", "sha256", "row_count", "columns", "read_error"
    ]
    write_csv(output_dir / "source_file_inventory.csv", inventory_fields, inventory)
    write_json(
        output_dir / "discovered_schema.json",
        {
            "status": status,
            "created_at": utc_now(),
            "annotation_source": str(annotation_path),
            "annotation_columns": annotation_headers,
            "annotation_rows": len(annotation_rows),
            "searched_tables": schemas,
            "primary_label_aliases": {
                "id_columns": list(PRIMARY_ID_COLUMNS),
                "name_columns": list(PRIMARY_NAME_COLUMNS),
            },
            "primary_label_claim_image_count": len(primary_claims),
        },
    )
    write_csv(output_dir / "full_image_candidate_manifest.csv", CANDIDATE_FIELDS, candidates)
    write_csv(
        output_dir / "label_conflicts.csv",
        [
            "image_id", "full_image_path", "discovered_class_ids", "discovered_class_names",
            "class_combination", "discovered_class_count", "primary_label_source",
            "conflict_status", "required_action",
        ],
        conflicts,
    )
    write_csv(
        output_dir / "duplicate_image_ids.csv",
        ["image_id", "image_path", "duplicate_filename_count", "reason"],
        duplicate_id_rows,
    )
    write_csv(
        output_dir / "missing_images.csv",
        ["image_id", "expected_image_filename", "source_annotation_file", "error_reason"],
        missing_rows,
    )
    write_csv(
        output_dir / "unreadable_images.csv",
        ["image_id", "full_image_path", "error_type", "error_reason"],
        unreadable_rows,
    )
    write_csv(
        output_dir / "invalid_labels.csv",
        ["source_file", "row_number", "image_id", "class_id", "class_name", "reason"],
        invalid_rows,
    )
    property_fields = [
        "image_id", "full_image_path", "original_width", "original_height", "original_mode",
        "original_format", "is_grayscale", "is_rgb", "is_224x224", "aspect_ratio",
        "file_size_bytes", "image_sha256", "duplicate_sha256_group", "warning",
    ]
    write_csv(
        output_dir / "image_properties.csv",
        property_fields,
        [properties[path] for path in sorted(properties)],
    )
    write_json(
        output_dir / "preprocessing_spec.json",
        {
            "task": "full-image single-label five-class classification",
            "uses_bbox": False,
            "uses_roi_crop": False,
            "source": "complete raw/full chest X-ray",
            "steps": [
                "PIL open full image",
                "Convert RGB",
                f"Resize full image directly to {args.image_size}x{args.image_size}",
                "ToTensor",
                "ImageNet mean/std normalization",
            ],
            "resize": {
                "size": [args.image_size, args.image_size],
                "interpolation": "BILINEAR",
                "antialias": True,
                "preserve_full_field_of_view": True,
                "center_crop": False,
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "expected_tensor_shape": [3, args.image_size, args.image_size],
            "source_images_modified": False,
            "augmentation_applied": False,
        },
    )

    class_distribution = []
    conflict_ids = {row["image_id"] for row in conflicts}
    for class_id, class_name in CLASS_MAPPING.items():
        candidate_count = sum(
            class_id in labels.get(row["image_id"], set()) for row in candidates
        )
        conflict_count = sum(
            class_id in labels.get(image_id, set()) for image_id in conflict_ids
        )
        master_count = 0 if args.dry_run or status != "PASS" else sum(
            int(row["primary_class_id"]) == class_id for row in potential_master
        )
        class_distribution.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "candidate_images": candidate_count,
                "conflict_images": conflict_count,
                "excluded_images": sum(
                    class_id in labels.get(row["image_id"], set())
                    and row["include_in_master"] == "FALSE"
                    for row in candidates
                ),
                "master_images": master_count,
                "train_images": 0,
                "val_images": 0,
                "test_images": 0,
            }
        )

    master_rows: list[dict[str, Any]] = []
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    leakage: dict[str, Any] = {"status": "NOT_RUN", "reason": "dry_run_or_unresolved_labels"}
    previews: list[str] = []
    if not args.dry_run and status == "PASS":
        for master_index, candidate in enumerate(
            sorted(potential_master, key=lambda row: row["image_id"])
        ):
            path = Path(candidate["full_image_path"])
            prop = properties[path]
            class_id = int(candidate["primary_class_id"])
            master_rows.append(
                {
                    "master_index": master_index,
                    "image_id": candidate["image_id"],
                    "source_image_id": candidate["image_id"],
                    "full_image_path": candidate["full_image_path"],
                    "image_filename": candidate["image_filename"],
                    "image_sha256": candidate["image_sha256"],
                    "class_id": class_id,
                    "class_name": CLASS_MAPPING[class_id],
                    "primary_label_source": candidate["primary_label_source"],
                    "original_width": prop["original_width"],
                    "original_height": prop["original_height"],
                    "original_mode": prop["original_mode"],
                    "source_annotation_file": str(annotation_path),
                }
            )
        splits = stratified_split(master_rows, ratios, args.seed)
        leakage = leakage_audit(splits)
        if leakage["status"] != "PASS":
            status = "FAIL"
        write_csv(output_dir / "full_image_master_manifest.csv", MASTER_FIELDS, master_rows)
        for split_name in ("train", "val", "test"):
            rows = []
            for row in splits[split_name]:
                rows.append({"split": split_name, **row})
            write_csv(output_dir / f"{split_name}_manifest.csv", SPLIT_FIELDS, rows)
        previews = create_previews(output_dir, master_rows, args.image_size)
        for distribution in class_distribution:
            class_id = int(distribution["class_id"])
            distribution["master_images"] = sum(
                int(row["class_id"]) == class_id for row in master_rows
            )
            for split_name in ("train", "val", "test"):
                distribution[f"{split_name}_images"] = sum(
                    int(row["class_id"]) == class_id for row in splits[split_name]
                )
        split_summary = []
        for split_name in ("train", "val", "test"):
            for class_id, class_name in CLASS_MAPPING.items():
                split_summary.append(
                    {
                        "split": split_name,
                        "class_id": class_id,
                        "class_name": class_name,
                        "image_count": sum(
                            int(row["class_id"]) == class_id for row in splits[split_name]
                        ),
                    }
                )
        write_csv(
            output_dir / "split_summary.csv",
            ["split", "class_id", "class_name", "image_count"],
            split_summary,
        )
        write_json(output_dir / "leakage_audit.json", leakage)

    write_csv(
        output_dir / "class_distribution.csv",
        [
            "class_id", "class_name", "candidate_images", "conflict_images",
            "excluded_images", "master_images", "train_images", "val_images", "test_images",
        ],
        class_distribution,
    )
    duplicate_hash_image_count = sum(len(paths) for paths in duplicate_hash_groups.values())
    audit = {
        "status": status,
        "dry_run": args.dry_run,
        "created_at": utc_now(),
        "task": "full-image 224x224 single-label five-class dataset",
        "full_image_source": str(images_dir),
        "annotation_source": str(annotation_path),
        "image_file_count": len(image_files),
        "annotation_row_count": len(annotation_rows),
        "candidate_unique_image_id_count": len(candidates),
        "single_annotation_class_image_count": sum(
            int(row["discovered_class_count"]) == 1 for row in candidates
        ),
        "multi_annotation_class_image_count": sum(
            int(row["discovered_class_count"]) > 1 for row in candidates
        ),
        "unresolved_label_conflict_count": unresolved_count,
        "formal_primary_label_claim_image_count": len(primary_claims),
        "potential_includable_image_count": len(potential_master),
        "master_image_count": len(master_rows),
        "missing_image_count": len(missing_rows),
        "unreadable_image_count": len(unreadable_rows),
        "duplicate_image_id_row_count": len(duplicate_id_rows),
        "duplicate_sha256_group_count": len(duplicate_hash_groups),
        "duplicate_sha256_image_count": duplicate_hash_image_count,
        "invalid_label_row_count": len(invalid_rows),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "leakage": leakage,
        "uses_roi_crop": False,
        "augmentation_applied": False,
        "model_inference_executed": False,
        "source_images_modified": False,
        "formal_training_ready": status == "PASS" and not args.dry_run,
        "preview_files": previews,
    }
    write_json(output_dir / "dataset_audit.json", audit)

    combination_counts = Counter(
        row["class_combination"] for row in conflicts
    )
    summary_lines = [
        "# Full-image 224 Single-label Dataset Audit",
        "",
        f"- Status: **{status}**",
        "- Final task: full-image single-label five-class classification.",
        "- Model input does not use BBox or ROI crops.",
        f"- Raw full images are resized directly to {args.image_size}x{args.image_size} during preprocessing.",
        f"- Full images found: {len(image_files)}",
        f"- Annotation rows: {len(annotation_rows)}",
        f"- Candidate unique image IDs: {len(candidates)}",
        f"- Single annotation-class image IDs: {sum(int(row['discovered_class_count']) == 1 for row in candidates)}",
        f"- Multi-label conflicts without formal primary label: {unresolved_count}",
        f"- Formal primary label claim image IDs found: {len(primary_claims)}",
        f"- Missing images: {len(missing_rows)}",
        f"- Unreadable images: {len(unreadable_rows)}",
        f"- Duplicate image IDs: {len(duplicate_ids)}",
        f"- Duplicate SHA256 groups/images: {len(duplicate_hash_groups)}/{duplicate_hash_image_count}",
        f"- Master Dataset images: {len(master_rows)}",
        f"- Train/Validation/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
        f"- Leakage: {leakage['status']}",
        "- Augmentation: not applied.",
        "- Model inference: not executed.",
        "- Existing ROI Patch work remains a comparison experiment only.",
        "",
        "## Candidate Class Distribution",
        "",
    ]
    for row in class_distribution:
        summary_lines.append(
            f"- Class {row['class_id']} {row['class_name']}: candidates {row['candidate_images']}, "
            f"conflicts {row['conflict_images']}, master {row['master_images']}"
        )
    if combination_counts:
        summary_lines.extend(["", "## Unresolved Conflict Combinations", ""])
        for combination, count in sorted(
            combination_counts.items(), key=lambda item: (item[0].count("+") + 1, item[0])
        ):
            names = " + ".join(CLASS_MAPPING[int(value)] for value in combination.split("+"))
            summary_lines.append(f"- {combination} ({names}): {count}")
    summary_lines.extend(
        [
            "",
            "## Training Readiness",
            "",
            (
                "A formal primary-label policy is required before Master Dataset creation or splitting."
                if status == "NEEDS_LABEL_POLICY"
                else "The dataset audit determines whether formal full-image training may proceed."
            ),
            "",
        ]
    )
    (output_dir / "dataset_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 2 if status == "NEEDS_LABEL_POLICY" else (1 if status == "FAIL" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
