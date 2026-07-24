#!/usr/bin/env python
"""Create the shared deterministic grouped Phase 2 8:1:1 split.

This script performs metadata-only processing. It does not import or load any
neural network, copy images, train a model, or modify Phase 0/Phase 1 outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


EXPECTED_BALANCED_MANIFEST_SHA256 = (
    "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
)
EXPECTED_SOURCE_COUNT = 590
EXPECTED_ORIGINAL_ROI_COUNT = 4546
EXPECTED_BALANCED_ROI_COUNT = 4725
EXPECTED_BALANCED_ORIGINAL_COUNT = 4256
EXPECTED_BALANCED_AUGMENTED_COUNT = 469
EXPECTED_SPLIT_SOURCE_COUNTS = {"train": 472, "val": 59, "test": 59}
EXPECTED_CLASS_IMAGE_COUNTS = {class_id: 350 for class_id in range(5)}
EXPECTED_ORIGINAL_CLASS_ROI_COUNTS = {0: 772, 1: 783, 2: 1118, 3: 1062, 4: 811}
TARGET_CLASS_IMAGE_BY_SPLIT = {
    "train": {class_id: 280 for class_id in range(5)},
    "val": {class_id: 35 for class_id in range(5)},
    "test": {class_id: 35 for class_id in range(5)},
}

CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}

CLASS_FOLDERS = {
    0: "0_aortic_enlargement",
    1: "1_cardiomegaly",
    2: "2_pleural_thickening",
    3: "3_pulmonary_fibrosis",
    4: "4_pleural_effusion",
}

ORIGINAL_FILENAME_RE = re.compile(
    r"^(?P<source_image_id>.+)_class(?P<class_id>[0-4])_rad(?P<rad_id>[^_]+)_bbox(?P<bbox_index>\d{4})\.png$"
)

ORIGINAL_FIELDS = [
    "original_record_index",
    "image_path",
    "relative_path",
    "filename",
    "class_id",
    "class_name",
    "source_image_id",
    "original_roi_id",
    "file_size_bytes",
    "image_width",
    "image_height",
    "image_mode",
    "image_sha256",
]

IMAGE_SPLIT_FIELDS = [
    "source_image_id",
    "split",
    "super_group_id",
    "class_0_present",
    "class_1_present",
    "class_2_present",
    "class_3_present",
    "class_4_present",
    "class_count",
    "is_multilabel",
    "original_roi_count",
    "balanced_roi_count",
    "balanced_augmented_count",
]

ROI_SPLIT_FIELDS = [
    "record_index",
    "source_image_id",
    "split",
    "image_path",
    "relative_path",
    "filename",
    "class_id",
    "class_name",
    "original_roi_id",
    "is_brightness_augmented",
    "image_sha256",
    "source_manifest",
    "source_record_index",
]

CLASS_COUNT_FIELDS = [
    "split",
    "class_id",
    "class_name",
    "source_image_count",
    "target_source_image_count",
    "difference_from_target",
]

ROI_COUNT_FIELDS = [
    "split",
    "class_id",
    "class_name",
    "total_roi_count",
    "original_roi_count",
    "brightness_augmented_roi_count",
]

DUPLICATE_FIELDS = [
    "image_sha256",
    "combined_record_count",
    "unique_path_count",
    "original_record_count",
    "balanced_record_count",
    "source_image_count",
    "source_image_ids",
    "is_cross_source",
    "super_group_ids",
]

LEAKAGE_FIELDS = ["audit_name", "leakage_count", "status", "details"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
    expected_rows: int | None = None,
) -> None:
    materialized = list(rows)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    reread = read_csv(temporary)
    if len(reread) != len(materialized):
        raise RuntimeError(f"Atomic CSV verification failed for {path.name}")
    if expected_rows is not None and len(reread) != expected_rows:
        raise RuntimeError(f"Unexpected row count for {path.name}: {len(reread)}")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    if temporary.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Atomic text verification failed for {path.name}")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def validate_balanced_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Balanced manifest not found: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_BALANCED_MANIFEST_SHA256:
        raise ValueError(f"Balanced manifest SHA256 mismatch: {digest}")
    rows = read_csv(path)
    if len(rows) != EXPECTED_BALANCED_ROI_COUNT:
        raise ValueError(f"Balanced manifest rows {len(rows)} != {EXPECTED_BALANCED_ROI_COUNT}")
    indices = [int(row["feature_index"]) for row in rows]
    if indices != list(range(EXPECTED_BALANCED_ROI_COUNT)):
        raise ValueError("Balanced feature_index is not exactly 0..4724")
    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Balanced manifest has {len(missing)} missing images")
    class_counts = Counter(int(row["class_id"]) for row in rows)
    if class_counts != Counter({class_id: 945 for class_id in CLASS_MAPPING}):
        raise ValueError(f"Balanced class counts are invalid: {dict(class_counts)}")
    augmented = sum(parse_bool(row["is_brightness_augmented"]) for row in rows)
    original = len(rows) - augmented
    if (original, augmented) != (
        EXPECTED_BALANCED_ORIGINAL_COUNT,
        EXPECTED_BALANCED_AUGMENTED_COUNT,
    ):
        raise ValueError(f"Balanced original/augmented counts are {original}/{augmented}")
    return rows, {
        "sha256": digest,
        "rows": len(rows),
        "original_count": original,
        "augmented_count": augmented,
        "class_counts": dict(sorted(class_counts.items())),
    }


def scan_original_rois(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    paths: list[tuple[int, Path]] = []
    for class_id in sorted(CLASS_FOLDERS):
        class_dir = root / CLASS_FOLDERS[class_id]
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Original ROI class folder missing: {class_dir}")
        for path in sorted(class_dir.iterdir(), key=lambda item: item.name):
            if path.is_file():
                paths.append((class_id, path))
    for class_id, path in paths:
        error = None
        match = ORIGINAL_FILENAME_RE.match(path.name) if path.suffix.lower() == ".png" else None
        if not match:
            error = "filename_parse_failed"
        elif int(match.group("class_id")) != class_id:
            error = "class_folder_filename_mismatch"
        if error:
            unresolved.append(
                {"image_path": str(path.resolve()), "filename": path.name, "error_reason": error}
            )
            continue
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
        except Exception as exc:
            unresolved.append(
                {
                    "image_path": str(path.resolve()),
                    "filename": path.name,
                    "error_reason": f"unreadable_image: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if (width, height, mode) != (224, 224, "L"):
            unresolved.append(
                {
                    "image_path": str(path.resolve()),
                    "filename": path.name,
                    "error_reason": f"unexpected_image_properties:{width}x{height}:{mode}",
                }
            )
            continue
        source_image_id = match.group("source_image_id")
        records.append(
            {
                "image_path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "filename": path.name,
                "class_id": class_id,
                "class_name": CLASS_MAPPING[class_id],
                "source_image_id": source_image_id,
                "original_roi_id": path.stem,
                "file_size_bytes": path.stat().st_size,
                "image_width": width,
                "image_height": height,
                "image_mode": mode,
                "image_sha256": sha256_file(path),
            }
        )
    records.sort(key=lambda row: (int(row["class_id"]), row["filename"]))
    for index, row in enumerate(records):
        row["original_record_index"] = index
    return records, unresolved


def validate_formal_sources(
    project_root: Path,
    original_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    annotations_path = project_root / "data" / "raw" / "annotations" / "annotations.csv"
    full_image_dir = project_root / "data" / "raw" / "images"
    image_manifest_path = project_root / "data" / "metadata" / "image_manifest.csv"
    annotations = read_csv(annotations_path)
    if len(annotations) != EXPECTED_ORIGINAL_ROI_COUNT:
        raise ValueError(f"Formal annotation rows {len(annotations)} != 4546")
    annotation_sources = {row["image_id"] for row in annotations}
    if len(annotation_sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"Formal annotation source count {len(annotation_sources)} != 590")
    annotation_class_roi = Counter(int(row["class_id"]) for row in annotations)
    if annotation_class_roi != Counter(EXPECTED_ORIGINAL_CLASS_ROI_COUNTS):
        raise ValueError(f"Formal annotation class ROI counts mismatch: {dict(annotation_class_roi)}")
    annotation_presence = defaultdict(set)
    duplicate_keys = []
    seen_keys = set()
    invalid_bbox_count = 0
    for row in annotations:
        class_id = int(row["class_id"])
        if row["class_name"] != CLASS_MAPPING[class_id]:
            raise ValueError("Formal annotation class mapping mismatch")
        annotation_presence[row["image_id"]].add(class_id)
        coordinates = [float(row[name]) for name in ("x_min", "y_min", "x_max", "y_max")]
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            invalid_bbox_count += 1
        key = tuple(row.get(name, "") for name in row.keys())
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
    if invalid_bbox_count or duplicate_keys:
        raise ValueError(
            f"Formal annotation invalid/duplicate counts: {invalid_bbox_count}/{len(duplicate_keys)}"
        )
    class_image_counts = Counter(
        class_id for classes in annotation_presence.values() for class_id in classes
    )
    if class_image_counts != Counter(EXPECTED_CLASS_IMAGE_COUNTS):
        raise ValueError(f"Formal class-image counts mismatch: {dict(class_image_counts)}")

    full_images = sorted(full_image_dir.glob("*.png"), key=lambda path: path.name)
    full_image_sources = {path.stem for path in full_images}
    original_sources = {row["source_image_id"] for row in original_rows}
    image_manifest_sources = {row["source_image_id"] for row in read_csv(image_manifest_path)}
    if not (
        len(full_images) == EXPECTED_SOURCE_COUNT
        and original_sources == annotation_sources == full_image_sources == image_manifest_sources
    ):
        raise ValueError("Source IDs differ among full images, annotations, metadata, and ROIs")
    return {
        "annotations_path": str(annotations_path),
        "full_image_dir": str(full_image_dir),
        "image_manifest_path": str(image_manifest_path),
        "source_ids": sorted(original_sources),
        "annotation_presence": annotation_presence,
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "class_roi_counts": dict(sorted(annotation_class_roi.items())),
    }


def build_sha_super_groups(
    source_ids: list[str],
    original_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any], dict[str, list[str]]]:
    sha_records: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in original_rows:
        sha_records[row["image_sha256"]].append(
            ("original", row["image_path"], row["source_image_id"])
        )
    for row in balanced_rows:
        sha_records[row["image_sha256"]].append(
            ("balanced", row["image_path"], row["source_image_id"])
        )
    union_find = UnionFind(source_ids)
    cross_source_hashes: dict[str, set[str]] = {}
    balanced_duplicate_group_count = 0
    balanced_sha_counts = Counter(row["image_sha256"] for row in balanced_rows)
    balanced_duplicate_group_count = sum(count > 1 for count in balanced_sha_counts.values())
    for digest, records in sha_records.items():
        sources = sorted({record[2] for record in records})
        if len(sources) > 1:
            cross_source_hashes[digest] = set(sources)
            for source_id in sources[1:]:
                union_find.union(sources[0], source_id)
    components: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        components[union_find.find(source_id)].append(source_id)
    ordered_components = sorted((sorted(values) for values in components.values()), key=lambda x: x[0])
    source_to_super_group: dict[str, str] = {}
    component_map: dict[str, list[str]] = {}
    for index, members in enumerate(ordered_components):
        super_group_id = f"sg_{index:04d}"
        component_map[super_group_id] = members
        for source_id in members:
            source_to_super_group[source_id] = super_group_id

    duplicate_rows = []
    for digest in sorted(sha_records):
        records = sha_records[digest]
        unique_paths = sorted({record[1] for record in records})
        if len(records) <= 1 and len(unique_paths) <= 1:
            continue
        sources = sorted({record[2] for record in records})
        duplicate_rows.append(
            {
                "image_sha256": digest,
                "combined_record_count": len(records),
                "unique_path_count": len(unique_paths),
                "original_record_count": sum(record[0] == "original" for record in records),
                "balanced_record_count": sum(record[0] == "balanced" for record in records),
                "source_image_count": len(sources),
                "source_image_ids": ";".join(sources),
                "is_cross_source": len(sources) > 1,
                "super_group_ids": ";".join(
                    sorted({source_to_super_group[source] for source in sources})
                ),
            }
        )
    stats = {
        "balanced_duplicate_sha256_group_count": balanced_duplicate_group_count,
        "combined_duplicate_sha256_group_count": len(duplicate_rows),
        "cross_source_duplicate_sha256_group_count": len(cross_source_hashes),
        "super_group_count": len(component_map),
        "non_singleton_super_group_count": sum(len(values) > 1 for values in component_map.values()),
        "largest_super_group_size": max(map(len, component_map.values())),
    }
    return source_to_super_group, duplicate_rows, stats, component_map


def choose_subset_by_size(
    candidate_indices: list[int],
    entity_sizes: np.ndarray,
    target: int,
    rng: random.Random,
) -> set[int] | None:
    ordered = candidate_indices.copy()
    rng.shuffle(ordered)
    predecessors: dict[int, tuple[int, int]] = {0: (-1, -1)}
    for entity_index in ordered:
        size = int(entity_sizes[entity_index])
        for subtotal in sorted(list(predecessors.keys()), reverse=True):
            next_total = subtotal + size
            if next_total <= target and next_total not in predecessors:
                predecessors[next_total] = (subtotal, entity_index)
        if target in predecessors:
            break
    if target not in predecessors:
        return None
    selected: set[int] = set()
    value = target
    while value:
        previous, entity_index = predecessors[value]
        selected.add(entity_index)
        value = previous
    return selected


def optimize_split(
    source_ids: list[str],
    source_to_super_group: dict[str, str],
    component_map: dict[str, list[str]],
    presence_by_source: dict[str, set[int]],
    roi_by_source_class: dict[str, list[int]],
    balanced_source_ids: set[str],
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    source_position = {source_id: index for index, source_id in enumerate(source_ids)}
    presence = np.zeros((len(source_ids), 5), dtype=np.int16)
    roi = np.zeros((len(source_ids), 5), dtype=np.int32)
    for source_id in source_ids:
        index = source_position[source_id]
        for class_id in presence_by_source[source_id]:
            presence[index, class_id] = 1
        roi[index] = roi_by_source_class[source_id]

    entity_ids = sorted(component_map)
    entity_members = [[source_position[source] for source in component_map[group]] for group in entity_ids]
    entity_sizes = np.array([len(members) for members in entity_members], dtype=np.int16)
    entity_presence = np.array([presence[members].sum(axis=0) for members in entity_members])
    entity_roi = np.array([roi[members].sum(axis=0) for members in entity_members])
    target_presence = np.array([[280] * 5, [35] * 5, [35] * 5], dtype=np.int32)
    rng = random.Random(seed)
    best: tuple[int, np.ndarray, np.ndarray, int, int] | None = None

    for restart in range(30):
        entity_count = len(entity_ids)
        if np.all(entity_sizes == 1):
            shuffled = list(range(entity_count))
            rng.shuffle(shuffled)
            assignment = np.empty(entity_count, dtype=np.int8)
            assignment[shuffled[:472]] = 0
            assignment[shuffled[472:531]] = 1
            assignment[shuffled[531:]] = 2
        else:
            all_indices = list(range(entity_count))
            validation = choose_subset_by_size(all_indices, entity_sizes, 59, rng)
            if validation is None:
                raise RuntimeError("INFEASIBLE_SUPER_GROUP_CARDINALITY: validation size 59")
            remaining = [index for index in all_indices if index not in validation]
            test = choose_subset_by_size(remaining, entity_sizes, 59, rng)
            if test is None:
                continue
            assignment = np.zeros(entity_count, dtype=np.int8)
            assignment[list(validation)] = 1
            assignment[list(test)] = 2
        counts = np.array(
            [entity_presence[assignment == split_index].sum(axis=0) for split_index in range(3)]
        )
        score = int(np.sum((counts - target_presence) ** 2))
        temperature = 3.0
        for step in range(300000):
            left = rng.randrange(entity_count)
            right = rng.randrange(entity_count)
            left_split = int(assignment[left])
            right_split = int(assignment[right])
            if left_split == right_split or entity_sizes[left] != entity_sizes[right]:
                continue
            old_score = np.sum((counts[left_split] - target_presence[left_split]) ** 2) + np.sum(
                (counts[right_split] - target_presence[right_split]) ** 2
            )
            new_left = counts[left_split] - entity_presence[left] + entity_presence[right]
            new_right = counts[right_split] - entity_presence[right] + entity_presence[left]
            new_score = np.sum((new_left - target_presence[left_split]) ** 2) + np.sum(
                (new_right - target_presence[right_split]) ** 2
            )
            delta = int(new_score - old_score)
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                assignment[left], assignment[right] = right_split, left_split
                counts[left_split] = new_left
                counts[right_split] = new_right
                score += delta
            temperature = max(0.02, temperature * 0.99997)
            if score == 0:
                break
        candidate = (score, assignment.copy(), counts.copy(), restart, step)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if score == 0:
            break
    if best is None or best[0] != 0:
        raise RuntimeError(
            f"INFEASIBLE_EXACT_CLASS_PRESENCE: best score={None if best is None else best[0]}"
        )

    assignment = best[1]
    entity_train_eligible = np.array(
        [
            all(source_ids[source_index] in balanced_source_ids for source_index in members)
            for members in entity_members
        ],
        dtype=bool,
    )
    forced_non_train_swaps = 0
    for entity_index in range(len(entity_ids)):
        if entity_train_eligible[entity_index] or int(assignment[entity_index]) != 0:
            continue
        candidates = [
            candidate
            for candidate in range(len(entity_ids))
            if int(assignment[candidate]) in {1, 2}
            and entity_train_eligible[candidate]
            and entity_sizes[candidate] == entity_sizes[entity_index]
            and np.array_equal(entity_presence[candidate], entity_presence[entity_index])
        ]
        if not candidates:
            raise RuntimeError(
                "INFEASIBLE_TRAIN_SOURCE_COVERAGE: no same-presence validation/test swap"
            )
        candidate = min(candidates, key=lambda value: entity_ids[value])
        assignment[entity_index], assignment[candidate] = (
            assignment[candidate],
            assignment[entity_index],
        )
        forced_non_train_swaps += 1
    roi_target = np.array(
        [roi.sum(axis=0) * 0.8, roi.sum(axis=0) * 0.1, roi.sum(axis=0) * 0.1]
    )
    roi_counts = np.array(
        [entity_roi[assignment == split_index].sum(axis=0) for split_index in range(3)],
        dtype=float,
    )

    def roi_objective(values: np.ndarray) -> float:
        return float(np.sum(((values - roi_target) ** 2) / (roi_target + 1.0)))

    secondary_initial = roi_objective(roi_counts)
    equivalent_groups: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for entity_index in range(len(entity_ids)):
        key = (int(entity_sizes[entity_index]), tuple(entity_presence[entity_index].tolist()))
        equivalent_groups[key].append(entity_index)
    swappable_groups = [values for values in equivalent_groups.values() if len(values) > 1]
    accepted_swaps = 0
    for _ in range(500000):
        values = swappable_groups[rng.randrange(len(swappable_groups))]
        left = values[rng.randrange(len(values))]
        right = values[rng.randrange(len(values))]
        left_split = int(assignment[left])
        right_split = int(assignment[right])
        if left_split == right_split:
            continue
        if (right_split == 0 and not entity_train_eligible[left]) or (
            left_split == 0 and not entity_train_eligible[right]
        ):
            continue
        old_score = np.sum(
            ((roi_counts[left_split] - roi_target[left_split]) ** 2)
            / (roi_target[left_split] + 1.0)
        ) + np.sum(
            ((roi_counts[right_split] - roi_target[right_split]) ** 2)
            / (roi_target[right_split] + 1.0)
        )
        new_left = roi_counts[left_split] - entity_roi[left] + entity_roi[right]
        new_right = roi_counts[right_split] - entity_roi[right] + entity_roi[left]
        new_score = np.sum(
            ((new_left - roi_target[left_split]) ** 2) / (roi_target[left_split] + 1.0)
        ) + np.sum(
            ((new_right - roi_target[right_split]) ** 2) / (roi_target[right_split] + 1.0)
        )
        tie_break = entity_ids[right] < entity_ids[left]
        if new_score < old_score - 1e-12 or (abs(new_score - old_score) < 1e-12 and tie_break):
            assignment[left], assignment[right] = right_split, left_split
            roi_counts[left_split] = new_left
            roi_counts[right_split] = new_right
            accepted_swaps += 1

    split_names = ["train", "val", "test"]
    source_split: dict[str, str] = {}
    for entity_index, super_group_id in enumerate(entity_ids):
        split = split_names[int(assignment[entity_index])]
        for source_id in component_map[super_group_id]:
            source_split[source_id] = split
    final_presence = np.array(
        [
            presence[[source_position[source] for source in source_ids if source_split[source] == split]].sum(
                axis=0
            )
            for split in split_names
        ]
    )
    source_counts = Counter(source_split.values())
    if source_counts != Counter(EXPECTED_SPLIT_SOURCE_COUNTS):
        raise RuntimeError(f"Exact source split count failed: {dict(source_counts)}")
    if not np.array_equal(final_presence, target_presence):
        raise RuntimeError(f"Exact class presence failed: {final_presence.tolist()}")
    return source_split, {
        "primary_score": int(best[0]),
        "primary_restart": int(best[3]),
        "primary_steps": int(best[4]) + 1,
        "secondary_initial_objective": secondary_initial,
        "secondary_final_objective": roi_objective(roi_counts),
        "secondary_accepted_swaps": accepted_swaps,
        "forced_non_train_source_swaps": forced_non_train_swaps,
        "class_presence_counts": final_presence.tolist(),
        "original_roi_counts": roi_counts.astype(int).tolist(),
        "original_roi_targets": roi_target.tolist(),
        "strategy": (
            "deterministic grouped multi-label fixed-cardinality swap optimization; "
            "primary exact class presence, secondary original ROI proportionality, "
            "source_image_id lexical tie-break"
        ),
    }


def build_manifests(
    original_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, str]],
    source_ids: list[str],
    source_split: dict[str, str],
    source_to_super_group: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    presence = defaultdict(set)
    original_counts = Counter()
    balanced_counts = Counter()
    augmented_counts = Counter()
    for row in original_rows:
        source = row["source_image_id"]
        presence[source].add(int(row["class_id"]))
        original_counts[source] += 1
    for row in balanced_rows:
        source = row["source_image_id"]
        balanced_counts[source] += 1
        augmented_counts[source] += parse_bool(row["is_brightness_augmented"])
    image_rows = []
    for source in source_ids:
        classes = presence[source]
        row = {
            "source_image_id": source,
            "split": source_split[source],
            "super_group_id": source_to_super_group[source],
            "class_count": len(classes),
            "is_multilabel": len(classes) > 1,
            "original_roi_count": original_counts[source],
            "balanced_roi_count": balanced_counts[source],
            "balanced_augmented_count": augmented_counts[source],
        }
        for class_id in CLASS_MAPPING:
            row[f"class_{class_id}_present"] = int(class_id in classes)
        image_rows.append(row)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for balanced in balanced_rows:
        source = balanced["source_image_id"]
        if source_split[source] != "train":
            continue
        split_rows["train"].append(
            {
                "source_image_id": source,
                "split": "train",
                "image_path": balanced["image_path"],
                "relative_path": balanced["relative_path"],
                "filename": balanced["filename"],
                "class_id": int(balanced["class_id"]),
                "class_name": balanced["class_name"],
                "original_roi_id": balanced["original_roi_id"],
                "is_brightness_augmented": parse_bool(
                    balanced["is_brightness_augmented"]
                ),
                "image_sha256": balanced["image_sha256"],
                "source_manifest": "roi_manifest.csv",
                "source_record_index": int(balanced["feature_index"]),
            }
        )
    for original in original_rows:
        source = original["source_image_id"]
        split = source_split[source]
        if split not in {"val", "test"}:
            continue
        split_rows[split].append(
            {
                "source_image_id": source,
                "split": split,
                "image_path": original["image_path"],
                "relative_path": original["relative_path"],
                "filename": original["filename"],
                "class_id": int(original["class_id"]),
                "class_name": original["class_name"],
                "original_roi_id": original["original_roi_id"],
                "is_brightness_augmented": False,
                "image_sha256": original["image_sha256"],
                "source_manifest": "original_roi_manifest.csv",
                "source_record_index": int(original["original_record_index"]),
            }
        )
    for split in split_rows:
        split_rows[split].sort(
            key=lambda row: (int(row["source_record_index"]), row["filename"])
        )
        for record_index, row in enumerate(split_rows[split]):
            row["record_index"] = record_index
    return image_rows, split_rows


def values_crossing_splits(
    split_rows: dict[str, list[dict[str, Any]]], field: str
) -> dict[str, set[str]]:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        for row in rows:
            value = str(row[field])
            locations[value].add(split)
    return {value: splits for value, splits in locations.items() if len(splits) > 1}


def audit_leakage(
    image_rows: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    source_to_super_group: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audits: list[dict[str, Any]] = []

    def add(name: str, values: dict[str, Any] | set[str] | list[str]) -> None:
        count = len(values)
        details = ";".join(sorted(values)[:20]) if values else ""
        audits.append(
            {
                "audit_name": name,
                "leakage_count": count,
                "status": "PASS" if count == 0 else "FAIL",
                "details": details,
            }
        )

    source_locations: dict[str, set[str]] = defaultdict(set)
    super_group_locations: dict[str, set[str]] = defaultdict(set)
    for row in image_rows:
        source_locations[row["source_image_id"]].add(row["split"])
        super_group_locations[row["super_group_id"]].add(row["split"])
    add(
        "source_image_id_cross_split",
        {key: value for key, value in source_locations.items() if len(value) > 1},
    )
    add(
        "super_group_cross_split",
        {key: value for key, value in super_group_locations.items() if len(value) > 1},
    )
    add("original_roi_id_cross_split", values_crossing_splits(split_rows, "original_roi_id"))
    add("image_path_cross_split", values_crossing_splits(split_rows, "image_path"))
    add("image_sha256_cross_split", values_crossing_splits(split_rows, "image_sha256"))

    validation_test_original_ids = {
        row["original_roi_id"]
        for split in ("val", "test")
        for row in split_rows[split]
    }
    augmented_sources = {
        row["original_roi_id"]
        for row in split_rows["train"]
        if parse_bool(row["is_brightness_augmented"])
    }
    add("augmentation_source_cross_split", augmented_sources & validation_test_original_ids)
    counts = {row["audit_name"]: int(row["leakage_count"]) for row in audits}
    if any(counts.values()):
        raise RuntimeError(f"Cross-split leakage detected: {counts}")
    return audits, counts


def build_count_tables(
    image_rows: list[dict[str, Any]], split_rows: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    class_image_rows = []
    roi_count_rows = []
    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        source_rows = [row for row in image_rows if row["split"] == split]
        records = split_rows[split]
        class_image = {
            class_id: sum(int(row[f"class_{class_id}_present"]) for row in source_rows)
            for class_id in CLASS_MAPPING
        }
        class_roi = Counter(int(row["class_id"]) for row in records)
        class_original = Counter(
            int(row["class_id"])
            for row in records
            if not parse_bool(row["is_brightness_augmented"])
        )
        class_augmented = Counter(
            int(row["class_id"])
            for row in records
            if parse_bool(row["is_brightness_augmented"])
        )
        for class_id, class_name in CLASS_MAPPING.items():
            target = TARGET_CLASS_IMAGE_BY_SPLIT[split][class_id]
            class_image_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image_count": class_image[class_id],
                    "target_source_image_count": target,
                    "difference_from_target": class_image[class_id] - target,
                }
            )
            roi_count_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "total_roi_count": class_roi[class_id],
                    "original_roi_count": class_original[class_id],
                    "brightness_augmented_roi_count": class_augmented[class_id],
                }
            )
        roi_count_rows.append(
            {
                "split": split,
                "class_id": "ALL",
                "class_name": "All classes",
                "total_roi_count": len(records),
                "original_roi_count": len(records) - sum(class_augmented.values()),
                "brightness_augmented_roi_count": sum(class_augmented.values()),
            }
        )
        summary[split] = {
            "source_image_count": len(source_rows),
            "class_image_counts": class_image,
            "roi_count": len(records),
            "class_roi_counts": dict(sorted(class_roi.items())),
            "original_roi_count": len(records) - sum(class_augmented.values()),
            "brightness_augmented_roi_count": sum(class_augmented.values()),
            "unilabel_source_count": sum(int(row["class_count"]) == 1 for row in source_rows),
            "multilabel_source_count": sum(int(row["class_count"]) > 1 for row in source_rows),
        }
    return class_image_rows, roi_count_rows, summary


def perform_analysis(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    balanced_manifest_path = Path(args.balanced_manifest).resolve()
    original_roi_dir = Path(args.original_roi_dir).resolve()
    balanced_rows, balanced_stats = validate_balanced_manifest(balanced_manifest_path)
    original_rows, unresolved = scan_original_rois(original_roi_dir)
    if unresolved:
        return {"status": "FAIL_UNRESOLVED", "unresolved": unresolved}
    if len(original_rows) != EXPECTED_ORIGINAL_ROI_COUNT:
        raise ValueError(f"Original ROI rows {len(original_rows)} != 4546")
    original_class_counts = Counter(int(row["class_id"]) for row in original_rows)
    if original_class_counts != Counter(EXPECTED_ORIGINAL_CLASS_ROI_COUNTS):
        raise ValueError(f"Original ROI class counts mismatch: {dict(original_class_counts)}")
    formal = validate_formal_sources(project_root, original_rows)
    source_ids = formal["source_ids"]
    balanced_sources = {row["source_image_id"] for row in balanced_rows}
    if not balanced_sources.issubset(set(source_ids)):
        raise ValueError("Balanced manifest contains source IDs outside the formal 590 sources")
    formal_sources_missing_from_balanced = sorted(set(source_ids) - balanced_sources)

    source_to_super_group, duplicate_rows, duplicate_stats, component_map = (
        build_sha_super_groups(source_ids, original_rows, balanced_rows)
    )
    presence_by_source: dict[str, set[int]] = defaultdict(set)
    roi_by_source_class: dict[str, list[int]] = {
        source_id: [0, 0, 0, 0, 0] for source_id in source_ids
    }
    for row in original_rows:
        source = row["source_image_id"]
        class_id = int(row["class_id"])
        presence_by_source[source].add(class_id)
        roi_by_source_class[source][class_id] += 1
    try:
        source_split, optimization = optimize_split(
            source_ids,
            source_to_super_group,
            component_map,
            presence_by_source,
            roi_by_source_class,
            balanced_sources,
            args.seed,
        )
    except RuntimeError as exc:
        if str(exc).startswith("INFEASIBLE"):
            return {
                "status": "FAIL_INFEASIBLE",
                "error_reason": str(exc),
                "duplicate_stats": duplicate_stats,
                "component_sizes": sorted(
                    (len(values) for values in component_map.values()), reverse=True
                ),
            }
        raise
    image_rows, split_rows = build_manifests(
        original_rows,
        balanced_rows,
        source_ids,
        source_split,
        source_to_super_group,
    )
    leakage_rows, leakage_counts = audit_leakage(
        image_rows, split_rows, source_to_super_group
    )
    class_image_rows, roi_count_rows, split_summary = build_count_tables(
        image_rows, split_rows
    )
    if any(
        split_summary[split]["source_image_count"] != EXPECTED_SPLIT_SOURCE_COUNTS[split]
        for split in EXPECTED_SPLIT_SOURCE_COUNTS
    ):
        raise RuntimeError("Final source counts are not exactly 472/59/59")
    if split_summary["val"]["brightness_augmented_roi_count"] != 0:
        raise RuntimeError("Validation contains augmented ROI")
    if split_summary["test"]["brightness_augmented_roi_count"] != 0:
        raise RuntimeError("Test contains augmented ROI")
    return {
        "status": "PASS",
        "project_root": project_root,
        "balanced_manifest_path": balanced_manifest_path,
        "original_roi_dir": original_roi_dir,
        "balanced_rows": balanced_rows,
        "balanced_stats": balanced_stats,
        "balanced_unique_source_count": len(balanced_sources),
        "formal_sources_missing_from_balanced": formal_sources_missing_from_balanced,
        "original_rows": original_rows,
        "original_class_counts": dict(sorted(original_class_counts.items())),
        "formal": formal,
        "source_to_super_group": source_to_super_group,
        "component_map": component_map,
        "duplicate_rows": duplicate_rows,
        "duplicate_stats": duplicate_stats,
        "source_split": source_split,
        "optimization": optimization,
        "image_rows": image_rows,
        "split_rows": split_rows,
        "leakage_rows": leakage_rows,
        "leakage_counts": leakage_counts,
        "class_image_rows": class_image_rows,
        "roi_count_rows": roi_count_rows,
        "split_summary": split_summary,
    }


def dry_run_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis["status"] != "PASS":
        return analysis
    return {
        "status": "PASS",
        "write_executed": False,
        "balanced_manifest_sha256": analysis["balanced_stats"]["sha256"],
        "original_roi_rows": len(analysis["original_rows"]),
        "balanced_roi_rows": len(analysis["balanced_rows"]),
        "balanced_unique_source_count": analysis["balanced_unique_source_count"],
        "formal_sources_missing_from_balanced": analysis[
            "formal_sources_missing_from_balanced"
        ],
        "unique_source_image_id": len(analysis["formal"]["source_ids"]),
        "formal_class_image_counts": analysis["formal"]["class_image_counts"],
        "formal_original_class_roi_counts": analysis["original_class_counts"],
        "duplicate_stats": analysis["duplicate_stats"],
        "optimization": analysis["optimization"],
        "split_summary": analysis["split_summary"],
        "leakage_counts": analysis["leakage_counts"],
    }


def write_outputs(analysis: dict[str, Any], output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    original_path = output_dir / "original_roi_manifest.csv"
    image_split_path = output_dir / "image_id_split_manifest.csv"
    train_path = output_dir / "train_roi_manifest.csv"
    val_path = output_dir / "val_roi_manifest.csv"
    test_path = output_dir / "test_roi_manifest.csv"
    class_counts_path = output_dir / "split_class_image_counts.csv"
    roi_counts_path = output_dir / "split_roi_counts.csv"
    duplicate_path = output_dir / "duplicate_sha256_super_groups.csv"
    leakage_path = output_dir / "cross_split_leakage_audit.csv"

    atomic_write_csv(
        original_path,
        ORIGINAL_FIELDS,
        analysis["original_rows"],
        expected_rows=EXPECTED_ORIGINAL_ROI_COUNT,
    )
    atomic_write_csv(
        image_split_path,
        IMAGE_SPLIT_FIELDS,
        analysis["image_rows"],
        expected_rows=EXPECTED_SOURCE_COUNT,
    )
    atomic_write_csv(
        train_path,
        ROI_SPLIT_FIELDS,
        analysis["split_rows"]["train"],
    )
    atomic_write_csv(val_path, ROI_SPLIT_FIELDS, analysis["split_rows"]["val"])
    atomic_write_csv(test_path, ROI_SPLIT_FIELDS, analysis["split_rows"]["test"])
    atomic_write_csv(
        class_counts_path, CLASS_COUNT_FIELDS, analysis["class_image_rows"], expected_rows=15
    )
    atomic_write_csv(roi_counts_path, ROI_COUNT_FIELDS, analysis["roi_count_rows"], expected_rows=18)
    atomic_write_csv(duplicate_path, DUPLICATE_FIELDS, analysis["duplicate_rows"])
    atomic_write_csv(
        leakage_path,
        LEAKAGE_FIELDS,
        analysis["leakage_rows"],
        expected_rows=len(analysis["leakage_rows"]),
    )

    manifest_paths = {
        "image_id_split_manifest": image_split_path,
        "original_roi_manifest": original_path,
        "train_roi_manifest": train_path,
        "val_roi_manifest": val_path,
        "test_roi_manifest": test_path,
    }
    manifest_hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    phase1_backbone = (
        analysis["project_root"]
        / "outputs"
        / "raddino_convnext_tiny_experiment_seed42"
        / "phase1_distillation"
        / "checkpoints"
        / "distilled_convnext_tiny_backbone.pt"
    )
    protocol = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "split_seed": seed,
        "split_strategy": analysis["optimization"]["strategy"],
        "grouping_unit": "source_image_id with SHA256-connected super-groups",
        "source_image_counts": {
            split: analysis["split_summary"][split]["source_image_count"]
            for split in ("train", "val", "test")
        },
        "class_mapping": CLASS_MAPPING,
        "balanced_roi_manifest_path": str(analysis["balanced_manifest_path"]),
        "balanced_roi_manifest_sha256": analysis["balanced_stats"]["sha256"],
        "manifest_sha256": manifest_hashes,
        "proposed_and_baseline_share_identical_split": True,
        "proposed_phase2_initialization": {
            "architecture": "convnext_tiny",
            "checkpoint": str(phase1_backbone),
            "checkpoint_sha256": sha256_file(phase1_backbone),
            "training_executed": False,
        },
        "baseline_initialization": {
            "architecture": "convnext_tiny",
            "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "phase1_distillation_used": False,
            "training_executed": False,
        },
        "phase2_training_executed": False,
        "test_evaluation_executed": False,
    }
    protocol_path = output_dir / "shared_training_protocol.json"
    atomic_write_json(protocol_path, protocol)

    summary = {
        "status": "PASS",
        "seed": seed,
        "source_image_count": EXPECTED_SOURCE_COUNT,
        "original_roi_count": len(analysis["original_rows"]),
        "balanced_roi_count": len(analysis["balanced_rows"]),
        "balanced_original_roi_count": analysis["balanced_stats"]["original_count"],
        "balanced_augmented_roi_count": analysis["balanced_stats"]["augmented_count"],
        "split_summary": analysis["split_summary"],
        "duplicate_stats": analysis["duplicate_stats"],
        "optimization": analysis["optimization"],
        "leakage_counts": analysis["leakage_counts"],
        "manifest_sha256": manifest_hashes,
    }
    atomic_write_json(output_dir / "split_summary.json", summary)

    report_lines = [
        "Phase 2 Shared Grouped Split Report",
        "===================================",
        "status: PASS",
        f"seed: {seed}",
        f"balanced_manifest_sha256: {analysis['balanced_stats']['sha256']}",
        f"unique_source_image_id: {EXPECTED_SOURCE_COUNT}",
        f"original_roi_rows: {len(analysis['original_rows'])}",
        f"balanced_roi_rows: {len(analysis['balanced_rows'])}",
        f"cross_source_duplicate_sha256_groups: {analysis['duplicate_stats']['cross_source_duplicate_sha256_group_count']}",
        f"super_groups: {analysis['duplicate_stats']['super_group_count']}",
        "",
    ]
    for split in ("train", "val", "test"):
        item = analysis["split_summary"][split]
        report_lines.extend(
            [
                f"[{split}]",
                f"source_images: {item['source_image_count']}",
                f"class_image_counts: {item['class_image_counts']}",
                f"roi_count: {item['roi_count']}",
                f"class_roi_counts: {item['class_roi_counts']}",
                f"original_roi_count: {item['original_roi_count']}",
                f"brightness_augmented_roi_count: {item['brightness_augmented_roi_count']}",
                "",
            ]
        )
    report_lines.append(f"leakage_counts: {analysis['leakage_counts']}")
    report_lines.append("Proposed and Baseline share identical split: True")
    atomic_write_text(output_dir / "split_report.txt", "\n".join(report_lines) + "\n")

    metadata = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "seed": seed,
        "split_strategy": analysis["optimization"]["strategy"],
        "source_image_counts": {
            split: analysis["split_summary"][split]["source_image_count"]
            for split in ("train", "val", "test")
        },
        "class_mapping": CLASS_MAPPING,
        "manifest_sha256": manifest_hashes,
        "source_dataset_paths": {
            "balanced_roi_manifest": str(analysis["balanced_manifest_path"]),
            "original_roi_dir": str(analysis["original_roi_dir"]),
            "formal_images": analysis["formal"]["full_image_dir"],
            "formal_annotations": analysis["formal"]["annotations_path"],
            "formal_image_manifest": analysis["formal"]["image_manifest_path"],
        },
        "duplicate_sha_grouping_strategy": (
            "Union source_image_id values connected by identical image SHA256 across "
            "original and balanced ROI records; each connected component is indivisible"
        ),
        "duplicate_stats": analysis["duplicate_stats"],
        "leakage_audit_result": {
            "status": "PASS",
            "counts": analysis["leakage_counts"],
        },
        "models_loaded": False,
        "training_executed": False,
        "source_images_copied": False,
    }
    atomic_write_json(output_dir / "split_metadata.json", metadata)

    required = {
        "image_id_split_manifest.csv": EXPECTED_SOURCE_COUNT,
        "original_roi_manifest.csv": EXPECTED_ORIGINAL_ROI_COUNT,
        "train_roi_manifest.csv": len(analysis["split_rows"]["train"]),
        "val_roi_manifest.csv": len(analysis["split_rows"]["val"]),
        "test_roi_manifest.csv": len(analysis["split_rows"]["test"]),
        "split_class_image_counts.csv": 15,
        "split_roi_counts.csv": 18,
        "duplicate_sha256_super_groups.csv": len(analysis["duplicate_rows"]),
        "cross_split_leakage_audit.csv": len(analysis["leakage_rows"]),
    }
    for filename, expected_rows in required.items():
        path = output_dir / filename
        if not path.is_file() or len(read_csv(path)) != expected_rows:
            raise RuntimeError(f"Final output verification failed: {filename}")
    for filename in (
        "shared_training_protocol.json",
        "split_summary.json",
        "split_metadata.json",
    ):
        value = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        if value.get("status") != "PASS":
            raise RuntimeError(f"Final JSON status failed: {filename}")
    if not (output_dir / "split_report.txt").is_file():
        raise RuntimeError("Final split report is missing")
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "manifest_paths": {name: str(path) for name, path in manifest_paths.items()},
        "manifest_sha256": manifest_hashes,
        "protocol_path": str(protocol_path),
        "split_summary": analysis["split_summary"],
        "duplicate_stats": analysis["duplicate_stats"],
        "leakage_counts": analysis["leakage_counts"],
    }


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = [str(path) for path in sorted(output_dir.rglob("*"))]
        raise FileExistsError(
            "Phase 2 split output already exists and is non-empty; refusing to overwrite:\n"
            + "\n".join(existing)
        )
    analysis = perform_analysis(args)
    if analysis["status"] == "FAIL_UNRESOLVED":
        if args.dry_run:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
            return 2
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_csv(
            output_dir / "unresolved_original_roi.csv",
            ["image_path", "filename", "error_reason"],
            analysis["unresolved"],
        )
        print(json.dumps({"status": "FAIL", "reason": "unresolved_original_roi"}, indent=2))
        return 2
    if analysis["status"] == "FAIL_INFEASIBLE":
        if args.dry_run:
            print(json.dumps(analysis, ensure_ascii=False, indent=2))
            return 2
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_text(
            output_dir / "infeasible_split_report.txt",
            json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps(dry_run_payload(analysis), ensure_ascii=False, indent=2))
        return 0
    result = write_outputs(analysis, output_dir, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument(
        "--balanced-manifest",
        default=str(
            project_root
            / "outputs"
            / "raddino_feature_cache"
            / "balanced_945_seed42"
            / "roi_manifest.csv"
        ),
    )
    parser.add_argument(
        "--original-roi-dir",
        default=str(project_root / "data" / "processed" / "bbox_crops_224"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            project_root
            / "outputs"
            / "raddino_convnext_tiny_experiment_seed42"
            / "phase2_split"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
