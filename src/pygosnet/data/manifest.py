"""Labelbox export parsing and dataset audit helpers."""
from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path


def load_labelbox_exports(annotation_dir: Path):
    grouped = defaultdict(list)
    for path in sorted(annotation_dir.glob("*.json")):
        source = path.name.split("_export", 1)[0].upper()
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            external_id = record.get("External ID")
            if not external_id:
                continue
            label = record.get("Label") or {}
            objects = label.get("objects") or []
            points = []
            for obj in objects:
                bbox = obj.get("bbox") or {}
                if not all(k in bbox for k in ("left", "top", "width", "height")):
                    continue
                # Labelbox exports tiny penguins as bounding boxes. Convert each
                # box to its center point; this is point supervision, not box training.
                x = float(bbox["left"]) + float(bbox["width"]) / 2.0
                y = float(bbox["top"]) + float(bbox["height"]) / 2.0
                points.append([x, y])
            grouped[external_id].append({
                "source": source,
                "record_id": record.get("ID"),
                "points": points,
                "record": record,
            })
    return grouped


def points_equal(a, b):
    return sorted((round(x, 4), round(y, 4)) for x, y in a) == sorted(
        (round(x, 4), round(y, 4)) for x, y in b
    )


def audit_dataset(image_paths, annotation_dir: Path):
    grouped = load_labelbox_exports(annotation_dir)
    image_by_name = {p.name: p for p in image_paths}
    duplicate_rows, orphan_rows, missing_rows = [], [], []
    unique_annotation_count = 0
    source_summary = defaultdict(lambda: {"records": 0, "objects": 0})
    source_summary = defaultdict(lambda: {"records": 0, "objects": 0})
    for external_id, records in sorted(grouped.items()):
        for rec in records:
            source_summary[rec["source"]]["records"] += 1
            source_summary[rec["source"]]["objects"] += len(rec["points"])
        if external_id not in image_by_name:
            orphan_rows.append({"external_id": external_id, "sources": [r["source"] for r in records]})
            continue
        if len(records) > 1:
            duplicate_rows.append({
                "external_id": external_id,
                "sources": [r["source"] for r in records],
                "counts": [len(r["points"]) for r in records],
                "annotations_identical": all(points_equal(records[0]["points"], r["points"]) for r in records[1:]),
            })
        unique_annotation_count += len(records[0]["points"])
    for name in sorted(set(image_by_name) - set(grouped)):
        missing_rows.append(name)
    return {
        "image_count": len(image_paths),
        "json_unique_external_ids": len(grouped),
        "matched_image_annotation_ids": len(set(image_by_name) & set(grouped)),
        "annotation_records": sum(len(v) for v in grouped.values()),
        "annotation_objects_by_export": dict(source_summary),
        "unique_image_object_count_if_duplicate_source_selected": {
            source: sum(len(next((r for r in records if r["source"] == source), records[0])["points"]) for external_id, records in grouped.items() if external_id in image_by_name)
            for source in ("MAISIE", "LUKE")
        },
        "proposal_reported_penguin_total": 135279,
        "difference_from_proposal_total_by_duplicate_choice": {
            source: sum(len(next((r for r in records if r["source"] == source), records[0])["points"]) for external_id, records in grouped.items() if external_id in image_by_name) - 135279
            for source in ("MAISIE", "LUKE")
        },
        "duplicate_external_ids": duplicate_rows,
        "orphan_annotation_ids": orphan_rows,
        "images_without_annotation": missing_rows,
        "note": "Duplicate annotations are not silently merged. Resolve them explicitly before creating the training manifest.",
    }


def build_rows(image_paths, annotation_dir: Path, duplicate_source: str | None):
    grouped = load_labelbox_exports(annotation_dir)
    image_by_name = {p.name: p for p in image_paths}
    rows, unresolved = [], []
    for external_id, records in sorted(grouped.items()):
        image_path = image_by_name.get(external_id)
        if image_path is None:
            continue  # orphan JSON annotation; recorded in audit report
        if len(records) == 1:
            chosen = records[0]
        else:
            if duplicate_source is None:
                unresolved.append(external_id)
                continue
            matches = [r for r in records if r["source"].upper() == duplicate_source.upper()]
            if len(matches) != 1:
                unresolved.append(external_id)
                continue
            chosen = matches[0]
        rows.append({
            "image_id": external_id,
            "image_path": image_path.as_posix(),
            "annotation_source": chosen["source"],
            "points": chosen["points"],
            "count": len(chosen["points"]),
        })
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} duplicate image annotations need a source decision. "
            f"Re-run with --duplicate-source MAISIE or --duplicate-source LUKE. "
            f"Examples: {unresolved[:5]}"
        )
    used = {r["image_id"] for r in rows}
    missing = sorted(set(image_by_name) - used)
    if missing:
        raise ValueError(f"{len(missing)} images have no selected annotation. Examples: {missing[:5]}")
    return rows


def write_conflict_csv(audit: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["external_id", "sources", "counts", "annotations_identical"])
        writer.writeheader()
        for row in audit["duplicate_external_ids"]:
            writer.writerow({**row, "sources": ";".join(row["sources"]), "counts": ";".join(map(str, row["counts"]))})
