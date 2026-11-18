"""Labelbox export parsing for evaluation GT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PENGUIN_VALUE = "penguin"
KNOWN_SITES = ("Jack", "Luke", "Maisie", "Thomas")


def _site_from_path(path: str | Path) -> str | None:
    # JACK_export-...json -> Jack
    stem = Path(path).name.split("_")[0]
    for site in KNOWN_SITES:
        if stem.lower() == site.lower():
            return site
    return None


def parse_record(record: dict, site: str | None = None, value: str = PENGUIN_VALUE) -> dict:
    label = record.get("Label")
    objects = label.get("objects", []) if isinstance(label, dict) else []

    centers_yx: list[list[float]] = []
    boxes_xyxy: list[list[float]] = []
    for obj in objects:
        if obj.get("value") != value:
            continue
        bbox = obj["bbox"]
        left = float(bbox["left"])
        top = float(bbox["top"])
        width = float(bbox["width"])
        height = float(bbox["height"])
        centers_yx.append([top + height / 2.0, left + width / 2.0])
        boxes_xyxy.append([left, top, left + width, top + height])

    name = record.get("External ID")
    key = f"{site}/{name}" if site else str(name)
    return {
        "site": site,
        "name": name,
        "key": key,
        "count": len(centers_yx),
        "points_yx": np.asarray(centers_yx, dtype=np.float64).reshape(-1, 2),
        "boxes_xyxy": np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4),
        "skipped": bool(record.get("Skipped", False)),
    }


def build_index(json_paths, value: str = PENGUIN_VALUE) -> dict[str, dict]:
    # {site/name.png: sample}
    index: dict[str, dict] = {}
    for path in json_paths:
        site = _site_from_path(path)
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        for record in records:
            sample = parse_record(record, site=site, value=value)
            index[sample["key"]] = sample
    return index


def load_index(pattern: str = "*_export-*.json", root: str | Path = ".") -> dict[str, dict]:
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No Labelbox export matching {pattern!r} under {Path(root).resolve()}")
    return build_index(paths)


def load_split(path: str | Path) -> list[str]:
    # normalize separators to /
    names = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(name).replace("\\", "/") for name in names]


def split_counts(index: dict[str, dict], keys) -> np.ndarray:
    # fails on missing samples
    missing = [key for key in keys if key not in index]
    if missing:
        preview = ", ".join(missing[:3])
        raise KeyError(f"{len(missing)} split samples missing from annotations, e.g.: {preview}")
    return np.asarray([index[key]["count"] for key in keys], dtype=np.float64)


def load_manifest_index(path):
    """Use the adjudicated A manifests as the single GT source for evaluation."""
    index = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(row["image_path"])
        key = f"{image.parent.name}/{image.name}"
        if key in index:
            raise ValueError(f"Duplicate manifest sample key: {key}")
        points = np.asarray(row["points"], dtype=np.float64).reshape(-1, 2)
        if int(row["count"]) != len(points):
            raise ValueError(f"Manifest count disagrees with points: {key}")
        index[key] = {"site": image.parent.name, "name": image.name, "key": key,
                      "image_path": row["image_path"], "count": len(points),
                      "points_yx": points[:, ::-1].copy()}
    return index
