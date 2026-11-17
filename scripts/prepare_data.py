#!/usr/bin/env python
"""Extract BAS archives, audit annotations, and create reproducible manifests.

First run without --duplicate-source to generate an audit report and stop if
conflicting duplicate annotations are present. After the team chooses a source
for overlapping exports, rerun with --duplicate-source MAISIE or LUKE.
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

from pygosnet.data.manifest import audit_dataset, build_rows, write_conflict_csv


def extract_archive(archive_path: Path, project_root: Path):
    raw_root = project_root / "data" / "raw"
    image_root = raw_root / "images"
    annotation_root = raw_root / "annotations"
    image_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as outer:
        for name in outer.namelist():
            if name.lower().endswith(".json") or name.lower() == "readme.md":
                target = annotation_root / Path(name).name
                target.write_bytes(outer.read(name))
            elif name.lower().endswith(".zip"):
                person = Path(name).stem
                target_dir = image_root / person
                target_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(outer.open(name)) as inner:
                    for member in inner.namelist():
                        # Ignore macOS metadata and non-PNG files.
                        if member.startswith("__MACOSX/") or "/__MACOSX/" in member:
                            continue
                        if not member.lower().endswith(".png"):
                            continue
                        out = target_dir / Path(member).name
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(inner.read(member))
    return image_root, annotation_root


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True, help="Original Dryad ZIP")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-source", choices=["MAISIE", "LUKE"], default=None,
                        help="Explicit source to use for the 13 overlapping image annotations")
    args = parser.parse_args()
    root = args.project_root.resolve()
    image_root, annotation_root = extract_archive(args.archive.resolve(), root)
    image_paths = sorted(p for p in image_root.rglob("*.png") if p.is_file())
    audit = audit_dataset(image_paths, annotation_root)
    audit_path = root / "data" / "processed" / "dataset_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_conflict_csv(audit, root / "data" / "processed" / "annotation_conflicts.csv")
    print(f"Audit report: {audit_path}")
    print(json.dumps({k: v for k, v in audit.items() if k != "duplicate_external_ids"}, ensure_ascii=False, indent=2))
    print(f"Conflicting/duplicate image IDs: {len(audit['duplicate_external_ids'])}")

    # Create the reproducible image split independently of annotation adjudication.
    # This means the split is available even while the team reviews the 13 overlaps.
    relative_images = [p.relative_to(root) for p in image_paths]
    rng = random.Random(args.seed)
    shuffled_images = list(relative_images)
    rng.shuffle(shuffled_images)
    train_images, val_images, test_images = shuffled_images[:446], shuffled_images[446:592], shuffled_images[592:]
    split_dir = root / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train_images), ("val", val_images), ("test", test_images)):
        (split_dir / f"{name}.txt").write_text("\n".join(p.name for p in subset) + "\n", encoding="utf-8")
    metadata = {
        "dataset": "BAS aerial penguin tiles from Dryad DOI 10.5061/dryad.8931zcrv8",
        "image_count": len(relative_images),
        "split_counts": {"train": len(train_images), "val": len(val_images), "test": len(test_images)},
        "seed": args.seed,
        "method": "random shuffle with fixed seed and fixed 446/146/146 sizes",
        "is_original_author_image_assignment": False,
        "duplicate_annotation_source": args.duplicate_source,
        "warning": "Reconstructed split, not a recovered official per-image split. The source paper reports the 3:1:1 split sizes but not the per-image assignment in the supplied files.",
    }
    (split_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Created reproducible image lists: data/splits/train.txt, val.txt, test.txt")

    # Full image/point manifests require a transparent choice for the duplicate exports.
    if args.duplicate_source is None and audit["duplicate_external_ids"]:
        print("\nAUDIT COMPLETE. Image split lists were created, but annotation manifests were not.")
        print("Review data/processed/annotation_conflicts.csv, agree on an annotator source, then rerun with --duplicate-source <your choice>.")
        return
    rows = build_rows(relative_images, annotation_root, args.duplicate_source)
    if len(rows) != 738:
        raise RuntimeError(f"Expected 738 matched images after annotation selection, got {len(rows)}")
    by_id = {row["image_id"]: row for row in rows}
    train_rows = [by_id[p.name] for p in train_images]
    val_rows = [by_id[p.name] for p in val_images]
    test_rows = [by_id[p.name] for p in test_images]
    write_jsonl(root / "data" / "processed" / "manifest.jsonl", rows)
    for name, subset in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        write_jsonl(split_dir / f"{name}.jsonl", subset)
    metadata["duplicate_annotation_source"] = args.duplicate_source
    (split_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Created annotation manifests for 446/146/146 images.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
