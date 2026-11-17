"""Arrange existing ZIP images to match the committed, adjudicated manifests.

This module never generates a split or chooses an annotation source.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def checked_image_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe image path: {relative}')
    target = (root / rel).resolve()
    image_root = (root / 'data/raw/images').resolve()
    if not target.is_relative_to(image_root) or len(target.relative_to(image_root).parts) != 2:
        raise ValueError(f'Expected data/raw/images/<site>/<file>: {relative}')
    if target.suffix.lower() != '.png':
        raise ValueError(f'Expected PNG: {relative}')
    return target


def validate_manifests(root):
    """Validate existing full/split manifests; never rewrite their contents."""
    root = Path(root).resolve()
    rows = read_jsonl(root / 'data/processed/manifest.jsonl')
    metadata = json.loads((root / 'data/splits/split_metadata.json').read_text())
    if len(rows) != metadata['image_count']:
        raise ValueError('Full manifest size differs from split_metadata.json.')
    by_id = {}
    paths = set()
    for row in rows:
        image_id = row['image_id']
        path = checked_image_path(root, row['image_path'])
        if image_id != path.name:
            raise ValueError(f'image_id differs from image filename: {image_id}')
        if image_id in by_id or path in paths:
            raise ValueError(f'Duplicate image id or path: {image_id}')
        if row['count'] != len(row['points']):
            raise ValueError(f'Count disagrees with points: {image_id}')
        if any(len(point) != 2 or not all(0 <= float(v) < 448 for v in point)
               for point in row['points']):
            raise ValueError(f'Invalid point coordinates: {image_id}')
        paths.add(path)
        by_id[image_id] = row
    used = set()
    split_counts = {}
    for split in ('train', 'val', 'test'):
        subset = read_jsonl(root / f'data/splits/{split}.jsonl')
        if len(subset) != metadata['split_counts'][split]:
            raise ValueError(f'{split} size differs from split_metadata.json.')
        for row in subset:
            image_id = row['image_id']
            if image_id in used:
                raise ValueError(f'Split overlap or duplicate: {image_id}')
            if row != by_id.get(image_id):
                raise ValueError(f'Split row differs from full manifest: {image_id}')
            used.add(image_id)
        split_counts[split] = len(subset)
    if used != set(by_id):
        raise ValueError('Split manifests do not cover the full manifest.')
    return rows, metadata, split_counts


def _check_png(path):
    try:
        with Image.open(path) as image:
            if image.format != 'PNG' or image.size != (448, 448):
                raise ValueError(f'Expected 448x448 PNG, got {image.format} {image.size}')
            image.verify()
    except Exception as error:
        raise ValueError(f'Invalid image {path}: {error}') from error


def prepare_images(project_root, verify_only=False):
    root = Path(project_root).resolve()
    rows, metadata, split_counts = validate_manifests(root)
    by_site = defaultdict(list)
    for row in rows:
        path = checked_image_path(root, row['image_path'])
        by_site[path.parent.name].append(path)
    extracted = existing = ignored_metadata = 0
    # Resolve every archive member first; fail before extraction on missing entries.
    plans = []
    for site, targets in sorted(by_site.items()):
        missing = []
        for target in targets:
            if target.exists():
                _check_png(target)
                existing += 1
            else:
                missing.append(target)
        if not missing:
            continue
        if verify_only:
            raise FileNotFoundError(f'{len(missing)} missing images in {site}, e.g. {missing[0]}')
        archive = root / f'data/raw/{site}.zip'
        if not archive.is_file():
            raise FileNotFoundError(f'Missing archive: {archive}')
        if not zipfile.is_zipfile(archive):
            raise ValueError(f'Not a ZIP (check whether Git LFS content was downloaded): {archive}')
        members = {}
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                parts = Path(member.filename).parts
                if '__MACOSX' in parts or any(p.startswith('._') for p in parts):
                    ignored_metadata += 1
                    continue
                if member.is_dir() or Path(member.filename).suffix.lower() != '.png':
                    continue
                name = Path(member.filename).name
                if name in members:
                    raise ValueError(f'Ambiguous duplicate filename in {archive}: {name}')
                members[name] = member.filename
        absent = [target.name for target in missing if target.name not in members]
        if absent:
            raise FileNotFoundError(f'{archive} lacks {len(absent)} manifest images, e.g. {absent[0]}')
        plans.append((archive, [(target, members[target.name]) for target in missing]))
    for archive, targets in plans:
        with zipfile.ZipFile(archive) as source:
            for target, member in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.png.tmp')
                try:
                    with source.open(member) as src, temporary.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    _check_png(temporary)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                extracted += 1
    # Re-read every actual PNG after extraction, including existing files.
    for row in rows:
        _check_png(checked_image_path(root, row['image_path']))
    return {'status': 'passed', 'image_count': len(rows), 'extracted': extracted,
            'existing': existing, 'ignored_zip_metadata_entries': ignored_metadata,
            'split_counts': split_counts,
            'images_by_site': dict(Counter(Path(r['image_path']).parent.name for r in rows)),
            'annotation_count': sum(r['count'] for r in rows),
            'duplicate_annotation_source': metadata.get('duplicate_annotation_source'),
            'manifests_modified': False,
            'is_original_author_image_assignment': metadata.get('is_original_author_image_assignment')}
