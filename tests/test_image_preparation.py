import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from pygosnet.data.dataset import PenguinDataset
from pygosnet.data.image_preparation import prepare_images


def fixture(root, size=(448, 448), member='Site/tile.png'):
    raw = root / 'data/raw'
    raw.mkdir(parents=True)
    row = {'image_id': 'tile.png', 'image_path': 'data/raw/images/Site/tile.png',
           'points': [[200., 200.]], 'count': 1, 'annotation_source': 'Site'}
    for name, rows in [('processed/manifest.jsonl', [row]), ('splits/train.jsonl', [row]),
                       ('splits/val.jsonl', []), ('splits/test.jsonl', [])]:
        path = root / 'data' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (root / 'data/splits/split_metadata.json').write_text(json.dumps({
        'image_count': 1, 'split_counts': {'train': 1, 'val': 0, 'test': 0},
        'duplicate_annotation_source': 'LUKE', 'is_original_author_image_assignment': False}))
    data = io.BytesIO()
    Image.new('RGB', size, (20, 30, 40)).save(data, format='PNG')
    with zipfile.ZipFile(raw / 'Site.zip', 'w') as z:
        z.writestr(member, data.getvalue())
        z.writestr('__MACOSX/Site/._tile.png', b'not an image')
    return raw / 'Site.zip', data.getvalue()


def test_prepare_then_dataset_and_idempotent_verify(tmp_path):
    fixture(tmp_path)
    manifests = list((tmp_path / 'data').rglob('*.jsonl'))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}
    report = prepare_images(tmp_path)
    assert report['extracted'] == 1
    assert report['ignored_zip_metadata_entries'] == 1
    assert report['manifests_modified'] is False
    sample = PenguinDataset('data/splits/train.jsonl', tmp_path, split='val')[0]
    assert sample['image'].shape == (3, 448, 448)
    assert sample['density'].sum().item() == pytest.approx(1., abs=1e-5)
    assert prepare_images(tmp_path)['existing'] == 1
    assert prepare_images(tmp_path, verify_only=True)['extracted'] == 0
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}


def test_verify_reports_missing_files_without_extracting(tmp_path):
    fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match='missing images'):
        prepare_images(tmp_path, verify_only=True)
    assert not (tmp_path / 'data/raw/images').exists()


def test_split_overlap_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'data/splits/val.jsonl').write_bytes((tmp_path / 'data/splits/train.jsonl').read_bytes())
    path = tmp_path / 'data/splits/split_metadata.json'
    metadata = json.loads(path.read_text())
    metadata['split_counts']['val'] = 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Split overlap'):
        prepare_images(tmp_path)


def test_count_mismatch_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'data/processed/manifest.jsonl'
    row = json.loads(path.read_text())
    row['count'] = 2
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Count disagrees'):
        prepare_images(tmp_path)


def test_archive_missing_image_rejected(tmp_path):
    fixture(tmp_path, member='Site/other.png')
    with pytest.raises(FileNotFoundError, match='lacks 1 manifest images'):
        prepare_images(tmp_path)


def test_wrong_image_size_rejected(tmp_path):
    fixture(tmp_path, size=(32, 32))
    with pytest.raises(ValueError, match='448x448'):
        prepare_images(tmp_path)
    assert not (tmp_path / 'data/raw/images/Site/tile.png').exists()
    assert not list(tmp_path.rglob('*.tmp'))


def test_unsafe_manifest_path_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'data/processed/manifest.jsonl'
    row = json.loads(path.read_text())
    row['image_path'] = '../tile.png'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Unsafe image path'):
        prepare_images(tmp_path)


def test_ambiguous_zip_names_rejected(tmp_path):
    archive, content = fixture(tmp_path)
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('another/tile.png', content)
    with pytest.raises(ValueError, match='Ambiguous duplicate filename'):
        prepare_images(tmp_path)


def test_lfs_placeholder_rejected(tmp_path):
    archive, _ = fixture(tmp_path)
    archive.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='Not a ZIP'):
        prepare_images(tmp_path)
