"""Cross-owner contracts using the actual Dataset, model, loss and Trainer."""
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from pygosnet.data.dataset import PenguinDataset
from pygosnet.data.transforms import TrainTransform
from pygosnet.engine import Trainer
from pygosnet.eval.labelbox import load_manifest_index
from pygosnet.losses import CombinedLoss
from pygosnet.models import PygosNet


def test_all_ablation_configs_construct_loss():
    for path in (Path(__file__).parents[1] / 'configs/ablation').glob('*.yaml'):
        cfg = yaml.safe_load(path.read_text())
        loss = CombinedLoss(**cfg['loss']['kwargs'])
        assert loss.use_hard_negative == cfg['loss']['kwargs']['use_hard_negative']


def test_background_toggle_is_independent():
    d = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
    batch = {'density': torch.zeros_like(d), 'structural_mask': torch.zeros_like(d),
             'seg_mask': torch.zeros_like(d)}
    output = {'density': d, 'segmentation_logits': torch.zeros_like(d, requires_grad=True)}
    on = CombinedLoss(num_scales=1, use_hard_negative=True)(output, batch)
    off = CombinedLoss(num_scales=1, use_hard_negative=False)(output, batch)
    assert off['background'].item() == 0
    torch.testing.assert_close(on['total'] - off['total'], .05 * on['background'])


def test_horizontal_flip_keeps_edge_point_mass():
    transform = TrainTransform(32, horizontal_flip_p=1.)
    _, points = transform(Image.new('RGB', (32, 32)), [(0., 16.), (31.9, 2.)])
    assert len(points) == 2
    assert all(0 <= x < 32 and 0 <= y < 32 for x, y in points)


@pytest.mark.parametrize('use_dcb', [False, True])
@pytest.mark.parametrize('use_hard_negative', [False, True])
def test_actual_abcs_train_validate_resume(tmp_path, use_dcb, use_hard_negative):
    torch.manual_seed(42)
    random.seed(42)
    image = tmp_path / 'Site' / 'tile.png'
    image.parent.mkdir()
    Image.new('RGB', (32, 32), (120, 130, 140)).save(image)
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(json.dumps({'image_id': 'tile.png', 'image_path': 'Site/tile.png',
                                    'points': [[16., 16.]], 'count': 1}) + '\n')
    dataset = PenguinDataset(manifest, tmp_path, crop_size=32)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch['structural_mask'].shape == (1, 1, 4, 4)
    torch.testing.assert_close(batch['structural_mask'], batch['seg_mask'])
    assert batch['density'].sum().item() == pytest.approx(1., abs=1e-5)
    assert batch['sample_key'] == ['Site/tile.png']
    index = load_manifest_index(manifest)
    assert index['Site/tile.png']['count'] == 1
    np.testing.assert_equal(index['Site/tile.png']['points_yx'], [[16., 16.]])
    model = PygosNet(use_dcb=use_dcb, pretrained=False)
    criterion = CombinedLoss(use_hard_negative=use_hard_negative)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, criterion, optimizer, 'cpu', tmp_path / 'run', selection_start_epoch=1)
    loader = DataLoader(dataset, batch_size=1)
    score = trainer.fit(loader, loader, epochs=1)
    assert np.isfinite(score)
    assert (tmp_path / 'run/best.pt').is_file()
    assert trainer.resume(tmp_path / 'run/last.pt')['epoch'] == 1
    assert trainer.start_epoch == 2
