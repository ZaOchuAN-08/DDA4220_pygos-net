import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from pygosnet.engine.device import select_device
from pygosnet.engine import pipeline


@pytest.mark.parametrize('cuda,mps,expected', [(True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    assert select_device('auto').type == expected


def test_explicit_unavailable_device_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        select_device('cuda')


def setup_project(tmp_path):
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / 'scripts/prepare_images.py').write_text('')
    (root / 'data/splits').mkdir(parents=True)
    for name in ('val', 'test'):
        (root / f'data/splits/{name}.jsonl').write_text('{}\n')
    config = {'seed': 42, 'training': {'epochs': 91, 'batch_size': 2},
              'model': {'class_path': 'pygosnet.models.PygosNet', 'kwargs': {'use_dcb': False}},
              'data': {'val_manifest': 'data/splits/val.jsonl'}}
    (root / 'configs/ablation').mkdir(parents=True)
    (root / 'configs/ablation/baseline.yaml').write_text(yaml.safe_dump(config))
    torch.save({'epoch': 37, 'config': config, 'model': {}}, root / 'saved.pt')
    return root


def install_fake_steps(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'environment_info', lambda requested: {
        'selected_device': 'cpu', 'requested_device': requested, 'gpu_name': None})
    def fake(root, out, name, command, state):
        calls.append((name, command))
        if name == 'train':
            config_path = Path(command[command.index('--config') + 1])
            cfg = yaml.safe_load(config_path.read_text())
            destination = out / 'training'
            destination.mkdir()
            torch.save({'epoch': cfg['training']['epochs'], 'config': cfg, 'model': {}}, destination / 'last.pt')
        if name == 'evaluate':
            (out / 'evaluation').mkdir()
            (out / 'evaluation/metrics.json').write_text(json.dumps({
                'overall': {'mae': 3.5, 'rmse': 4.5}, 'tiers': []}))
    monkeypatch.setattr(pipeline, 'run_step', fake)
    return calls


def test_default_is_check_only_no_training(tmp_path, monkeypatch):
    root = setup_project(tmp_path)
    calls = install_fake_steps(monkeypatch)
    out = pipeline.execute(pipeline.parser().parse_args(['--project-root', str(root)]))
    assert [name for name, _ in calls] == ['prepare_images']
    assert json.loads((out / 'pipeline_status.json').read_text())['status'] == 'complete'
    assert not (out / 'summary.json').exists()


def test_evaluate_uses_actual_epoch_and_requested_split(tmp_path, monkeypatch):
    root = setup_project(tmp_path)
    calls = install_fake_steps(monkeypatch)
    out = pipeline.execute(pipeline.parser().parse_args([
        '--project-root', str(root), '--mode', 'evaluate', '--checkpoint', 'saved.pt', '--split', 'test']))
    assert [name for name, _ in calls] == ['prepare_images', 'predict', 'evaluate']
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['checkpoint_epoch'] == 37
    assert summary['split'] == 'test'
    assert summary['manifest'].endswith('test.jsonl')
    assert '--use-seg' in calls[-1][1]


@pytest.mark.parametrize('override,expected', [([], 91), (['--epochs', '7'], 7)])
def test_training_budget_is_configurable(tmp_path, monkeypatch, override, expected):
    root = setup_project(tmp_path)
    calls = install_fake_steps(monkeypatch)
    out = pipeline.execute(pipeline.parser().parse_args([
        '--project-root', str(root), '--mode', 'train-eval', *override]))
    assert [name for name, _ in calls] == ['prepare_images', 'train', 'predict', 'evaluate']
    assert json.loads((out / 'summary.json').read_text())['checkpoint_epoch'] == expected
    assert yaml.safe_load((root / 'configs/ablation/baseline.yaml').read_text())['training']['epochs'] == 91


def test_existing_output_not_overwritten(tmp_path, monkeypatch):
    root = setup_project(tmp_path)
    install_fake_steps(monkeypatch)
    out = root / 'existing'
    out.mkdir()
    marker = out / 'keep.txt'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        pipeline.execute(pipeline.parser().parse_args(['--project-root', str(root), '--output-dir', str(out)]))
    assert marker.read_text() == 'keep'


def test_stage_failure_is_recorded_and_stops_pipeline(tmp_path, monkeypatch):
    root = setup_project(tmp_path)
    install_fake_steps(monkeypatch)
    calls = []
    def fail(root, out, name, command, state):
        calls.append(name)
        raise RuntimeError('data broken')
    monkeypatch.setattr(pipeline, 'run_step', fail)
    out = root / 'failure'
    with pytest.raises(RuntimeError, match='data broken'):
        pipeline.execute(pipeline.parser().parse_args([
            '--project-root', str(root), '--mode', 'evaluate', '--checkpoint', 'saved.pt', '--output-dir', str(out)]))
    assert calls == ['prepare_images']
    assert json.loads((out / 'pipeline_status.json').read_text())['status'] == 'failed'


def test_subprocess_failure_retains_log(tmp_path):
    state = {'steps': []}
    with pytest.raises(RuntimeError, match='exit 7'):
        pipeline.run_step(tmp_path, tmp_path, 'child',
                          [sys.executable, '-u', '-c', 'print("failure detail"); raise SystemExit(7)'], state)
    assert 'failure detail' in (tmp_path / 'child.txt').read_text()
    assert state['steps'][0]['exit_code'] == 7


@pytest.mark.parametrize('options', [
    ['--mode', 'evaluate'], ['--epochs', '2'], ['--checkpoint', 'saved.pt'],
    ['--split', 'test'], ['--mode', 'train-eval', '--epochs', '0'],
])
def test_invalid_modes_rejected_before_work(tmp_path, options):
    root = setup_project(tmp_path)
    with pytest.raises(ValueError):
        pipeline.execute(pipeline.parser().parse_args(['--project-root', str(root), *options]))
