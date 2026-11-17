"""Portable environment/data checks and checkpoint evaluation orchestration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import yaml

from .device import select_device


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def environment_info(requested):
    device = select_device(requested)
    # Actually allocate and compute; availability alone is insufficient.
    probe = (torch.ones(4, device=device) * 2).sum().item()
    if probe != 8:
        raise RuntimeError(f'Device computation check failed: {device}')
    versions = {}
    for package in ('torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML', 'pandas', 'matplotlib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f'Missing dependency: {package}. Install the project dependencies.')
    result = {'requested_device': requested, 'selected_device': str(device),
              'python': platform.python_version(), 'platform': platform.platform(),
              'packages': versions, 'cuda_available': torch.cuda.is_available(),
              'mps_available': torch.backends.mps.is_available(),
              'pytorch_cuda_build': torch.version.cuda, 'device_probe': 'passed'}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory)
    else:
        result['gpu_name'] = 'Apple GPU / MPS' if device.type == 'mps' else None
        result['cpu_architecture'] = platform.machine()
    return result


def source_info(root):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git('status', '--porcelain')
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'dirty': bool(status) if status is not None else None}


def run_step(root, out, name, command, state):
    """Stream output while retaining a log; do not silently continue on failure."""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['MPLCONFIGDIR'] = str(out / '.matplotlib')
    state['current_step'] = name
    step = {'name': name, 'command': command, 'status': 'running'}
    state['steps'].append(step)
    write_json(out / 'pipeline_status.json', state)
    print(f'\n=== {name} ===', flush=True)
    start = time.monotonic()
    with (out / f'{name}.txt').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            step.update(status='interrupted', seconds=time.monotonic() - start)
            raise
    step.update(status='passed' if code == 0 else 'failed', exit_code=code,
                seconds=time.monotonic() - start)
    write_json(out / 'pipeline_status.json', state)
    if code != 0:
        raise RuntimeError(f'{name} failed (exit {code}). See {out / (name + ".txt")}')


def parser():
    p = argparse.ArgumentParser(description='Check data/device, evaluate a checkpoint, or explicitly train then evaluate.')
    p.add_argument('--mode', choices=['check', 'evaluate', 'train-eval'], default='check')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    p.add_argument('--checkpoint', type=Path, help='Existing trusted project checkpoint for evaluate mode')
    p.add_argument('--config', type=Path, default=Path('configs/ablation/baseline.yaml'))
    p.add_argument('--manifest', type=Path, help='Override evaluation JSONL manifest')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--epochs', type=int, help='Train-eval only; omitted means use YAML budget')
    p.add_argument('--batch-size', type=int, help='Training batch size; omitted means use YAML')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--no-pretrained', action='store_true', help='Train-eval engineering checks only')
    p.add_argument('--output-dir', type=Path, help='New directory; existing directories are never overwritten')
    p.add_argument('--no-figures', action='store_true')
    return p


def project_path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def execute(args):
    if args.mode == 'evaluate' and not args.checkpoint:
        raise ValueError('--mode evaluate requires --checkpoint.')
    if args.mode != 'evaluate' and args.checkpoint:
        raise ValueError('--checkpoint is only used in evaluate mode; no implicit retraining/resume.')
    if args.mode != 'train-eval' and (args.epochs is not None or args.no_pretrained or args.batch_size is not None):
        raise ValueError('Training options require --mode train-eval.')
    if args.mode == 'check' and (args.manifest or args.split != 'val'):
        raise ValueError('Manifest/split selection requires evaluate or train-eval mode.')
    if args.num_workers < 0 or (args.epochs is not None and args.epochs < 1) or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError('Epochs/batch size must be positive; workers must be nonnegative.')
    root = args.project_root.resolve()
    if not (root / 'scripts/prepare_images.py').is_file():
        raise ValueError(f'Not an integration checkout: {root}')
    output = args.output_dir or Path('outputs/pipeline') / (
        datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.mode)
    out = project_path(root, output)
    # Inputs are checked before creating a run directory.
    checkpoint = project_path(root, args.checkpoint) if args.checkpoint else None
    if checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = None
    if args.mode == 'train-eval':
        config_path = project_path(root, args.config)
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            config['training']['batch_size'] = args.batch_size
        config['data'].update(num_workers=args.num_workers)
        if args.no_pretrained:
            config['model'].setdefault('kwargs', {})['pretrained'] = False
        # Device selection occurs below; source YAML will not be modified.
    out.mkdir(parents=True, exist_ok=False)
    state = {'mode': args.mode, 'status': 'running', 'current_step': 'environment',
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'source': source_info(root), 'steps': [], 'output_dir': str(out)}
    write_json(out / 'pipeline_status.json', state)
    try:
        environment = environment_info(args.device)
        device = environment['selected_device']
        write_json(out / 'environment.json', environment)
        print(f'Device: {device} | {environment.get("gpu_name") or "CPU"}', flush=True)
        print(f'Output: {out}', flush=True)
        python = sys.executable
        def script(name, *arguments):
            return [python, '-u', str(root / 'scripts' / name), *map(str, arguments)]
        run_step(root, out, 'prepare_images', script('prepare_images.py', '--project-root', root,
                 '--report', out / 'data_report.json'), state)
        if args.mode == 'check':
            state.update(status='complete', current_step=None,
                         message='Environment and all image/manifest checks passed; no training was started.')
        else:
            if args.mode == 'train-eval':
                config['device'] = device
                config['data']['pin_memory'] = device == 'cuda'
                config['output_dir'] = str(out / 'training')
                effective = out / 'effective_config.yaml'
                effective.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
                run_step(root, out, 'train', script('train.py', '--project-root', root,
                         '--config', effective, '--device', device), state)
                checkpoint = out / 'training/last.pt'
            # Checkpoint is trusted local project data, including config and optimizer state.
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            checkpoint_config = saved['config']
            actual_epoch = int(saved['epoch'])
            manifest = project_path(root, args.manifest or checkpoint_config['data'].get(
                f'{args.split}_manifest', f'data/splits/{args.split}.jsonl'))
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            summary = {'mode': args.mode, 'split': args.split, 'manifest': str(manifest),
                       'manifest_sha256': sha256(manifest), 'checkpoint': str(checkpoint),
                       'checkpoint_sha256': sha256(checkpoint), 'checkpoint_epoch': actual_epoch,
                       'checkpoint_selection': 'Provided checkpoint' if args.mode == 'evaluate' else 'Final epoch (last.pt), not validation-selected best',
                       'model': checkpoint_config['model'], 'training': checkpoint_config['training'],
                       'device': device, 'source': state['source']}
            write_json(out / 'checkpoint_config.json', checkpoint_config)
            del saved
            run_step(root, out, 'predict', script('predict.py', '--checkpoint', checkpoint,
                     '--project-root', root, '--manifest', manifest, '--out-dir', out / 'predictions',
                     '--device', device, '--batch-size', 1), state)
            command = script('evaluate.py', '--manifest', manifest, '--pred-dir', out / 'predictions',
                             '--use-seg', '--out-dir', out / 'evaluation')
            if args.no_figures:
                command.append('--no-figures')
            run_step(root, out, 'evaluate', command, state)
            metrics = json.loads((out / 'evaluation/metrics.json').read_text(encoding='utf-8'))
            summary['overall'] = metrics['overall']
            summary['tiers'] = metrics['tiers']
            write_json(out / 'summary.json', summary)
            print(f'\nEpoch {actual_epoch} | {args.split} MAE={metrics["overall"]["mae"]:.4f} '
                  f'RMSE={metrics["overall"]["rmse"]:.4f}', flush=True)
            state.update(status='complete', current_step=None, checkpoint_epoch=actual_epoch)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        write_json(out / 'pipeline_status.json', state)
        raise
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(out / 'pipeline_status.json', state)
    print(f'Pipeline complete: {out}', flush=True)
    return out


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        execute(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        p.exit(1, f'Pipeline error: {error}\n')


if __name__ == '__main__':
    main()
