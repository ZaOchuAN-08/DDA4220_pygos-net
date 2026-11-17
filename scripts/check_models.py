"""Synthetic engineering checks, not counting-accuracy experiments.

Run after `pip install -e .`: python scripts/check_models.py --device cpu
"""
import argparse
import json
import platform
from pathlib import Path
import time
import torch
import torchvision
from torch.nn import functional as F
from pygosnet.models import PygosNet


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--output', default='outputs/model_checks.json')
    args = parser.parse_args()
    selected = args.device
    if selected == 'auto':
        selected = ('cuda' if torch.cuda.is_available() else
                    'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(selected)
    torch.set_num_threads(4)
    report = {'python': platform.python_version(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'platform': platform.platform(),
              'device': str(device), 'mps_built': torch.backends.mps.is_built(),
              'mps_available': torch.backends.mps.is_available(),
              'pretrained': args.pretrained, 'seed': 42,
              'note': 'Synthetic inputs; no MAE/RMSE or training-speed benchmark. '
                      'Times are single cold forwards; gradients use a controlled +0.1 output bias.',
              'variants': []}
    for use_dcb in (False, True):
        torch.manual_seed(42)
        model = PygosNet(use_dcb=use_dcb, pretrained=args.pretrained).to(device)
        entry = {'use_dcb': use_dcb,
                 'parameters': sum(p.numel() for p in model.parameters()), 'shapes': []}
        model.eval()
        for size in (256, 448):
            x = torch.randn(1, 3, size, size, device=device)
            synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                out = model.predict(x)
            synchronize(device)
            assert out['density'].shape == (1, 1, size // 8, size // 8)
            assert (out['density'] >= 0).all().item()
            assert all(torch.isfinite(t).all().item() for t in out.values())
            torch.testing.assert_close(out['count'], out['density_masked'].sum((1, 2, 3)))
            entry['shapes'].append({'input': list(x.shape),
                                    'density': list(out['density'].shape),
                                    'cold_forward_seconds': time.perf_counter() - start})
            del x, out
        model.train()
        with torch.no_grad():
            model.density_head[-2].bias.fill_(.1)
        x = torch.randn(1, 3, 256, 256, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        out = model(x)
        loss = F.mse_loss(out['density'], torch.ones_like(out['density']))
        loss += F.binary_cross_entropy_with_logits(out['segmentation_logits'],
                                                   torch.ones_like(out['segmentation']))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
                   for p in model.parameters())
        optimizer.step()
        synchronize(device)
        entry['synthetic_256_backward_and_adam_step'] = 'passed'
        if selected == 'mps':
            entry['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
            entry['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        if selected == 'cuda':
            entry['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        report['variants'].append(entry)
        del model, optimizer, x, out, loss
        if selected == 'mps':
            torch.mps.empty_cache()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
