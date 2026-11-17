"""Export raw density and segmentation from a checkpoint for D's evaluator."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pygosnet.data.dataset import PenguinDataset
from pygosnet.engine.device import select_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=1)
    args = p.parse_args()
    device = select_device(args.device)
    # Load only a trusted project checkpoint, which includes its model config.
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    model_config = config['model']
    module, name = model_config['class_path'].rsplit('.', 1)
    kwargs = dict(model_config.get('kwargs', {}))
    # Checkpoint supplies every weight; downloading initialization is unnecessary.
    kwargs['pretrained'] = False
    model = getattr(importlib.import_module(module), name)(**kwargs)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    data = config['data']
    dataset = PenguinDataset(args.manifest, args.project_root.resolve(), split='val',
                             sigma=data['sigma'], stride=data['stride'],
                             support_threshold=data['support_threshold'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    written = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['image'].to(device))
            for i, key in enumerate(batch['sample_key']):
                destination = args.out_dir / Path(key).with_suffix('.npz')
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination,
                                    density=outputs['density'][i].cpu().numpy(),
                                    segmentation=outputs['segmentation'][i].cpu().numpy())
                written += 1
    print(f'Exported {written} unmasked density/segmentation pairs to {args.out_dir}.')
    print('Evaluate with --use-seg to apply the proposal inference mask.')


if __name__ == '__main__':
    main()
