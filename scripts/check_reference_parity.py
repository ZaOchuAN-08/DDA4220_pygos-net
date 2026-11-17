"""Compare against the inspected, pinned author archive (download separately).

python scripts/check_reference_parity.py --reference /path/to/model/vgg.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pygosnet.models import Baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', default='outputs/reference_parity.json')
    args = parser.parse_args()
    path = Path(args.reference)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = '873eec5c55780ad7d87c25ddef63622ebb47b7a15cccceae23f5ba79c189e0d7'
    if digest != expected:
        raise ValueError('Reference file differs from the inspected Zenodo archive; review before running.')
    source = '\n'.join(line for line in path.read_text().splitlines()
                       if not line.startswith('from timm.'))
    namespace = {}
    exec(compile(source, str(path), 'exec'), namespace)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    reference = namespace['VGG'](namespace['make_layers'](namespace['cfg']['E'])).eval()
    model = Baseline(pretrained=False).eval()
    model.encoder.features.load_state_dict(reference.features.state_dict())
    model.density_head.load_state_dict(reference.reg_layer.state_dict())
    model.segmentation_head.load_state_dict(reference.cls_16.state_dict())
    results = []
    for size in (256, 448):
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            den, seg = reference(x)
            actual = model(x)
        torch.testing.assert_close(actual['density'], den, rtol=0, atol=0)
        torch.testing.assert_close(actual['segmentation'], seg, rtol=0, atol=0)
        results.append({'input_size': size,
                        'density_max_abs_difference': (actual['density'] - den).abs().max().item(),
                        'segmentation_max_abs_difference': (actual['segmentation'] - seg).abs().max().item()})
    report = {'reference_url': 'https://zenodo.org/records/7317264',
              'reference_file_sha256': digest,
              'method': 'CPU float32; identical randomly initialized parameters; '
                        'unused timm import removed; no pretrained download.',
              'results': results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
