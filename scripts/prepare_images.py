#!/usr/bin/env python
"""Prepare the raw ZIPs already committed to the repository; preserve manifests."""
import argparse
import json
from pathlib import Path
from pygosnet.data.image_preparation import prepare_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--verify-only', action='store_true', help='Check all paths and images without extracting')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    report = prepare_images(args.project_root, args.verify_only)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
