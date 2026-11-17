# Data preparation

Download the Dryad archive `doi_10_5061_dryad_8931zcrv8__v20230306.zip` and keep it outside Git. The archive contains four image ZIPs and four Labelbox JSON exports.

Run the preparation command first **without** a duplicate-source choice:

```bash
python scripts/prepare_data.py --archive /path/to/doi_10_5061_dryad_8931zcrv8__v20230306.zip
```

It extracts PNGs/JSON files, writes `data/processed/dataset_audit.json`, and stops if multiple non-identical annotations map to the same image. The supplied archive has 738 PNG images, 753 Labelbox records, 740 unique `External ID` values, 13 duplicate image IDs, and 2 annotation IDs without a matching PNG. The 13 duplicate annotations are not identical. Do not merge them or select one silently.

After the group/instructor confirms which annotator export to use for those overlaps, rerun with:

```bash
python scripts/prepare_data.py --archive /path/to/doi_10_5061_dryad_8931zcrv8__v20230306.zip --duplicate-source MAISIE
# or --duplicate-source LUKE, if that is the agreed choice
```

The first run (even without a duplicate-source decision) writes the image lists `data/splits/train.txt`, `val.txt`, and `test.txt` with 446/146/146 filenames. After an annotation source is selected, the rerun also writes `data/processed/manifest.jsonl` and the corresponding `train.jsonl`, `val.jsonl`, and `test.jsonl` files. `split_metadata.json` records seed, method, sizes, and the provenance warning.

The split is a fixed-seed random reconstruction of the paper's published 446/146/146 sizes. It is **not** claimed to be the authors' original per-image assignment. Keep these generated manifests unchanged for every ablation.
