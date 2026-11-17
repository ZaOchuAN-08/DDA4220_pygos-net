# Member B handoff

Local branch: `feat/model-architecture`, based on repository commit `f6bfd62`.
No changes have been pushed to GitHub.

## Deliverables

- `src/pygosnet/models/`: encoder, baseline, DCB, shared model and exports.
- `tests/test_models.py`: 16 synthetic CPU tests.
- `scripts/check_models.py`: device and synthetic forward/backward checks.
- `docs/architecture.md`: exact model API, source references and reproduction limits.
- `pyproject.toml` and `.gitignore`: minimal install/test setup and generated-file exclusions.

## Proposed interfaces (not yet agreed by A/C)

A supplies normalized RGB `[B,3,H,W]`, with H/W divisible by 16. C receives `density` (unmasked), `segmentation` (probability) and `segmentation_logits` (logits), all `[B,1,H/8,W/8]`. Segmentation supervision is mandatory because the threshold gate is nondifferentiable. D gets masked predictions through `model.eval(); model.predict(images)`.

## Evidence and limits

The delivery archive includes test logs, device-check JSON and author-model parity JSON. CPU tests and MPS batch-1 engineering checks passed. These checks do not demonstrate counting accuracy, improved background suppression or real-data training convergence. Pretrained-loading status is recorded separately. The English report is a B-only contribution section, to be merged with the other members' work.

Before merging, A and C should review the proposed API and preprocessing. No team communication or remote write has been performed.
