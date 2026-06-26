# GPU-byteplane-analysis

Public reproducibility-oriented snapshot for the GPU byte-plane analysis project.

This repository keeps the materials needed to understand and reproduce the report/storyline:

- source code and benchmark code
- plotting and experiment scripts
- `research/` notes and reports
- `paper/` drafts
- filtered `results/` artifacts used for analysis, figures, and claims

The main concise report is:

- `research/2026-06-25_project_concise_report_zh.md`

## Included

- `benchmarks/`
- `buff_encoder/`
- `scripts/`
- `tests/`
- `paper/`
- `research/`
- `spec/`
- `results/`:
  includes figures, CSV summaries, JSON manifests, reports, and paper-facing result packets

## Excluded

Large raw artifacts are intentionally excluded from this public repo, including:

- raw binary datasets such as `*.bin` and `*.f64le.bin`
- encoded container blobs such as `*.buff64`
- decoded binary dumps
- large Nsight Compute report blobs such as `*.ncu-rep`
- local virtual environments and build outputs
- the original `datasets/` tree

These files are not needed to read the report, inspect the claims, or regenerate the included figures from the committed summaries.

## Notes

- This is a curated export from the working research repository, not a full byte-for-byte mirror.
- Claim boundaries and artifact-selection rationale are documented in `REPO_MANIFEST.md`.
