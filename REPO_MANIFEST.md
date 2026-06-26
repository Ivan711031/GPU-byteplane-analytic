# Repository Manifest

This repository is a public, filtered export of the internal working tree.

## Goal

Keep the materials needed for the report and paper story:

- experiment and plotting scripts
- benchmark / source code
- research notes
- paper drafts
- result summaries, figures, and claim-facing CSVs

## Included result artifact types

- `*.csv`
- `*.json`
- `*.md`
- `*.txt`
- `*.png`
- `*.svg`
- `*.pdf`
- `*.tsv`
- `*.log`

## Excluded artifact types

- `*.bin`
- `*.f64le.bin`
- `*.buff64`
- decoded binary exports
- `*.ncu-rep`
- local env / cache files

## Important consequence

This repo is designed for:

- reading the report
- tracing paper claims to committed summaries
- rerunning plotting / aggregation scripts against included summaries

This repo is not designed for:

- rebuilding every artifact from raw source datasets alone
- preserving all intermediate binary outputs
- acting as an archival mirror of the full internal workspace
