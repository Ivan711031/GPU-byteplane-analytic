#!/usr/bin/env python3
"""Add fault_plan_id to existing fault plan JSON metadata and generate inventory.

Usage:
  python3 scripts/patch_fault_plan_ids.py \
    --artifact-root /work/u4063895/datasets/reliability_layer1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    fp_root = root / "fault_plans"
    if not fp_root.exists():
        print(f"ERROR: {fp_root} not found")
        return

    patched = 0
    skipped = 0
    errors = 0
    inventory = {"generated": [], "skipped": [], "invalid": [], "exploratory": []}

    for fp_path in sorted(fp_root.rglob("*.json")):
        rel = str(fp_path.relative_to(root))

        try:
            data = json.loads(fp_path.read_text())
        except Exception as e:
            inventory["invalid"].append({"path": rel, "error": str(e)})
            errors += 1
            continue

        meta = data.get("metadata", {})
        existing_id = meta.get("fault_plan_id", "")

        if existing_id:
            skipped += 1
            inventory["skipped"].append(rel)
            continue

        # Construct fault_plan_id from metadata or directory structure
        if not existing_id:
            meta["fault_plan_id"] = rel

        if not args.dry_run:
            fp_path.write_text(json.dumps(data, indent=2))

        patched += 1
        inventory["generated"].append(rel)

    # Artifact inventory
    art_root = root / "artifacts"
    for ds_dir in sorted(art_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for n_dir in sorted(ds_dir.iterdir()):
            for scale_dir in sorted(n_dir.iterdir()):
                art_json = scale_dir / "artifact.json"
                if art_json.exists():
                    inventory["generated"].append(str(art_json.relative_to(root)))

    print(f"Patched: {patched}, Skipped (already has id): {skipped}, Errors: {errors}")

    inv_path = root / "inventory.json"
    if not args.dry_run:
        inv_path.write_text(json.dumps(inventory, indent=2))
        print(f"Inventory: {inv_path}")
    else:
        print(f"DRY RUN: would write {inv_path}")
        print(json.dumps(inventory, indent=2)[:2000])


if __name__ == "__main__":
    main()
