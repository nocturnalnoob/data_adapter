"""CLI: run the calibrated hand adapter over one or all clips and write
hand_boxes_corrected.json per clip (instruction.md SS2's "Returns" contract
-- every detection tagged reported/merged/rejected/interpolated).

Usage:
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --clip 887c633e_t028
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter.config import AdapterConfig, hands_config
from adapter.io_clip import list_clip_ids, load_clip
from adapter.pipeline import run_pipeline, check_completeness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "adapter_out", "calibrated_config.json")
DEFAULT_OUT = os.path.join(ROOT, "adapter_out", "corrected")


def load_config(path: str) -> AdapterConfig:
    if os.path.exists(path):
        with open(path) as f:
            return AdapterConfig(**json.load(f))
    print(f"warning: {path} not found, using uncalibrated hands_config() defaults", file=sys.stderr)
    return hands_config()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="single clip id")
    ap.add_argument("--all", action="store_true", help="run over every clip in downloads/")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.clip and not args.all:
        ap.error("pass --clip <cid> or --all")

    cfg = load_config(args.config)
    cids = [args.clip] if args.clip else list_clip_ids()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for cid in cids:
        clip = load_clip(cid)
        result = run_pipeline(clip, cfg)
        completeness = check_completeness(clip, result)

        out_path = os.path.join(args.out_dir, f"{cid}.json")
        with open(out_path, "w") as f:
            json.dump(result.to_json(), f, indent=1)

        status_counts = result.by_status()
        summary[cid] = {"status_counts": status_counts, "completeness": completeness}
        print(f"{cid}: {status_counts}  completeness_ok={completeness['ok']}")

    summary_path = os.path.join(args.out_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {len(cids)} clip(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
