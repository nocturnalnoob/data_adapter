"""Draws raw vs. corrected boxes on sampled frames, color-coded by status,
for qualitative spot-checking. There's no hand-level ground truth anywhere
in this corpus, so this is the substitute for eyeballing correctness on the
instruction.md SS6 edge cases (crossing hands, border exits, bystander
rejection) that the automated weak-label checks in validate.py can't
directly confirm.

Usage:
    python scripts/visualize.py --clip 887c633e_t028 --n-frames 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from adapter.config import AdapterConfig, hands_config
from adapter.io_clip import load_clip
from adapter.pipeline import run_pipeline
from adapter.types import Status

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "adapter_out", "calibrated_config.json")

COLORS = {  # BGR
    Status.REPORTED: (60, 200, 60),
    Status.MERGED: (220, 130, 40),
    Status.INTERPOLATED: (0, 210, 255),
    Status.REJECTED: (40, 40, 220),
}


def draw_frame(frame, dets_this_frame, raw_dets_this_frame, show_rejected: bool):
    out = frame.copy()
    if raw_dets_this_frame:
        for d in raw_dets_this_frame:
            x1, y1, x2, y2 = (int(v) for v in d.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for d in dets_this_frame:
        if d.status == Status.REJECTED and not show_rejected:
            continue
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = COLORS[d.status]
        thickness = 1 if d.status == Status.REJECTED else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = d.status.value + (f"#{d.track_id}" if d.track_id is not None else "")
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--show-rejected", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = AdapterConfig(**json.load(f))
    else:
        cfg = hands_config()

    clip = load_clip(args.clip)
    result = run_pipeline(clip, cfg)

    by_frame: dict[int, list] = {}
    for d in result.detections:
        by_frame.setdefault(d.frame, []).append(d)

    multibox_frames = sorted(f for f, dets in clip.frames.items() if len(dets) >= 3)
    candidate_frames = multibox_frames if multibox_frames else sorted(clip.frames.keys())
    if len(candidate_frames) > args.n_frames:
        stride = len(candidate_frames) / args.n_frames
        sample = [candidate_frames[int(i * stride)] for i in range(args.n_frames)]
    else:
        sample = candidate_frames

    out_dir = args.out_dir or os.path.join(ROOT, "adapter_out", "viz", args.clip)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(clip.video_left_path())
    last_idx = -1
    written = 0
    for fidx in sample:
        gap = fidx - (last_idx + 1)
        if gap < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        else:
            for _ in range(gap):
                cap.read()
        ok, frame = cap.read()
        last_idx = fidx
        if not ok:
            continue
        annotated = draw_frame(frame, by_frame.get(fidx, []), clip.frames.get(fidx, []), args.show_rejected)
        out_path = os.path.join(out_dir, f"frame_{fidx:05d}.png")
        cv2.imwrite(out_path, annotated)
        written += 1
    cap.release()
    print(f"wrote {written} annotated frames to {out_dir}")


if __name__ == "__main__":
    main()
