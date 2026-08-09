"""Ranks the 39 clips by fraction of frames with >=3 raw detections --
README notes multi-box frames concentrate where bystanders are present, so
this directly selects the clips that exercise the stereo/bystander path,
cheaply, without needing labels."""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter.io_clip import list_clip_ids, load_clip


def main():
    rows = []
    for cid in list_clip_ids():
        clip = load_clip(cid)
        n_frames = clip.frame_count
        n_multi = sum(1 for dets in clip.frames.values() if len(dets) >= 3)
        frac = n_multi / n_frames if n_frames else 0.0
        rows.append((cid, n_frames, n_multi, frac))

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'cid':<16} {'frames':>7} {'>=3box':>7} {'frac':>7}")
    for cid, n_frames, n_multi, frac in rows:
        print(f"{cid:<16} {n_frames:>7} {n_multi:>7} {frac:>7.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "adapter_out", "devset_ranking.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            [{"cid": c, "n_frames": n, "n_multibox": m, "frac_multibox": fr} for c, n, m, fr in rows],
            f, indent=1,
        )
    print(f"\nwrote {out_path}")
    print("\ntop 3 dev-set clips:", [r[0] for r in rows[:3]])


if __name__ == "__main__":
    main()
