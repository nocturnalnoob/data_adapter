"""Mines the 39-clip corpus for the SS7 calibration quantities, and the
FP-class frequency / weak-label checks that feed SS8 validation. No
hand-level ground truth exists anywhere in this corpus, so every quantity
here is measured off weak labels available for free in the raw data:

- "stable tracks": built with a permissive, effectively-unconstrained pass
  (dedup only, generous association gate, no size/shape/speed/static
  rejection) and then filtered to long ones. A track that persists for a
  long, mostly-continuous stretch without any of our rules touching it is
  about as close to "almost certainly a real hand" as this corpus gets, and
  gives us box-size/aspect/speed/dropout distributions to set thresholds
  from instead of picking them by eye (instruction.md SS7: "No value here
  should be set by inspection").
- the README's observation that any frame with >=3 raw boxes holds >=1
  duplicate/bystander/FP, used to (a) rank candidate-pool width by how often
  a "true" (top-2-track) box is outranked by a spurious one, and (b) label
  wearer vs. bystander-candidate boxes for the stereo disparity split.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict

import numpy as np

from adapter.config import AdapterConfig, hands_config
from adapter.io_clip import ClipData, list_clip_ids, load_clip
from adapter.geometry import area, aspect_ratio, centroid, diag
from adapter.reject_geometric import geometric_reject_frame
from adapter.association import associate
from adapter.selection import rank_tracks_per_frame
from adapter.types import Track
from adapter.vio_utils import VioSeries
from adapter.stereo import VideoPairReader, box_disparity

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adapter_out")


def _loose_config(base: AdapterConfig) -> AdapterConfig:
    """Dedup only; everything else wide open so continuity alone drives
    track formation, uncontaminated by not-yet-calibrated thresholds."""
    cfg = copy.deepcopy(base)
    cfg.min_area_px2 = 0.0
    cfg.max_area_px2 = 1e12
    cfg.max_aspect_ratio = 1e6
    cfg.stereo_enabled = False
    cfg.max_speed_px_per_frame = 600.0  # generous but not infinite (avoid pathological matches)
    cfg.max_dropout_frames = 90  # ~3s at 30fps: let long gaps through for inspection
    cfg.resume_gate_px = 250.0
    return cfg


def build_loose_tracks(clip: ClipData, base_cfg: AdapterConfig) -> list[Track]:
    cfg = _loose_config(base_cfg)
    frame_cands = {}
    for fidx, dets in clip.frames.items():
        cands, _ = geometric_reject_frame(dets, cfg)
        frame_cands[fidx] = cands
    return associate(frame_cands, cfg)


def stable_tracks(tracks: list[Track], min_len: int = 30) -> list[Track]:
    return [t for t in tracks if len(t) >= min_len]


# ---------------------------------------------------------------------------
# size / shape / speed from stable tracks
# ---------------------------------------------------------------------------

def size_shape_speed_stats(tracks: list[Track]) -> dict:
    areas, aspects, speeds = [], [], []
    for t in tracks:
        frames = t.frames
        for f in frames:
            b = t.candidates[f].xyxy
            areas.append(area(b))
            aspects.append(aspect_ratio(b))
        for i in range(1, len(frames)):
            f0, f1 = frames[i - 1], frames[i]
            if f1 - f0 != 1:
                continue  # only consecutive-frame pairs -> genuine per-frame speed
            c0 = centroid(t.candidates[f0].xyxy)
            c1 = centroid(t.candidates[f1].xyxy)
            speeds.append(((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5)

    def pct(vals, ps):
        return {str(p): float(np.percentile(vals, p)) for p in ps} if vals else {}

    return {
        "n_boxes": len(areas),
        "area_pct": pct(areas, [1, 5, 50, 95, 99, 99.5]),
        "aspect_pct": pct(aspects, [50, 90, 95, 99, 99.5]),
        "speed_pct": pct(speeds, [50, 90, 95, 99, 99.5, 99.9]),
    }


# ---------------------------------------------------------------------------
# dropout-length distribution (flicker gaps on stable tracks)
# ---------------------------------------------------------------------------

def dropout_stats(tracks: list[Track]) -> dict:
    gaps = []
    for t in tracks:
        frames = t.frames
        for i in range(1, len(frames)):
            f0, f1 = frames[i - 1], frames[i]
            gap = f1 - f0 - 1
            if gap <= 0:
                continue
            b0 = t.candidates[f0].xyxy
            b1 = t.candidates[f1].xyxy
            c0, c1 = centroid(b0), centroid(b1)
            resume_dist = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
            box_scale = max(diag(b0), diag(b1))
            if box_scale > 0 and resume_dist <= 1.5 * box_scale:
                gaps.append(gap)  # "flicker": resumed close to where it left off

    if not gaps:
        return {"n_gaps": 0}
    ps = [50, 75, 90, 95, 99]
    return {
        "n_gaps": len(gaps),
        "gap_pct": {str(p): float(np.percentile(gaps, p)) for p in ps},
        "max_gap_seen": int(max(gaps)),
    }


# ---------------------------------------------------------------------------
# candidate-pool width: rank of the "true" (top-2-track) box among raw
# per-frame confidences, in frames with >=3 raw detections
# ---------------------------------------------------------------------------

def pool_width_stats(clip: ClipData, tracks: list[Track]) -> dict:
    ranked_per_frame = rank_tracks_per_frame(tracks)
    by_track = {t.track_id: t for t in tracks}
    true_ranks = []

    for fidx, dets in clip.frames.items():
        if len(dets) < 3:
            continue
        top2 = ranked_per_frame.get(fidx, [])[:2]
        true_boxes = [by_track[tid].candidates[fidx].xyxy for tid in top2 if fidx in by_track[tid].candidates]
        if not true_boxes:
            continue
        by_conf = sorted(dets, key=lambda d: d.confidence, reverse=True)
        for tb in true_boxes:
            for rank, d in enumerate(by_conf, start=1):
                if d.xyxy == tb:
                    true_ranks.append(rank)
                    break

    return {"n_true_boxes_in_multibox_frames": len(true_ranks), "ranks": true_ranks}


# ---------------------------------------------------------------------------
# stereo disparity: wearer (top-2 track) vs bystander-candidate boxes
# ---------------------------------------------------------------------------

def stereo_disparity_stats(clip: ClipData, tracks: list[Track], max_frames: int = 150) -> dict:
    ranked_per_frame = rank_tracks_per_frame(tracks)
    by_track = {t.track_id: t for t in tracks}

    candidate_frames = [f for f, dets in clip.frames.items() if len(dets) >= 3]
    if len(candidate_frames) > max_frames:
        stride = len(candidate_frames) / max_frames
        candidate_frames = [candidate_frames[int(i * stride)] for i in range(max_frames)]

    reader = VideoPairReader(clip.video_left_path(), clip.video_right_path())
    wearer_disp, bystander_disp = [], []
    try:
        for fidx in candidate_frames:
            pair = reader.gray_pair(fidx)
            if pair is None:
                continue
            gl, gr = pair
            top2 = set(ranked_per_frame.get(fidx, [])[:2])
            wearer_boxes = {
                by_track[tid].candidates[fidx].xyxy
                for tid in top2 if fidx in by_track[tid].candidates
            }
            for d in clip.frames[fidx]:
                disp, conf = box_disparity(gl, gr, d.xyxy)
                if disp is None or conf < 0.4:
                    continue
                if d.xyxy in wearer_boxes:
                    wearer_disp.append(disp)
                else:
                    bystander_disp.append(disp)
    finally:
        reader.release()

    return {"wearer_disp": wearer_disp, "bystander_disp": bystander_disp}


def pick_disparity_threshold(
    wearer_disp: list[float], bystander_disp: list[float], min_accuracy: float = 0.68
) -> dict:
    """Scans for the best-separating threshold and reports whether it's
    trustworthy enough to use. Two failure modes are checked explicitly,
    because a threshold that merely beats a low bar without checking these
    is worse than no threshold: (1) the direction must match physics
    (wearer boxes, being closer, should have *larger* disparity than
    bystander candidates -- if the medians come out backwards, the wearer
    label itself is unreliable, not just noisy) and (2) accuracy at the
    best split must clear min_accuracy."""
    if not wearer_disp or not bystander_disp:
        return {"threshold": None, "trusted": False, "reason": "insufficient samples"}

    lo, hi = min(bystander_disp + wearer_disp), max(bystander_disp + wearer_disp)
    best_t, best_score = None, -1.0
    for t in np.linspace(lo, hi, 200):
        correct = sum(1 for d in wearer_disp if d >= t) + sum(1 for d in bystander_disp if d < t)
        score = correct / (len(wearer_disp) + len(bystander_disp))
        if score > best_score:
            best_score, best_t = score, float(t)

    direction_ok = float(np.median(wearer_disp)) > float(np.median(bystander_disp))
    trusted = direction_ok and best_score >= min_accuracy
    reason = None
    if not direction_ok:
        reason = (
            "wearer-labeled boxes have LOWER median disparity than bystander-labeled "
            "boxes -- backwards from physical expectation (closer = larger disparity). "
            "The 'top-2-longest-track = wearer' weak label is likely dominated by "
            "near-duplicate/detector-noise boxes at the same depth as the real hand in "
            "multi-box frames, not genuine distant bystanders. Needs labelled examples."
        )
    elif best_score < min_accuracy:
        reason = f"best achievable split accuracy ({best_score:.2f}) below trust bar ({min_accuracy})"

    return {
        "threshold": best_t, "accuracy": best_score, "trusted": trusted, "reason": reason,
        "wearer_median": float(np.median(wearer_disp)), "bystander_median": float(np.median(bystander_disp)),
    }


# ---------------------------------------------------------------------------
# camera-motion / static-rule thresholds
# ---------------------------------------------------------------------------

def camera_motion_stats(clips: list[ClipData]) -> dict:
    speeds, ang_rates = [], []
    for clip in clips:
        vio = VioSeries(clip.vio)
        speeds.extend(vio.speed.tolist())
        ang_rates.extend(vio.ang_rate_deg.tolist())
    return {
        "speed_pct": {str(p): float(np.percentile(speeds, p)) for p in [10, 25, 40, 50, 75, 90]},
        "ang_rate_pct": {str(p): float(np.percentile(ang_rates, p)) for p in [10, 25, 40, 50, 75, 90]},
    }


def static_motion_threshold(clip_tracks: list[tuple[ClipData, list[Track]]], moving_speed_t: float, moving_ang_t: float) -> dict:
    speeds_while_moving = []
    for clip, tracks in clip_tracks:
        vio = VioSeries(clip.vio)
        for t in tracks:
            frames = t.frames
            for i in range(1, len(frames)):
                f0, f1 = frames[i - 1], frames[i]
                if f1 - f0 != 1:
                    continue
                if not vio.is_camera_moving(f1, moving_speed_t, moving_ang_t):
                    continue
                c0 = centroid(t.candidates[f0].xyxy)
                c1 = centroid(t.candidates[f1].xyxy)
                speeds_while_moving.append(((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5)
    if not speeds_while_moving:
        return {"n": 0}
    return {
        "n": len(speeds_while_moving),
        "pct": {str(p): float(np.percentile(speeds_while_moving, p)) for p in [1, 5, 10, 25, 50]},
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def calibrate(dev_clip_ids: list[str], all_clip_ids: list[str] | None = None) -> tuple[AdapterConfig, dict]:
    all_clip_ids = all_clip_ids or list_clip_ids()
    base = hands_config()

    all_clips = [load_clip(cid) for cid in all_clip_ids]
    loose_tracks_by_clip = {c.cid: build_loose_tracks(c, base) for c in all_clips}
    stable_by_clip = {cid: stable_tracks(ts) for cid, ts in loose_tracks_by_clip.items()}
    all_stable = [t for ts in stable_by_clip.values() for t in ts]

    sss = size_shape_speed_stats(all_stable)
    drop = dropout_stats(all_stable)

    pool_ranks = []
    for c in all_clips:
        r = pool_width_stats(c, loose_tracks_by_clip[c.cid])
        pool_ranks.extend(r["ranks"])
    pool_ranks_arr = np.array(pool_ranks) if pool_ranks else np.array([1])
    pool_width = int(max(2, np.percentile(pool_ranks_arr, 95))) if len(pool_ranks_arr) else 4

    wearer_disp, bystander_disp = [], []
    dev_clips = [c for c in all_clips if c.cid in dev_clip_ids]
    for c in dev_clips:
        r = stereo_disparity_stats(c, loose_tracks_by_clip[c.cid])
        wearer_disp.extend(r["wearer_disp"])
        bystander_disp.extend(r["bystander_disp"])
    disp_result = pick_disparity_threshold(wearer_disp, bystander_disp)
    reach_threshold = disp_result["threshold"] if disp_result["trusted"] else None

    cam_stats = camera_motion_stats(all_clips)
    moving_speed_t = cam_stats["speed_pct"]["40"]
    moving_ang_t = cam_stats["ang_rate_pct"]["40"]
    static_stats = static_motion_threshold(
        [(c, stable_by_clip[c.cid]) for c in all_clips], moving_speed_t, moving_ang_t
    )

    cfg = hands_config()
    if sss["area_pct"]:
        cfg.min_area_px2 = sss["area_pct"]["1"] * 0.5
        cfg.max_area_px2 = sss["area_pct"]["99.5"] * 1.5
    if sss["aspect_pct"]:
        cfg.max_aspect_ratio = max(1.5, sss["aspect_pct"]["99"] * 1.1)
    if sss["speed_pct"]:
        cfg.max_speed_px_per_frame = sss["speed_pct"]["99.5"] * 1.2
    if drop.get("n_gaps"):
        cfg.max_dropout_frames = max(2, int(np.ceil(drop["gap_pct"]["90"])))
    cfg.candidate_pool_width = pool_width
    cfg.stereo_enabled = reach_threshold is not None
    cfg.reach_disparity_threshold_px = reach_threshold
    cfg.camera_speed_moving_thresh = moving_speed_t
    cfg.camera_ang_moving_thresh_deg = moving_ang_t
    if static_stats.get("n"):
        cfg.static_motion_px_threshold = max(1.0, static_stats["pct"]["10"])

    report = {
        "size_shape_speed": sss,
        "dropout": drop,
        "pool_width_ranks_p95": float(np.percentile(pool_ranks_arr, 95)) if len(pool_ranks_arr) else None,
        "n_pool_samples": len(pool_ranks),
        "stereo": {
            "n_wearer": len(wearer_disp), "n_bystander": len(bystander_disp),
            **disp_result,
        },
        "camera_motion": cam_stats,
        "static_motion": static_stats,
        "derived_config": cfg.to_json(),
    }
    return cfg, report


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev_ids = ["887c633e_t028", "c6754e88_t019", "ce045be4_t019"]
    cfg, report = calibrate(dev_clip_ids=dev_ids)

    with open(os.path.join(OUT_DIR, "calibrated_config.json"), "w") as f:
        json.dump(cfg.to_json(), f, indent=1)
    with open(os.path.join(OUT_DIR, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(cfg.to_json(), indent=1))
    print("\nwrote", OUT_DIR)


if __name__ == "__main__":
    main()
