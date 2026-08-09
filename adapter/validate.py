"""SS8 validation: per-stage precision/recall, and the interpolation-rate
standing check. No hand-level ground truth exists, so every metric here is
either (a) proxy precision/recall against the same "stable loose track"
weak label used in calibrate.py, explicitly flagged as a proxy, or (b) fully
label-free (interpolation rate, synthetic-dropout recovery error).

Synthetic dropout injection is the one recall check that needs no proxy
label at all: take a real, contiguous run of observed frames on a selected
track, blank out an interior span, run interpolation, and compare the
synthesized boxes against the real (withheld) ones. This directly measures
interpolation quality against ground truth without ever needing hand
annotations, because the "ground truth" is just the real detections we
chose to hide.
"""
from __future__ import annotations

import copy
import json
import os
import random

import numpy as np

from adapter.config import AdapterConfig
from adapter.io_clip import ClipData, list_clip_ids, load_clip
from adapter.geometry import centroid, iou, diag
from adapter.reject_geometric import geometric_reject_frame
from adapter.association import associate
from adapter.reject_temporal import reject_temporal_all
from adapter.selection import select_cap
from adapter.interpolate import interpolate_tracks
from adapter.vio_utils import VioSeries
from adapter.pipeline import run_pipeline, PipelineResult
from adapter.calibrate import build_loose_tracks, stable_tracks
from adapter.types import Track, Status

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adapter_out")


# ---------------------------------------------------------------------------
# proxy precision / recall through the rejection stages
# ---------------------------------------------------------------------------

def proxy_precision_recall(clip: ClipData, result: PipelineResult, real_ids: set[int]) -> dict:
    total_raw = sum(len(d) for d in clip.frames.values())
    stages = {"raw": {d for dets in clip.frames.values() for d in [x.det_id for x in dets]}}
    stages.update(result.stage_survivor_ids)

    out = {}
    for name, ids in stages.items():
        n = len(ids)
        tp = len(ids & real_ids)
        precision = tp / n if n else None
        recall = tp / len(real_ids) if real_ids else None
        out[name] = {"n_survivors": n, "precision_proxy": precision, "recall_proxy": recall}
    return out


# ---------------------------------------------------------------------------
# frame-coverage recall (captures interpolation's recall contribution)
# ---------------------------------------------------------------------------

def frame_coverage_recall(clip: ClipData, stable_loose: list[Track], result: PipelineResult) -> dict:
    by_frame_final: dict[int, list] = {}
    by_frame_final_no_interp: dict[int, list] = {}
    for d in result.detections:
        if d.status == Status.REJECTED:
            continue
        by_frame_final.setdefault(d.frame, []).append(d)
        if d.status != Status.INTERPOLATED:
            by_frame_final_no_interp.setdefault(d.frame, []).append(d)

    def covered(frame, box, index):
        cands = index.get(frame, [])
        if not cands:
            return False
        bc = centroid(box)
        bscale = max(diag(box), 1.0)
        for d in cands:
            dc = centroid(d.xyxy)
            dist = ((bc[0] - dc[0]) ** 2 + (bc[1] - dc[1]) ** 2) ** 0.5
            if dist <= bscale:
                return True
        return False

    total, covered_with_interp, covered_without_interp = 0, 0, 0
    for t in stable_loose:
        for f in t.frames:
            box = t.candidates[f].xyxy
            total += 1
            if covered(f, box, by_frame_final):
                covered_with_interp += 1
            if covered(f, box, by_frame_final_no_interp):
                covered_without_interp += 1

    return {
        "n_stable_track_frames": total,
        "recall_without_interpolation": covered_without_interp / total if total else None,
        "recall_with_interpolation": covered_with_interp / total if total else None,
        "recall_gain_from_interpolation": (
            (covered_with_interp - covered_without_interp) / total if total else None
        ),
    }


# ---------------------------------------------------------------------------
# multi-box frame resolution + interpolation-rate monitor (label-free)
# ---------------------------------------------------------------------------

def multibox_resolution(clip: ClipData, result: PipelineResult, class_max: int = 2) -> dict:
    by_frame: dict[int, int] = {}
    for d in result.detections:
        if d.status == Status.REJECTED:
            continue
        by_frame[d.frame] = by_frame.get(d.frame, 0) + 1

    multibox_frames = [f for f, dets in clip.frames.items() if len(dets) >= 3]
    if not multibox_frames:
        return {"n_multibox_frames": 0}
    resolved = sum(1 for f in multibox_frames if by_frame.get(f, 0) <= class_max)
    return {"n_multibox_frames": len(multibox_frames), "frac_resolved_to_cap": resolved / len(multibox_frames)}


def interpolation_rate(result: PipelineResult) -> float:
    counts = result.by_status()
    total = sum(counts.values())
    return counts.get("interpolated", 0) / total if total else 0.0


# ---------------------------------------------------------------------------
# synthetic dropout injection
# ---------------------------------------------------------------------------

def _contiguous_runs(frames: list[int]) -> list[list[int]]:
    runs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    runs.append(cur)
    return runs


def synthetic_dropout_eval(
    clip: ClipData, cfg: AdapterConfig, k_values: list[int], trials_per_k: int, seed: int = 0
) -> dict:
    rng = random.Random(seed)
    frame_cands = {}
    for fidx, dets in clip.frames.items():
        cands, _ = geometric_reject_frame(dets, cfg)
        frame_cands[fidx] = cands
    tracks = associate(frame_cands, cfg)
    vio = VioSeries(clip.vio)
    surviving, _ = reject_temporal_all(tracks, cfg, vio)
    selected, _ = select_cap(surviving, cfg)

    w = clip.meta.get("width", 1920)
    h = clip.meta.get("height", 1200)

    results_by_k: dict[int, list[dict]] = {k: [] for k in k_values}

    candidate_tracks = [t for t in selected if len(t) >= 20]
    for k in k_values:
        for _ in range(trials_per_k):
            eligible = []
            for t in candidate_tracks:
                for run in _contiguous_runs(t.frames):
                    if len(run) >= k + 10:
                        eligible.append((t, run))
            if not eligible:
                continue
            t, run = rng.choice(eligible)
            start = rng.randint(5, len(run) - k - 5)
            gap_frames = run[start:start + k]

            true_boxes = {f: t.candidates[f].xyxy for f in gap_frames}
            nt = Track(track_id=t.track_id)
            nt.frames = [f for f in t.frames if f not in true_boxes]
            nt.candidates = {f: t.candidates[f] for f in nt.frames}

            dets, _ = interpolate_tracks([nt], cfg, w, h)
            by_frame = {d.frame: d for d in dets}

            ious, dists, n_recovered = [], [], 0
            for f, true_box in true_boxes.items():
                d = by_frame.get(f)
                if d is None or d.status != Status.INTERPOLATED:
                    continue
                n_recovered += 1
                ious.append(iou(true_box, d.xyxy))
                dc, tc = centroid(d.xyxy), centroid(true_box)
                dists.append(((dc[0] - tc[0]) ** 2 + (dc[1] - tc[1]) ** 2) ** 0.5)

            results_by_k[k].append({
                "n_frames": k, "n_recovered": n_recovered,
                "mean_iou": float(np.mean(ious)) if ious else None,
                "mean_center_dist_px": float(np.mean(dists)) if dists else None,
            })

    summary = {}
    for k, trials in results_by_k.items():
        if not trials:
            continue
        recovery_rate = sum(t["n_recovered"] for t in trials) / (k * len(trials))
        ious = [t["mean_iou"] for t in trials if t["mean_iou"] is not None]
        dists = [t["mean_center_dist_px"] for t in trials if t["mean_center_dist_px"] is not None]
        summary[k] = {
            "n_trials": len(trials),
            "recovery_rate": recovery_rate,
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_center_dist_px": float(np.mean(dists)) if dists else None,
        }
    return summary


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def validate_corpus(cfg: AdapterConfig, clip_ids: list[str] | None = None) -> dict:
    clip_ids = clip_ids or list_clip_ids()
    per_clip = {}
    agg_precision_recall: dict[str, list] = {}
    interp_rates = []

    for cid in clip_ids:
        clip = load_clip(cid)
        result = run_pipeline(clip, cfg)

        loose = build_loose_tracks(clip, cfg)
        stable = stable_tracks(loose)
        real_ids = {sid for t in stable for c in t.candidates.values() for sid in c.source_ids}

        ppr = proxy_precision_recall(clip, result, real_ids)
        fcr = frame_coverage_recall(clip, stable, result)
        mb = multibox_resolution(clip, result)
        irate = interpolation_rate(result)
        interp_rates.append(irate)

        for stage, vals in ppr.items():
            agg_precision_recall.setdefault(stage, []).append(vals)

        per_clip[cid] = {
            "status_counts": result.by_status(),
            "proxy_precision_recall": ppr,
            "frame_coverage_recall": fcr,
            "multibox_resolution": mb,
            "interpolation_rate": irate,
        }

    stage_summary = {}
    for stage, rows in agg_precision_recall.items():
        precisions = [r["precision_proxy"] for r in rows if r["precision_proxy"] is not None]
        recalls = [r["recall_proxy"] for r in rows if r["recall_proxy"] is not None]
        stage_summary[stage] = {
            "mean_precision_proxy": float(np.mean(precisions)) if precisions else None,
            "mean_recall_proxy": float(np.mean(recalls)) if recalls else None,
        }

    interp_rates_arr = np.array(interp_rates)
    mean_ir, std_ir = float(interp_rates_arr.mean()), float(interp_rates_arr.std())
    flagged = [
        cid for cid, r in zip(clip_ids, interp_rates)
        if r > mean_ir + 3 * std_ir or r > 0.20
    ]

    return {
        "per_clip": per_clip,
        "stage_precision_recall_summary": stage_summary,
        "interpolation_rate": {
            "mean": mean_ir, "std": std_ir,
            "flagged_clips": flagged,
        },
    }


def main():
    from adapter.calibrate import calibrate

    os.makedirs(OUT_DIR, exist_ok=True)
    calib_path = os.path.join(OUT_DIR, "calibrated_config.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            cfg = AdapterConfig(**json.load(f))
    else:
        dev_ids = ["887c633e_t028", "c6754e88_t019", "ce045be4_t019"]
        cfg, _ = calibrate(dev_clip_ids=dev_ids)

    report = validate_corpus(cfg)

    dropout_clip = load_clip("887c633e_t028")
    k_values = sorted(set([2, 5, 10, cfg.max_dropout_frames, cfg.max_dropout_frames + 10, cfg.max_dropout_frames * 2]))
    report["synthetic_dropout"] = synthetic_dropout_eval(dropout_clip, cfg, k_values, trials_per_k=15)

    with open(os.path.join(OUT_DIR, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)

    print(json.dumps(report["stage_precision_recall_summary"], indent=1))
    print(json.dumps(report["interpolation_rate"], indent=1))
    print(json.dumps(report["synthetic_dropout"], indent=1))
    print("\nwrote", os.path.join(OUT_DIR, "validation_report.json"))


if __name__ == "__main__":
    main()
