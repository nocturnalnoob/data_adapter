"""Orchestrates the full adapter: geometric reject -> associate -> temporal
reject -> select-to-cap -> interpolate (instruction.md SS39's mandated
order). Every input raw detection ends up accounted for in exactly one
finalized Detection's source_ids (or contributes nothing, if the frame had
no detections) -- see `check_completeness`.
"""
from __future__ import annotations

from dataclasses import dataclass

from adapter.config import AdapterConfig
from adapter.io_clip import ClipData
from adapter.reject_geometric import geometric_reject_frame
from adapter.association import associate
from adapter.reject_temporal import reject_temporal_all
from adapter.selection import select_cap
from adapter.interpolate import interpolate_tracks
from adapter.vio_utils import VioSeries
from adapter.stereo import VideoPairReader, is_within_reach
from adapter.types import Detection


@dataclass
class PipelineResult:
    cid: str
    detections: list[Detection]
    # raw det_id -> set of surviving det_ids after each stage boundary, for
    # validate.py's proxy precision/recall (see adapter/validate.py).
    stage_survivor_ids: dict[str, set[int]] | None = None

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.detections:
            out[d.status.value] = out.get(d.status.value, 0) + 1
        return out

    def to_json(self) -> dict:
        return {"cid": self.cid, "detections": [d.to_json() for d in self.detections]}


def run_pipeline(clip: ClipData, cfg: AdapterConfig, use_stereo: bool | None = None) -> PipelineResult:
    use_stereo = cfg.stereo_enabled if use_stereo is None else use_stereo

    reach_fn = None
    video_reader = None
    if use_stereo and cfg.reach_disparity_threshold_px is not None:
        video_reader = VideoPairReader(clip.video_left_path(), clip.video_right_path())

        def reach_fn(frame: int, xyxy):
            pair = video_reader.gray_pair(frame)
            if pair is None:
                return True, None, 0.0
            gl, gr = pair
            return is_within_reach(gl, gr, xyxy, cfg)

    finalized: list[Detection] = []
    frame_cands = {}
    try:
        for fidx in sorted(clip.frames.keys()):
            dets = clip.frames[fidx]
            cands, rej = geometric_reject_frame(dets, cfg, reach_fn=reach_fn)
            finalized.extend(rej)
            frame_cands[fidx] = cands
    finally:
        if video_reader is not None:
            video_reader.release()

    def _ids(cands_iterable) -> set[int]:
        s: set[int] = set()
        for c in cands_iterable:
            s.update(c.source_ids)
        return s

    geo_ids = _ids(c for cands in frame_cands.values() for c in cands)

    tracks = associate(frame_cands, cfg)

    vio = VioSeries(clip.vio)
    surviving, temporal_rej = reject_temporal_all(tracks, cfg, vio)
    finalized.extend(temporal_rej)
    temporal_ids = _ids(c for t in surviving for c in t.candidates.values())

    selected, cap_rej = select_cap(surviving, cfg)
    finalized.extend(cap_rej)
    cap_ids = _ids(c for t in selected for c in t.candidates.values())

    w = clip.meta.get("width", 1920)
    h = clip.meta.get("height", 1200)
    interp_dets, _ = interpolate_tracks(selected, cfg, w, h)
    finalized.extend(interp_dets)

    finalized.sort(key=lambda d: d.frame)
    stage_ids = {"geometric": geo_ids, "temporal": temporal_ids, "cap": cap_ids}
    return PipelineResult(cid=clip.cid, detections=finalized, stage_survivor_ids=stage_ids)


def check_completeness(clip: ClipData, result: PipelineResult) -> dict:
    """Every raw det_id must appear in exactly one finalized detection's
    source_ids (interpolated detections carry none, by construction)."""
    total_raw = sum(len(dets) for dets in clip.frames.values())
    seen: dict[int, int] = {}
    for d in result.detections:
        for sid in d.source_ids:
            seen[sid] = seen.get(sid, 0) + 1

    accounted = len(seen)
    duplicated = [sid for sid, n in seen.items() if n > 1]
    missing = total_raw - accounted
    return {
        "total_raw": total_raw,
        "accounted": accounted,
        "missing": missing,
        "duplicated_ids": duplicated,
        "ok": missing == 0 and not duplicated,
    }
