"""Stages 4-6 (implausible displacement, unsupported detection, static
detection). These need tracks, so they run after association -- per
instruction.md SS39, "temporal rules need tracks, which only exist after the
association step."
"""
from __future__ import annotations

import numpy as np

from adapter.config import AdapterConfig
from adapter.geometry import centroid
from adapter.types import Track, Detection, Status, RejectStage
from adapter.vio_utils import VioSeries


def reject_displacement(track: Track, cfg: AdapterConfig) -> tuple[Track, list[Detection]]:
    """Walks the track's observed frames; a jump whose implied speed
    exceeds max_speed_px_per_frame breaks continuity with everything after
    it -- those later boxes are rejected as displacement failures rather
    than trusted as the same object (association's own gate should mostly
    prevent this; this is the defensive, spec-mandated pass over the
    resulting tracks)."""
    if len(track) <= 1:
        return track, []

    rejected: list[Detection] = []
    kept_frames = [track.frames[0]]
    last_frame = track.frames[0]
    last_c = centroid(track.candidates[last_frame].xyxy)

    for f in track.frames[1:]:
        c = track.candidates[f]
        cx, cy = centroid(c.xyxy)
        elapsed = f - last_frame
        dist = ((cx - last_c[0]) ** 2 + (cy - last_c[1]) ** 2) ** 0.5
        speed = dist / max(1, elapsed)
        if speed > cfg.max_speed_px_per_frame:
            rejected.append(
                Detection(
                    frame=f, xyxy=c.xyxy, status=Status.REJECTED,
                    confidence=c.confidence, reject_stage=RejectStage.DISPLACEMENT,
                    source_ids=c.source_ids,
                )
            )
            continue  # don't advance last_* -- keep comparing against the last valid box
        kept_frames.append(f)
        last_frame, last_c = f, (cx, cy)

    new_track = Track(track_id=track.track_id)
    new_track.frames = kept_frames
    new_track.candidates = {f: track.candidates[f] for f in kept_frames}
    return new_track, rejected


def reject_unsupported(track: Track, cfg: AdapterConfig) -> tuple[bool, list[Detection]]:
    """A track with too few observed frames and no continuation is a
    flicker, not an object. Returns (is_supported, rejected_detections)."""
    if len(track) >= cfg.min_track_len_supported:
        return True, []
    rejected = [
        Detection(
            frame=f, xyxy=track.candidates[f].xyxy, status=Status.REJECTED,
            confidence=track.candidates[f].confidence, reject_stage=RejectStage.UNSUPPORTED,
            source_ids=track.candidates[f].source_ids,
        )
        for f in track.frames
    ]
    return False, rejected


def reject_static(track: Track, cfg: AdapterConfig, vio: VioSeries) -> tuple[Track, list[Detection]]:
    """Flags sustained windows where the box barely moves in image space
    while the camera (per VIO) is clearly moving -- indicating scene
    structure rather than a hand. See vio_utils.VioSeries for the ego-motion
    signal and config.py for the "camera moving" thresholds.

    Simplification, documented: this uses box-motion magnitude gated on
    camera motion, not a full parallax/depth-aware reprojection (no
    intrinsics are available in this corpus -- see adapter/stereo.py). It
    will under-flag a background object during pure camera translation with
    no rotation (rare for a head-mounted rig) and, per instruction.md SS6's
    last row, is one of the properties flagged as needing labelled data to
    validate properly.
    """
    if len(track) < 2:
        return track, []

    frames = track.frames
    boxes = [track.candidates[f].xyxy for f in frames]
    centroids = [centroid(b) for b in boxes]

    W = cfg.static_window_frames
    static_frames: set[int] = set()

    for i in range(len(frames)):
        lo = max(0, i - W // 2)
        hi = min(len(frames), i + W // 2 + 1)
        if hi - lo < max(3, W // 2):
            continue
        window_frames = frames[lo:hi]
        # require reasonably dense observation in the window
        span = window_frames[-1] - window_frames[0]
        if span <= 0 or (hi - lo) / (span + 1) < 0.5:
            continue

        speeds = []
        cam_moving_count = 0
        for j in range(lo + 1, hi):
            f0, f1 = frames[j - 1], frames[j]
            elapsed = f1 - f0
            dx = centroids[j][0] - centroids[j - 1][0]
            dy = centroids[j][1] - centroids[j - 1][1]
            speeds.append(((dx**2 + dy**2) ** 0.5) / max(1, elapsed))
            if vio.is_camera_moving(f1, cfg.camera_speed_moving_thresh, cfg.camera_ang_moving_thresh_deg):
                cam_moving_count += 1

        if not speeds:
            continue
        mean_speed = float(np.mean(speeds))
        camera_moving_frac = cam_moving_count / len(speeds)

        if mean_speed < cfg.static_motion_px_threshold and camera_moving_frac > 0.6:
            static_frames.add(frames[i])

    if not static_frames:
        return track, []

    rejected = [
        Detection(
            frame=f, xyxy=track.candidates[f].xyxy, status=Status.REJECTED,
            confidence=track.candidates[f].confidence, reject_stage=RejectStage.STATIC,
            source_ids=track.candidates[f].source_ids,
        )
        for f in static_frames
    ]
    kept_frames = [f for f in frames if f not in static_frames]
    new_track = Track(track_id=track.track_id)
    new_track.frames = kept_frames
    new_track.candidates = {f: track.candidates[f] for f in kept_frames}
    return new_track, rejected


def reject_temporal_all(
    tracks: list[Track], cfg: AdapterConfig, vio: VioSeries
) -> tuple[list[Track], list[Detection]]:
    """Runs stages 4 -> 5 -> 6 (displacement gates continuity first, then
    unsupported drops flicker-length tracks, then static prunes scene
    structure from what remains) and returns the surviving tracks plus all
    finalized rejections."""
    finalized: list[Detection] = []
    surviving: list[Track] = []

    for t in tracks:
        t, rej = reject_displacement(t, cfg)
        finalized.extend(rej)
        if len(t) == 0:
            continue

        supported, rej = reject_unsupported(t, cfg)
        if not supported:
            finalized.extend(rej)
            continue

        t, rej = reject_static(t, cfg, vio)
        finalized.extend(rej)
        if len(t) == 0:
            continue

        surviving.append(t)

    return surviving, finalized
