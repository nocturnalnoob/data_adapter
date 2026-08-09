"""Recovers false negatives: extrapolates each track across a break and
fills the gap when the track resumes near its predicted position. Before
filling, tests whether the track was leaving the frame -- instruction.md
SS4's governing constraint: "an object leaving the frame is not a false
negative. Filling that gap invents an object in empty footage."

This module is the authority on final track identity, independent of what
association's (looser, purely positional) gate decided: any break that
either (a) looks like a frame-exit, or (b) doesn't resume near the
predicted position within the calibrated max dropout, ends the track there
and starts a fresh track id for whatever comes after -- matching the
"re-enters after a long absence -> start a new track" edge case.
"""
from __future__ import annotations

from adapter.config import AdapterConfig
from adapter.geometry import centroid
from adapter.types import Track, Detection, Status, RejectStage

BORDERS = ("left", "top", "right", "bottom")


def _velocity(track: Track, at_frame: int) -> tuple[float, float]:
    frames = track.frames
    idx = frames.index(at_frame)
    if idx == 0:
        return (0.0, 0.0)
    f0, f1 = frames[idx - 1], frames[idx]
    c0 = centroid(track.candidates[f0].xyxy)
    c1 = centroid(track.candidates[f1].xyxy)
    dt = max(1, f1 - f0)
    return ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)


def is_leaving(
    box: tuple[float, float, float, float],
    velocity: tuple[float, float],
    frame_w: float,
    frame_h: float,
    cfg: AdapterConfig,
) -> bool:
    x1, y1, x2, y2 = box
    vx, vy = velocity
    dist_to_border = {
        "left": x1,
        "top": y1,
        "right": frame_w - x2,
        "bottom": frame_h - y2,
    }
    outward_speed = {
        "left": -vx,
        "top": -vy,
        "right": vx,
        "bottom": vy,
    }
    for b in BORDERS:
        weight = cfg.border_weights.get(b, 1.0)
        margin = cfg.border_margin_px * weight
        if dist_to_border[b] <= margin and outward_speed[b] >= cfg.outward_velocity_min_px_per_frame:
            return True
    return False


def _lerp_box(b0, b1, t: float) -> tuple[float, float, float, float]:
    return tuple(b0[i] + (b1[i] - b0[i]) * t for i in range(4))


def interpolate_tracks(
    tracks: list[Track], cfg: AdapterConfig, frame_w: float, frame_h: float
) -> tuple[list[Detection], int]:
    """Returns (all finalized detections for these tracks, next_free_track_id
    high-water mark used for id renumbering on splits)."""
    detections: list[Detection] = []
    next_id = max((t.track_id for t in tracks), default=-1) + 1

    for t in tracks:
        frames = t.frames
        seg_frames = [frames[0]]

        def flush_segment(seg_frames, track_id):
            for f in seg_frames:
                c = t.candidates[f]
                detections.append(
                    Detection(
                        frame=f, xyxy=c.xyxy,
                        status=Status.MERGED if c.was_merged else Status.REPORTED,
                        confidence=c.confidence, track_id=track_id,
                        source_ids=c.source_ids,
                    )
                )

        current_id = t.track_id
        for i in range(1, len(frames)):
            f0, f1 = frames[i - 1], frames[i]
            gap = f1 - f0 - 1

            if gap == 0:
                seg_frames.append(f1)
                continue

            box0 = t.candidates[f0].xyxy
            vel0 = _velocity(t, f0)
            leaving = is_leaving(box0, vel0, frame_w, frame_h, cfg)

            box1 = t.candidates[f1].xyxy
            pred_x = centroid(box0)[0] + vel0[0] * (gap + 1)
            pred_y = centroid(box0)[1] + vel0[1] * (gap + 1)
            actual = centroid(box1)
            resume_dist = ((actual[0] - pred_x) ** 2 + (actual[1] - pred_y) ** 2) ** 0.5
            resumes_near_prediction = resume_dist <= cfg.resume_gate_px + cfg.max_speed_px_per_frame * gap

            can_fill = (not leaving) and gap <= cfg.max_dropout_frames and resumes_near_prediction

            if can_fill:
                for k in range(1, gap + 1):
                    tf = f0 + k
                    frac = k / (gap + 1)
                    ibox = _lerp_box(box0, box1, frac)
                    detections.append(
                        Detection(
                            frame=tf, xyxy=ibox, status=Status.INTERPOLATED,
                            confidence=None, track_id=current_id,
                        )
                    )
                seg_frames.append(f1)
            else:
                # end this track here; f1 (and onward) starts a fresh identity
                flush_segment(seg_frames, current_id)
                current_id = next_id
                next_id += 1
                seg_frames = [f1]

        flush_segment(seg_frames, current_id)

    return detections, next_id
