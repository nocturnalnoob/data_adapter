"""Associates per-frame geometric survivors into per-object tracks.

Matching uses only position/box geometry + a constant-velocity motion
prediction -- never the detector's reported handedness label, which
instruction.md SS5 flags as unreliable once hands cross.

A track can bridge a gap of missing observations (the detector dropped the
object for a few frames): the predicted position is extrapolated forward by
the track's last known velocity, and the search gate widens with elapsed
frames. This is what lets reject_temporal's "unsupported detection" rule and
interpolate's gap-fill later operate on a single continuous track identity
rather than fragments. Bridging is capped by cfg.max_dropout_frames * 2 --
a candidate that shows up only after a longer gap starts a new track, per
the "hand re-enters after a long absence" edge case (instruction.md SS6).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from adapter.config import AdapterConfig
from adapter.types import Candidate, Track
from adapter.geometry import iou, centroid

GIVE_UP_MULTIPLIER = 2.0  # track removed from the active pool beyond this * max_dropout_frames


def _predicted_position(track: Track, frame: int) -> tuple[float, float]:
    frames = track.frames
    last_frame = frames[-1]
    last_c = centroid(track.candidates[last_frame].xyxy)
    if len(frames) < 2:
        return last_c
    prev_frame = frames[-2]
    prev_c = centroid(track.candidates[prev_frame].xyxy)
    dt = last_frame - prev_frame
    if dt <= 0:
        return last_c
    vx = (last_c[0] - prev_c[0]) / dt
    vy = (last_c[1] - prev_c[1]) / dt
    elapsed = frame - last_frame
    return (last_c[0] + vx * elapsed, last_c[1] + vy * elapsed)


def _gate(cfg: AdapterConfig, elapsed: int) -> float:
    return cfg.resume_gate_px + cfg.max_speed_px_per_frame * elapsed


class Associator:
    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 0

    def _new_track(self, frame: int, cand: Candidate) -> Track:
        t = Track(track_id=self._next_id)
        self._next_id += 1
        t.frames.append(frame)
        t.candidates[frame] = cand
        cand.track_id = t.track_id
        return t

    def step(self, frame: int, candidates: list[Candidate]) -> None:
        cfg = self.cfg
        # drop tracks that have been silent too long
        give_up = int(cfg.max_dropout_frames * GIVE_UP_MULTIPLIER) + 1
        still_active = []
        for t in self.active:
            if frame - t.last_frame() > give_up:
                self.finished.append(t)
            else:
                still_active.append(t)
        self.active = still_active

        if not candidates:
            return

        if not self.active:
            for c in candidates:
                self.active.append(self._new_track(frame, c))
            return

        n_tracks, n_cands = len(self.active), len(candidates)
        cost = np.full((n_tracks, n_cands), 1e6)
        for ti, t in enumerate(self.active):
            elapsed = frame - t.last_frame()
            if elapsed <= 0:
                continue  # already has a detection this frame (shouldn't happen)
            pred = _predicted_position(t, frame)
            gate = _gate(cfg, elapsed)
            last_box = t.last_box()
            for ci, c in enumerate(candidates):
                cx, cy = centroid(c.xyxy)
                dist = ((cx - pred[0]) ** 2 + (cy - pred[1]) ** 2) ** 0.5
                if dist > gate:
                    continue
                iou_score = iou(last_box, c.xyxy) if elapsed <= 2 else 0.0
                cost[ti, ci] = dist - cfg.association_iou_weight * iou_score * gate

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks, matched_cands = set(), set()
        for ti, ci in zip(row_ind, col_ind):
            if cost[ti, ci] >= 1e6:
                continue
            t = self.active[ti]
            c = candidates[ci]
            t.frames.append(frame)
            t.candidates[frame] = c
            c.track_id = t.track_id
            matched_tracks.add(ti)
            matched_cands.add(ci)

        for ci, c in enumerate(candidates):
            if ci not in matched_cands:
                self.active.append(self._new_track(frame, c))

    def finalize(self) -> list[Track]:
        self.finished.extend(self.active)
        self.active = []
        return self.finished


def associate(frame_candidates: dict[int, list[Candidate]], cfg: AdapterConfig) -> list[Track]:
    assoc = Associator(cfg)
    for frame in sorted(frame_candidates.keys()):
        assoc.step(frame, frame_candidates[frame])
    return assoc.finalize()
