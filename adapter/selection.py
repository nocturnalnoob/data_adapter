"""Caps per-frame instance count to cfg.class_max, run after association and
temporal rejection so ranking uses each track's *post-rejection* support
(length, mean confidence) rather than a single frame's score --
instruction.md SS3: "ranking candidates by the track supporting them rather
than by their score in one frame."

This is deliberately a per-frame decision, not "keep only the top class_max
tracks globally": tracks that never overlap in time (e.g. a hand leaves and
a different track starts later) don't compete for the same slot.
"""
from __future__ import annotations

import numpy as np

from adapter.config import AdapterConfig
from adapter.types import Track, Detection, Status, RejectStage


def _track_support(track: Track) -> tuple[int, float]:
    confs = [c.confidence for c in track.candidates.values()]
    return (len(track), float(np.mean(confs)) if confs else 0.0)


def rank_tracks_per_frame(tracks: list[Track]) -> dict[int, list[int]]:
    """frame -> track_ids active in that frame, best-support-first. Shared by
    selection (cap enforcement) and calibrate.py (which uses the top-2 as a
    label-free proxy for "the wearer's two hands" when mining statistics)."""
    support = {t.track_id: _track_support(t) for t in tracks}
    frame_members: dict[int, list[int]] = {}
    for t in tracks:
        for f in t.frames:
            frame_members.setdefault(f, []).append(t.track_id)
    return {
        f: sorted(ids, key=lambda tid: support[tid], reverse=True)
        for f, ids in frame_members.items()
    }


def select_cap(tracks: list[Track], cfg: AdapterConfig) -> tuple[list[Track], list[Detection]]:
    if cfg.class_max is None or not tracks:
        return tracks, []

    by_track = {t.track_id: t for t in tracks}
    ranked_per_frame = rank_tracks_per_frame(tracks)

    rejected_frame_of_track: dict[int, set[int]] = {t.track_id: set() for t in tracks}
    finalized: list[Detection] = []

    for f, ranked in ranked_per_frame.items():
        if len(ranked) <= cfg.class_max:
            continue
        losers = ranked[cfg.class_max:]
        for tid in losers:
            rejected_frame_of_track[tid].add(f)
            c = by_track[tid].candidates[f]
            finalized.append(
                Detection(
                    frame=f, xyxy=c.xyxy, status=Status.REJECTED,
                    confidence=c.confidence, reject_stage=RejectStage.CAP,
                    source_ids=c.source_ids,
                )
            )

    surviving: list[Track] = []
    for t in tracks:
        drop = rejected_frame_of_track[t.track_id]
        if not drop:
            surviving.append(t)
            continue
        kept_frames = [f for f in t.frames if f not in drop]
        if not kept_frames:
            continue
        nt = Track(track_id=t.track_id)
        nt.frames = kept_frames
        nt.candidates = {f: t.candidates[f] for f in kept_frames}
        surviving.append(nt)

    return surviving, finalized
