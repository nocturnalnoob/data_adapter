"""Stages 1-3 (dedup, implausible size, implausible shape) plus the
hand-specific stereo arm's-reach check, which is geometric in nature (needs
only the current frame's stereo pair, no track) and so runs in this stage
per instruction.md SS5 ("no equivalent in the generic adapter").

Order (instruction.md SS3, table + SS39): duplicate merge -> size -> shape,
before anything temporal. The stereo check is hand-specific; it slots in
after shape and before association, since -- like size/shape -- it needs
only a single time instant.
"""
from __future__ import annotations

from typing import Callable, Optional

from adapter.config import AdapterConfig
from adapter.geometry import overlap_ratio, area, aspect_ratio
from adapter.types import RawDetection, Candidate, Detection, Status, RejectStage

ReachFn = Callable[[int, tuple], tuple[bool, float | None, float]]  # (frame, xyxy) -> (within_reach, disp, conf)


def _merge_duplicates(dets: list[RawDetection], cfg: AdapterConfig) -> list[Candidate]:
    """Greedy clustering by overlap_ratio: any pair above threshold is the
    same object. Representative = highest confidence in the cluster."""
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_ratio(dets[i].xyxy, dets[j].xyxy) >= cfg.duplicate_overlap_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    candidates = []
    for members in clusters.values():
        best = max(members, key=lambda i: dets[i].confidence)
        rep = dets[best]
        candidates.append(
            Candidate(
                frame=rep.frame,
                xyxy=rep.xyxy,
                confidence=rep.confidence,
                source_ids=[dets[i].det_id for i in members],
                was_merged=len(members) > 1,
            )
        )
    return candidates


def geometric_reject_frame(
    dets: list[RawDetection],
    cfg: AdapterConfig,
    reach_fn: Optional[ReachFn] = None,
) -> tuple[list[Candidate], list[Detection]]:
    """Returns (surviving candidates, finalized rejections) for one frame."""
    if not dets:
        return [], []

    finalized: list[Detection] = []

    # stage 1: duplicate merge
    candidates = _merge_duplicates(dets, cfg)

    # stage 2: implausible size
    survivors = []
    for c in candidates:
        a = area(c.xyxy)
        if a < cfg.min_area_px2 or a > cfg.max_area_px2:
            finalized.append(
                Detection(
                    frame=c.frame, xyxy=c.xyxy, status=Status.REJECTED,
                    confidence=c.confidence, reject_stage=RejectStage.SIZE,
                    source_ids=c.source_ids,
                )
            )
        else:
            survivors.append(c)
    candidates = survivors

    # stage 3: implausible shape
    survivors = []
    for c in candidates:
        if aspect_ratio(c.xyxy) > cfg.max_aspect_ratio:
            finalized.append(
                Detection(
                    frame=c.frame, xyxy=c.xyxy, status=Status.REJECTED,
                    confidence=c.confidence, reject_stage=RejectStage.SHAPE,
                    source_ids=c.source_ids,
                )
            )
        else:
            survivors.append(c)
    candidates = survivors

    # hand-specific: stereo arm's-reach (geometric -- single time instant)
    if cfg.stereo_enabled and reach_fn is not None:
        survivors = []
        for c in candidates:
            within_reach, disp, conf = reach_fn(c.frame, c.xyxy)
            if not within_reach:
                finalized.append(
                    Detection(
                        frame=c.frame, xyxy=c.xyxy, status=Status.REJECTED,
                        confidence=c.confidence, reject_stage=RejectStage.STEREO_REACH,
                        source_ids=c.source_ids,
                    )
                )
            else:
                survivors.append(c)
        candidates = survivors

    return candidates, finalized
