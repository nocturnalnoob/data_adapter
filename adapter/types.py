"""Core data types shared across the adapter pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    REPORTED = "reported"
    MERGED = "merged"
    REJECTED = "rejected"
    INTERPOLATED = "interpolated"


class RejectStage(str, Enum):
    DUPLICATE = "duplicate"
    SIZE = "size"
    SHAPE = "shape"
    STEREO_REACH = "stereo_reach"
    DISPLACEMENT = "displacement"
    UNSUPPORTED = "unsupported"
    STATIC = "static"
    CAP = "cap"
    LEAVING_FRAME = "leaving_frame"
    UNBRIDGED_BREAK = "unbridged_break"


@dataclass
class RawDetection:
    """A single detector output box, as read from hand_boxes.json."""

    frame: int
    xyxy: tuple[float, float, float, float]
    cls: int
    handedness: str
    confidence: float
    det_id: int = -1  # assigned on load: unique id within the clip

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2.0

    @property
    def w(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def h(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class Detection:
    """A corrected/annotated detection emitted by the adapter."""

    frame: int
    xyxy: tuple[float, float, float, float]
    status: Status
    confidence: Optional[float] = None
    track_id: Optional[int] = None
    reject_stage: Optional[RejectStage] = None
    source_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "frame": self.frame,
            "xyxy": list(self.xyxy),
            "status": self.status.value,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "reject_stage": self.reject_stage.value if self.reject_stage else None,
            "source_ids": self.source_ids,
        }


@dataclass
class Candidate:
    """A detection still alive in the pipeline (not yet finalized as
    rejected/kept). Carries source_ids so a later merge/interpolation can
    point back to the raw detections it derives from."""

    frame: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    source_ids: list[int] = field(default_factory=list)
    was_merged: bool = False
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2.0


@dataclass
class Track:
    """A per-object track built during association. `candidates` holds only
    *observed* (geometrically-surviving) frames -- gaps are implicit and
    handled later by reject_temporal/interpolate."""

    track_id: int
    frames: list[int] = field(default_factory=list)  # observed frames, ascending
    candidates: dict[int, "Candidate"] = field(default_factory=dict)
    rejected: bool = False
    reject_stage: Optional[RejectStage] = None

    def last_frame(self) -> int:
        return self.frames[-1]

    def first_frame(self) -> int:
        return self.frames[0]

    def last_box(self) -> tuple[float, float, float, float]:
        return self.candidates[self.frames[-1]].xyxy

    def box_at(self, frame: int) -> tuple[float, float, float, float]:
        return self.candidates[frame].xyxy

    def __len__(self) -> int:
        return len(self.frames)
