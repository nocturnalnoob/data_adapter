"""All adapter thresholds live here as data. Defaults are placeholders;
adapter/calibrate.py derives the real values from the corpus and can dump
a populated AdapterConfig back out (see calibrate.py:CalibratedParams).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class AdapterConfig:
    class_name: str = "generic"

    # class cap: how many instances of this class may exist in one frame
    # after selection. None = uncapped.
    class_max: int | None = None
    # how many candidates to request from the detector above class_max
    # before selection (association ranks by track support, not score).
    candidate_pool_width: int = 4

    # --- stage 1: duplicate merge (geometric) ---
    duplicate_overlap_threshold: float = 0.6  # overlap_ratio (intersection / smaller box)

    # --- stage 2: implausible size (geometric) ---
    # box area, in pixels^2, plausible range at working distance
    min_area_px2: float = 400.0
    max_area_px2: float = 400_000.0

    # --- stage 3: implausible shape (geometric) ---
    max_aspect_ratio: float = 3.0  # long/short side; equant objects: keep tight

    # --- hand-specific: stereo arm's-reach (geometric, hands only) ---
    stereo_enabled: bool = False
    # disparity (px, left_x - right_x) below this => too far to be the wearer's hand.
    # None disables rejection (stereo signal not yet calibrated).
    reach_disparity_threshold_px: float | None = None
    # matchTemplate NCC score below this => unreliable match, skip the check
    # rather than risk a false reject (e.g. low-texture / motion-blurred patch).
    stereo_match_min_confidence: float = 0.5

    # --- association ---
    max_speed_px_per_frame: float = 250.0  # gates both association and stage-4 rejection
    association_iou_weight: float = 0.5

    # --- stage 4: implausible displacement (temporal) ---
    # reuses max_speed_px_per_frame

    # --- stage 5: unsupported detection (temporal) ---
    min_track_len_supported: int = 3  # tracks shorter than this, with no continuation, are dropped

    # --- stage 6: static detection (temporal) ---
    static_window_frames: int = 15
    static_motion_px_threshold: float = 3.0  # px of image motion considered "not moving"
    static_correlation_threshold: float = 0.2  # |corr| below this while camera moves -> static
    # "camera is moving" gate for the static rule: VIO translational speed
    # (m/s) or angular rate (deg/frame) above either -> treat ego-motion as
    # active this frame. Coarse defaults from inspecting one clip's VIO
    # distribution (median speed ~0.046 m/s); not a primary calibration
    # target -- see adapter/calibrate.py notes.
    camera_speed_moving_thresh: float = 0.02
    camera_ang_moving_thresh_deg: float = 0.15

    # --- interpolation ---
    max_dropout_frames: int = 10  # calibrated from dropout-length distribution
    resume_gate_px: float = 60.0  # predicted-vs-actual position gate to accept a resume

    # --- leaving-the-frame test ---
    border_margin_px: float = 40.0
    border_weights: dict = field(
        default_factory=lambda: {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0}
    )
    outward_velocity_min_px_per_frame: float = 2.0

    def to_json(self) -> dict:
        return asdict(self)


def hands_config() -> AdapterConfig:
    return AdapterConfig(
        class_name="hands",
        class_max=2,
        candidate_pool_width=4,
        duplicate_overlap_threshold=0.6,
        max_aspect_ratio=2.2,  # hands are roughly equant
        stereo_enabled=True,
        association_iou_weight=0.5,
        min_track_len_supported=3,
        static_window_frames=15,
        border_margin_px=50.0,
        border_weights={"top": 0.6, "bottom": 1.6, "left": 1.0, "right": 1.0},
        outward_velocity_min_px_per_frame=2.0,
    )
