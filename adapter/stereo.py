"""Per-box stereo disparity for the hand adapter's arm's-reach discriminator.

No intrinsics/baseline are provided anywhere in the corpus (checked: no
"baseline"/"focal"/"intrinsic" field in any meta.json), and a quick ORB
feature-match check between video_left/video_right frame 0 of one clip gave
median |dy| ~= 2px on a 1200px-tall frame with a ~45px mean horizontal
disparity -- i.e. the pair is rectified. That's exactly enough to do
epipolar-line (horizontal-only) template matching and get *relative*
disparity, without needing metric depth: the adapter only needs a binary
"within arm's reach" classifier (see instruction.md SS5/SS6), and relative
disparity is monotonic with distance for a fixed rectified rig.
"""
from __future__ import annotations

import numpy as np
import cv2

from adapter.types import RawDetection
from adapter.config import AdapterConfig

Box = tuple[float, float, float, float]


def box_disparity(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    box: Box,
    max_disp: int = 220,
    min_disp: int = 0,
    row_pad: int = 6,
    center_crop: float = 0.7,
) -> tuple[float | None, float]:
    """Estimate horizontal disparity (left_x - right_x) for one box via
    template matching along the epipolar line. Returns (disparity_px or
    None, match_confidence). center_crop shrinks the template to the box's
    central region to reduce background bleed at the box edges.
    """
    H, W = gray_left.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, 0.0

    # shrink to central region of the box
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(x1 + bw * (1 - center_crop) / 2)
    cx2 = int(x2 - bw * (1 - center_crop) / 2)
    cy1 = int(y1 + bh * (1 - center_crop) / 2)
    cy2 = int(y2 - bh * (1 - center_crop) / 2)
    patch = gray_left[cy1:cy2, cx1:cx2]
    if patch.size == 0 or patch.std() < 4.0:
        return None, 0.0  # low-texture patch, matching would be unreliable

    row0, row1 = max(0, cy1 - row_pad), min(H, cy2 + row_pad)
    col_start = max(0, cx1 - max_disp)
    col_end = min(W, cx2 - min_disp)
    if col_end <= col_start:
        return None, 0.0
    search = gray_right[row0:row1, col_start:col_end]
    if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
        return None, 0.0

    res = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    matched_x1_right = col_start + max_loc[0]
    disparity = float(cx1 - matched_x1_right)
    return disparity, float(max_val)


def is_within_reach(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    box: Box,
    cfg: AdapterConfig,
) -> tuple[bool, float | None, float]:
    """True (default) unless the stereo signal confidently says "too far".
    Returns (within_reach, disparity, match_confidence)."""
    if not cfg.stereo_enabled or cfg.reach_disparity_threshold_px is None:
        return True, None, 0.0
    disp, conf = box_disparity(gray_left, gray_right, box)
    if disp is None or conf < cfg.stereo_match_min_confidence:
        return True, disp, conf  # unreliable signal -> don't reject on it
    return disp >= cfg.reach_disparity_threshold_px, disp, conf


class VideoPairReader:
    """Cheap sequential-ish reader for a left/right video pair; grabs a
    specific frame index by seeking (adequate for the sparse frames the
    stereo check runs on -- multi-box frames are ~5% of the corpus)."""

    def __init__(self, left_path: str, right_path: str):
        self.capL = cv2.VideoCapture(left_path)
        self.capR = cv2.VideoCapture(right_path)
        self._last_idx = -1

    def gray_pair(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        # h264 .set(POS_FRAMES) seeks to the nearest keyframe then decodes
        # forward anyway, so for the modest forward skips this reader is
        # actually used for (ascending, sparse sampling), sequential
        # read-and-discard is at least as fast and more reliable. Only seek
        # for backward jumps or the very first read.
        gap = frame_idx - (self._last_idx + 1)
        if gap < 0 or self._last_idx < 0:
            self.capL.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.capR.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        else:
            for _ in range(gap):
                self.capL.read()
                self.capR.read()
        okL, fL = self.capL.read()
        okR, fR = self.capR.read()
        self._last_idx = frame_idx
        if not okL or not okR:
            return None
        return (
            cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY),
        )

    def release(self):
        self.capL.release()
        self.capR.release()
