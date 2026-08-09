"""Frame-indexed access to a clip's VIO pose, for the static-detection rule.

vio_pose.json is native-rate (~30Hz, same as video) but its length doesn't
always exactly match hand_boxes' video_frame_count (off by a couple of
frames in samples checked) -- index by frame number, clipping to the
array's range, rather than assuming exact 1:1 length equality.

yaw/pitch/roll are in degrees (checked directly: values run e.g. -138..-66
for one clip, i.e. plainly not radians); wrap diffs to (-180, 180] before
using them as an angular-rate signal.
"""
from __future__ import annotations

import numpy as np


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


class VioSeries:
    def __init__(self, vio: dict):
        self.n = len(vio["t"])
        self.speed = np.array(vio["speed"], dtype=float)  # m/s, translational
        yaw = np.array(vio["yaw"], dtype=float)
        pitch = np.array(vio["pitch"], dtype=float)
        roll = np.array(vio["roll"], dtype=float)
        dyaw = _wrap_deg(np.diff(yaw, prepend=yaw[0]))
        dpitch = _wrap_deg(np.diff(pitch, prepend=pitch[0]))
        droll = _wrap_deg(np.diff(roll, prepend=roll[0]))
        self.ang_rate_deg = np.sqrt(dyaw**2 + dpitch**2 + droll**2)  # deg/frame

    def _clip(self, frame: int) -> int:
        return max(0, min(self.n - 1, frame))

    def speed_at(self, frame: int) -> float:
        return float(self.speed[self._clip(frame)])

    def ang_rate_at(self, frame: int) -> float:
        return float(self.ang_rate_deg[self._clip(frame)])

    def is_camera_moving(self, frame: int, speed_thresh: float, ang_thresh_deg: float) -> bool:
        return self.speed_at(frame) > speed_thresh or self.ang_rate_at(frame) > ang_thresh_deg
