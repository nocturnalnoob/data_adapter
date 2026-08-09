"""Loaders for one clip bundle (hand_boxes.json, frame_ts.json, vio_pose.json, meta.json)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from adapter.types import RawDetection

DOWNLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads")


def list_clip_ids(downloads_dir: str = DOWNLOADS_DIR) -> list[str]:
    return sorted(
        d for d in os.listdir(downloads_dir)
        if os.path.isdir(os.path.join(downloads_dir, d))
    )


@dataclass
class ClipData:
    cid: str
    dir: str
    meta: dict
    frame_ts: dict  # frame_index(str) -> unix_ns
    frames: dict[int, list[RawDetection]]  # frame_index -> detections
    frame_count: int
    vio: dict

    def video_left_path(self) -> str:
        return os.path.join(self.dir, "video_left.mp4")

    def video_right_path(self) -> str:
        return os.path.join(self.dir, "video_right.mp4")


def load_clip(cid: str, downloads_dir: str = DOWNLOADS_DIR) -> ClipData:
    clip_dir = os.path.join(downloads_dir, cid)

    with open(os.path.join(clip_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(clip_dir, "frame_ts.json")) as f:
        frame_ts_doc = json.load(f)
    with open(os.path.join(clip_dir, "hand_boxes.json")) as f:
        hb_doc = json.load(f)
    with open(os.path.join(clip_dir, "vio_pose.json")) as f:
        vio = json.load(f)

    frames: dict[int, list[RawDetection]] = {}
    det_id = 0
    for frame_entry in hb_doc["frames"]:
        fidx = frame_entry["frame"]
        dets = []
        for d in frame_entry["detections"]:
            dets.append(
                RawDetection(
                    frame=fidx,
                    xyxy=tuple(d["xyxy"]),
                    cls=d["class"],
                    handedness=d.get("handedness", ""),
                    confidence=d["confidence"],
                    det_id=det_id,
                )
            )
            det_id += 1
        frames[fidx] = dets

    return ClipData(
        cid=cid,
        dir=clip_dir,
        meta=meta,
        frame_ts=frame_ts_doc["frame_ts"],
        frames=frames,
        frame_count=hb_doc["video_frame_count"],
        vio=vio,
    )
