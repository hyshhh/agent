"""关键帧提取器 — 从人体区域提取连续关键帧"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from models.schemas import PersonDetection, BoundingBox
from utils.image_utils import pad_bbox, crop_region, encode_image_to_base64, compute_adaptive_padding
from utils.logger import get_logger

logger = get_logger()


class FrameExtractor:
    """
    对检测到的人体区域提取关键帧。

    处理流程：
    1. 对人体框加 padding 扩展
    2. 从原始帧中裁剪扩展区域
    3. 按间隔提取关键帧
    4. 编码为 base64 供千问模型使用
    """

    def __init__(
        self,
        padding_ratio: float = 0.15,
        keyframe_interval: int = 3,
        keyframe_count: int = 5,
        min_region_size: int = 32,
        adaptive_padding: bool = True,
        pixel_threshold: float = 10000.0,
    ):
        """
        Args:
            padding_ratio: 正常扩展比例（大目标使用，小目标时作为下限）
            keyframe_interval: 每隔几帧提取一帧
            keyframe_count: 最大关键帧数量
            min_region_size: 最小有效区域像素
            adaptive_padding: 是否启用自适应 padding
            pixel_threshold: 最小裁剪面积阈值（像素²）。
                             当检测框面积小于此值时，padding 自动放大
                             使裁剪区域达到该面积。提供此接口供用户微调。
        """
        self.padding_ratio = padding_ratio
        self.keyframe_interval = keyframe_interval
        self.keyframe_count = keyframe_count
        self.min_region_size = min_region_size
        self.adaptive_padding = adaptive_padding
        self.pixel_threshold = pixel_threshold

    def _get_padding(self, bbox_w: float, bbox_h: float, frame_w: int, frame_h: int) -> float:
        """根据配置返回 padding 比例（固定或自适应）"""
        if self.adaptive_padding:
            return compute_adaptive_padding(
                bbox_w, bbox_h, frame_w, frame_h,
                self.padding_ratio, self.pixel_threshold,
            )
        return self.padding_ratio

    def extract_from_detections(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
    ) -> list[dict]:
        """
        从单帧中为每个检测到的人体提取裁剪区域。

        Args:
            frame: 原始帧（BGR）
            detections: 人体检测结果列表

        Returns:
            每个检测对象的裁剪信息列表：
            [{
                "detection": PersonDetection,
                "padded_bbox": (x1, y1, x2, y2),
                "crop_b64": "base64字符串",
                "crop_image": np.ndarray,  # 供显示用
            }, ...]
        """
        h, w = frame.shape[:2]
        results = []

        for det in detections:
            bbox = det.bbox
            bw = bbox.x2 - bbox.x1
            bh = bbox.y2 - bbox.y1

            # Step 1: 自适应 padding
            pad_ratio = self._get_padding(bw, bh, w, h)
            px1, py1, px2, py2 = pad_bbox(
                bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                pad_ratio, w, h,
            )

            # 检查区域大小
            region_w = px2 - px1
            region_h = py2 - py1
            if region_w < self.min_region_size or region_h < self.min_region_size:
                logger.debug(f"区域过小 ({region_w}x{region_h})，跳过")
                continue

            # Step 2: 裁剪
            crop = crop_region(frame, px1, py1, px2, py2)
            if crop is None:
                continue

            # Step 3: 编码为 base64
            crop_b64 = encode_image_to_base64(crop, fmt=".jpg", quality=85)

            results.append({
                "detection": det,
                "padded_bbox": (px1, py1, px2, py2),
                "crop_b64": crop_b64,
                "crop_image": crop,
            })

        return results

    def extract_keyframe_sequence(
        self,
        frame_buffer: list[tuple[np.ndarray, list[PersonDetection]]],
        target_person_id: int | None = None,
    ) -> list[str]:
        """
        从帧缓冲区中提取连续关键帧序列。

        对同一个人在多帧中的人体区域进行 padding 扩展后裁剪，
        按 keyframe_interval 间隔提取关键帧。

        Args:
            frame_buffer: [(frame, detections), ...] 帧和对应检测结果的列表
            target_person_id: 目标人物索引（默认取置信度最高的人）

        Returns:
            base64 编码的关键帧列表
        """
        if not frame_buffer:
            return []

        keyframes_b64: list[str] = []

        for i, (frame, detections) in enumerate(frame_buffer):
            # 按间隔采样
            if i % self.keyframe_interval != 0:
                continue

            if not detections:
                continue

            # 选择目标人物
            target_idx = target_person_id if target_person_id is not None else 0
            if target_idx >= len(detections):
                target_idx = 0  # fallback

            det = detections[target_idx]
            h, w = frame.shape[:2]
            bw = det.bbox.x2 - det.bbox.x1
            bh = det.bbox.y2 - det.bbox.y1

            # 自适应扩展并裁剪
            pad_ratio = self._get_padding(bw, bh, w, h)
            px1, py1, px2, py2 = pad_bbox(
                det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2,
                pad_ratio, w, h,
            )

            crop = crop_region(frame, px1, py1, px2, py2)
            if crop is None:
                continue

            crop_b64 = encode_image_to_base64(crop, fmt=".jpg", quality=80)
            keyframes_b64.append(crop_b64)

            # 达到最大关键帧数
            if len(keyframes_b64) >= self.keyframe_count:
                break

        return keyframes_b64

    def extract_multi_person_keyframes(
        self,
        frame_buffer: list[tuple[np.ndarray, list[PersonDetection]]],
        tracker_enabled: bool = True,
    ) -> dict[int, list[str]]:
        """
        为帧缓冲区中的每个人物分别提取关键帧序列。

        优先使用 track_id（ByteTrack 跟踪 ID）做人物归组；
        若未启用跟踪，按单帧逐人独立分析（避免跨帧身份混合）。

        Args:
            frame_buffer: [(frame, detections), ...] 缓冲区
            tracker_enabled: 是否已启用跟踪（由 pipeline 传入，不靠运行时嗅探）

        Returns:
            {person_key: [base64_frame, ...], ...}
            person_key 为 track_id（有跟踪）或 list index（无跟踪，仅限单帧内）
        """
        if not frame_buffer:
            return {}

        # 判断实际是否获得了 track_id（配置说启用但实际可能没有时，以实际为准）
        has_tracking = tracker_enabled and any(
            det.track_id is not None
            for _, detections in frame_buffer
            for det in detections
        )

        if has_tracking:
            # ===== 跟踪模式：按 track_id 跨帧归组 =====
            person_frames: dict[int, list[tuple[np.ndarray, PersonDetection]]] = {}
            for frame, detections in frame_buffer:
                for det in detections:
                    if det.track_id is None:
                        continue
                    key = det.track_id
                    if key not in person_frames:
                        person_frames[key] = []
                    person_frames[key].append((frame, det))
        else:
            # ===== 无跟踪模式：只用最新一帧，逐人独立分析 =====
            last_frame, last_detections = frame_buffer[-1]
            person_frames = {}
            for idx, det in enumerate(last_detections):
                person_frames[idx] = [(last_frame, det)]

        result: dict[int, list[str]] = {}

        for person_key, frames_list in person_frames.items():
            keyframes_b64: list[str] = []

            for i, (frame, det) in enumerate(frames_list):
                if i % self.keyframe_interval != 0:
                    continue

                h, w = frame.shape[:2]
                bw = det.bbox.x2 - det.bbox.x1
                bh = det.bbox.y2 - det.bbox.y1

                pad_ratio = self._get_padding(bw, bh, w, h)
                px1, py1, px2, py2 = pad_bbox(
                    det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2,
                    pad_ratio, w, h,
                )

                crop = crop_region(frame, px1, py1, px2, py2)
                if crop is None:
                    continue

                crop_b64 = encode_image_to_base64(crop, fmt=".jpg", quality=80)
                keyframes_b64.append(crop_b64)

                if len(keyframes_b64) >= self.keyframe_count:
                    break

            if keyframes_b64:
                result[person_key] = keyframes_b64

        return result
