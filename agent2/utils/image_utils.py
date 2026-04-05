"""图像处理工具"""

from __future__ import annotations

import io
import base64
from typing import Optional

import cv2
import numpy as np


def pad_bbox(
    x1: float, y1: float, x2: float, y2: float,
    padding_ratio: float,
    frame_w: int, frame_h: int,
) -> tuple[int, int, int, int]:
    """
    对检测框按比例扩展 padding，并裁剪到画面边界内。

    Args:
        x1, y1, x2, y2: 原始检测框坐标
        padding_ratio: 扩展比例（如 0.15 表示四边各扩 15%）
        frame_w, frame_h: 帧的宽高

    Returns:
        (px1, py1, px2, py2) 整数坐标，保证在画面范围内
    """
    w = x2 - x1
    h = y2 - y1
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio

    px1 = max(0, int(x1 - pad_w))
    py1 = max(0, int(y1 - pad_h))
    px2 = min(frame_w, int(x2 + pad_w))
    py2 = min(frame_h, int(y2 + pad_h))

    return px1, py1, px2, py2


def crop_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """从帧中裁剪指定区域"""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def encode_image_to_base64(image: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """
    将 OpenCV 图像编码为 base64 字符串。

    Args:
        image: BGR 格式的 numpy 数组
        fmt: 编码格式（.jpg / .png）
        quality: JPEG 质量（1-100）

    Returns:
        base64 编码的字符串（不包含 data URI 前缀）
    """
    if fmt == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        encode_params = []

    success, buffer = cv2.imencode(fmt, image, encode_params)
    if not success:
        raise ValueError(f"Failed to encode image to {fmt}")

    return base64.b64encode(buffer).decode("utf-8")


def resize_keep_ratio(image: np.ndarray, max_width: int = 0, max_height: int = 0) -> np.ndarray:
    """等比缩放图像到指定最大尺寸"""
    h, w = image.shape[:2]

    if max_width > 0 and max_height > 0:
        scale = min(max_width / w, max_height / h)
    elif max_width > 0:
        scale = max_width / w
    elif max_height > 0:
        scale = max_height / h
    else:
        return image

    if scale >= 1.0:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_detections(
    frame: np.ndarray,
    detections: list,
    behaviors: list[dict] | None = None,
) -> np.ndarray:
    """
    在帧上绘制检测框和行为标签。

    Args:
        frame: 原始帧（BGR）
        detections: PersonDetection 列表
        behaviors: 行为结果列表，index 与 detections 对应

    Returns:
        标注后的帧
    """
    annotated = frame.copy()
    severity_colors = {
        "critical": (0, 0, 255),   # 红色
        "warning": (0, 165, 255),  # 橙色
        "normal": (0, 255, 0),     # 绿色
    }

    for i, det in enumerate(detections):
        bbox = det.bbox
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        color = (0, 255, 0)  # 默认绿色
        label = f"Person {bbox.confidence:.2f}"

        if behaviors and i < len(behaviors):
            b = behaviors[i]
            color = severity_colors.get(b.get("severity", "normal"), (0, 255, 0))
            label = f"{b.get('behavior_label', '?')} {bbox.confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return annotated


def save_image(image: np.ndarray, path: str) -> bool:
    """保存图像到文件"""
    return cv2.imwrite(path, image)
