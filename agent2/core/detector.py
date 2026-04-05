"""人体检测器 — 基于 YOLOv8"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from models.schemas import BoundingBox, PersonDetection
from utils.logger import get_logger

logger = get_logger()


class PersonDetector:
    """
    使用 YOLOv8 进行人体检测。

    支持：
    - 自动下载预训练模型
    - CPU / GPU 推理
    - 可配置置信度阈值
    - 可配置检测分辨率（降低推理时间）
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
        class_id: int = 0,
        detect_width: int = 0,
        detect_height: int = 0,
    ):
        """
        Args:
            model_path: YOLOv8 模型路径（首次运行自动下载）
            confidence: 最低置信度阈值
            device: 推理设备 ("cpu" / "cuda:0")
            class_id: COCO 数据集中 person 类的 ID（0）
            detect_width: 检测推理宽度（0=保持原始分辨率）
            detect_height: 检测推理高度（0=保持原始分辨率）
        """
        self.confidence = confidence
        self.device = device
        self.class_id = class_id
        self.detect_width = detect_width
        self.detect_height = detect_height

        logger.info(f"加载 YOLOv8 模型: {model_path} (device={device})")
        self.model = YOLO(model_path)
        logger.info(
            f"模型加载完成, 检测分辨率: "
            f"{'原始' if detect_width == 0 else f'{detect_width}x{detect_height}'}"
        )

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[PersonDetection]:
        """
        在单帧图像中检测人体。

        Args:
            frame: BGR 格式的帧图像
            frame_index: 帧序号（用于追踪）

        Returns:
            PersonDetection 列表，按置信度降序排列
        """
        import time

        orig_h, orig_w = frame.shape[:2]

        # 缩放到检测分辨率（需求1：降低推理时间）
        if self.detect_width > 0 and self.detect_height > 0:
            infer_frame = cv2.resize(
                frame, (self.detect_width, self.detect_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = orig_w / self.detect_width
            scale_y = orig_h / self.detect_height
        else:
            infer_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        results = self.model(
            infer_frame,
            device=self.device,
            classes=[self.class_id],
            conf=self.confidence,
            verbose=False,
        )

        detections: list[PersonDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # 提取坐标和置信度
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # 将坐标映射回原始分辨率
                x1 = float(xyxy[0]) * scale_x
                y1 = float(xyxy[1]) * scale_y
                x2 = float(xyxy[2]) * scale_x
                y2 = float(xyxy[3]) * scale_y

                bbox = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )

                # 过滤太小的框（可能是噪声）
                if bbox.width < 20 or bbox.height < 20:
                    continue

                detections.append(
                    PersonDetection(
                        frame_index=frame_index,
                        timestamp=time.time(),
                        bbox=bbox,
                    )
                )

        # 按置信度降序
        detections.sort(key=lambda d: d.bbox.confidence, reverse=True)

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[PersonDetection]]:
        """批量检测多帧"""
        return [self.detect(frame, i) for i, frame in enumerate(frames)]
