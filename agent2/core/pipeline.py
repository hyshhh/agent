"""主流水线 — 将检测、提取、分类串联"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from typing import Optional, Callable

import cv2
import numpy as np

from core.detector import PersonDetector
from core.frame_extractor import FrameExtractor
from core.behavior_classifier import BehaviorClassifier
from core.video_source import VideoSource, VideoSourceType
from models.schemas import (
    FrameAnalysis,
    PersonDetection,
    BehaviorResult,
    AnalysisReport,
    Severity,
)
from utils.image_utils import draw_detections, save_image
from utils.logger import get_logger

logger = get_logger()


class CameraBehaviorLog:
    """
    摄像头行为日志管理器（需求2）。

    功能：
    - 记录每次行为识别结果，包含时间戳和行为信息
    - 定期清理超过保留时长的日志条目
    - 以 JSON 格式持久化存储
    """

    def __init__(
        self,
        output_dir: str,
        retention_hours: float = 2.0,
        log_filename: str = "camera_behavior_log.json",
    ):
        """
        Args:
            output_dir: 输出目录
            retention_hours: 日志保留时长（小时）
            log_filename: 日志文件名
        """
        self.output_dir = output_dir
        self.retention_seconds = retention_hours * 3600
        self.log_path = os.path.join(output_dir, log_filename)

        # 日志条目列表 [{timestamp, datetime, frame_index, person_idx, behavior_id, behavior_label, severity, description}, ...]
        self._entries: list[dict] = []

        # 加载已有日志
        self._load_existing()

    def _load_existing(self):
        """加载已有的日志文件"""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                logger.info(f"已加载摄像头日志: {len(self._entries)} 条记录")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"加载日志失败，重新开始: {e}")
                self._entries = []

    def add_entry(
        self,
        frame_index: int,
        person_idx: int,
        result: BehaviorResult,
    ):
        """
        添加一条行为日志。

        Args:
            frame_index: 帧序号
            person_idx: 人物索引
            result: 行为识别结果
        """
        now = time.time()
        entry = {
            "timestamp": round(now, 3),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "frame_index": frame_index,
            "person_idx": person_idx,
            "behavior_id": result.behavior_id,
            "behavior_label": result.behavior_label,
            "severity": result.severity.value,
            "description": result.description,
        }
        self._entries.append(entry)
        # 每次添加日志后立即保存
        self.save()

    def _cleanup(self):
        """清理超过保留时长的日志条目"""
        now = time.time()
        cutoff = now - self.retention_seconds
        before_count = len(self._entries)
        self._entries = [e for e in self._entries if e.get("timestamp", 0) >= cutoff]
        removed = before_count - len(self._entries)
        if removed > 0:
            logger.debug(f"摄像头日志清理: 删除 {removed} 条过期记录")

    def save(self):
        """保存日志到文件（先清理再保存）"""
        self._cleanup()
        try:
            logger.debug(f"开始保存摄像头日志: {self.log_path}, 条目数={len(self._entries)}")
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            logger.info(f"摄像头日志文件已写入: {self.log_path}")
        except Exception as e:
            logger.error(f"保存摄像头日志失败: {e}")

    @property
    def entry_count(self) -> int:
        return len(self._entries)


class Pipeline:
    """
    行为识别主流水线。

    处理流程（每帧）：
    1. 从视频源读取帧
    2. 人体检测器识别画面中的人
    3. 将检测结果累积到帧缓冲区
    4. 达到指定帧数后，提取关键帧序列
    5. 调用千问模型分析行为
    6. 输出结果（显示/保存/告警回调）

    帧缓冲机制：
    - 维护一个滑动窗口帧缓冲区
    - 每 process_every_n_frames 帧触发一次完整分析
    - 为每个人物单独提取关键帧并分析

    持续帧检测机制（需求3）：
    - 只有连续 sustained_detection_frames 帧都有检测结果时，才触发API调用
    - 任何一帧检测中断则计数器重置
    """

    def __init__(
        self,
        detector: PersonDetector,
        frame_extractor: FrameExtractor,
        classifier: BehaviorClassifier,
        video_source: VideoSource,
        process_every_n_frames: int = 5,
        camera_interval: float = 0.1,  # 摄像头调用间隔（秒）
        alert_cooldown: int = 30,
        sustained_detection_frames: int = 1,  # 需求3：连续N帧检测到才触发API
        output_dir: str = "output",
        save_annotated: bool = True,
        save_crops: bool = True,
        save_report: bool = True,
        display: bool = True,
        display_scale: float = 0.5,  # 视频窗口缩放比例
        camera_log_enabled: bool = True,         # 需求2：是否启用摄像头日志
        camera_log_retention_hours: float = 2.0,  # 需求2：日志保留时长
        camera_log_filename: str = "camera_behavior_log.json",
        alert_callback: Optional[Callable] = None,
    ):
        """
        Args:
            detector: 人体检测器
            frame_extractor: 关键帧提取器
            classifier: 行为分类器
            video_source: 视频输入源
            process_every_n_frames: 每 N 帧触发一次行为分析
            camera_interval: 摄像头调用间隔（秒）
            alert_cooldown: 同一行为告警冷却秒数
            sustained_detection_frames: 连续N帧检测到目标才触发API（需求3）
            output_dir: 输出目录
            save_annotated: 是否保存标注帧
            save_crops: 是否保存人体裁剪图
            save_report: 是否保存分析报告
            display: 是否显示实时画面
            display_scale: 视频窗口缩放比例
            camera_log_enabled: 是否启用摄像头行为日志（需求2）
            camera_log_retention_hours: 日志保留时长-小时（需求2）
            camera_log_filename: 日志文件名（需求2）
            alert_callback: 告警回调函数 fn(behavior_result, frame, detections)
        """
        self.detector = detector
        self.extractor = frame_extractor
        self.classifier = classifier
        self.source = video_source
        self.process_interval = process_every_n_frames
        self.camera_interval = camera_interval
        self.alert_cooldown = alert_cooldown
        self.sustained_detection_frames = max(1, sustained_detection_frames)
        self.output_dir = output_dir
        self.save_annotated = save_annotated
        self.save_crops = save_crops
        self.save_report = save_report
        self.display = display
        self.display_scale = display_scale
        self.alert_callback = alert_callback

        # 帧缓冲区（滑动窗口）
        self._frame_buffer: deque[tuple[np.ndarray, list[PersonDetection]]] = deque(
            maxlen=process_every_n_frames
        )

        # 告警冷却记录 {behavior_id: last_alert_time}
        self._alert_cooldowns: dict[str, float] = {}

        # 分析报告数据
        self._report = AnalysisReport(source="")
        self._frame_count = 0
        self._processed_count = 0

        # 需求3：持续帧计数器 — 连续检测到目标的帧数
        self._consecutive_detection_count = 0

        # 需求2：摄像头行为日志
        self._camera_log: Optional[CameraBehaviorLog] = None
        self._camera_log_enabled = camera_log_enabled
        self._camera_log_retention_hours = camera_log_retention_hours
        self._camera_log_filename = camera_log_filename

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_crops:
            os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
        if save_annotated:
            os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)

        # 需求2：如果是摄像头模式且启用日志，初始化日志管理器
        logger.debug(f"调试信息: _camera_log_enabled={self._camera_log_enabled}, source_type={self.source.source_type}, type={type(self.source.source_type)}")
        # 检查是否为摄像头模式（USB摄像头或RTSP流）
        is_camera_mode = self.source.source_type in (VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP)
        logger.debug(f"调试信息: is_camera_mode={is_camera_mode}")
        
        if self._camera_log_enabled and is_camera_mode:
            self._camera_log = CameraBehaviorLog(
                output_dir=output_dir,
                retention_hours=camera_log_retention_hours,
                log_filename=camera_log_filename,
            )
            logger.info(f"摄像头日志已初始化: {self._camera_log.log_path}")
        else:
            logger.debug(f"摄像头日志未初始化: _camera_log_enabled={self._camera_log_enabled}, is_camera_mode={is_camera_mode}")

    def run(self):
        """
        启动主流水线。

        从视频源读取帧，执行检测→提取→分类的完整流程。
        支持按键中断（按 'q' 退出）。
        """
        logger.info("=" * 50)
        logger.info("行为识别流水线启动")
        logger.info(f"  持续帧阈值: {self.sustained_detection_frames} 帧")
        logger.info(f"  处理间隔: 每 {self.process_interval} 帧")
        if self._camera_log is not None:
            logger.info(f"  摄像头日志: 已启用, 保留 {self._camera_log_retention_hours}h")
        logger.info("=" * 50)

        self._report.source = str(self.source.source_type.value)
        self._report.start_time = time.time()

        try:
            last_camera_time = 0
            for frame in self.source.frames():
                # 控制摄像头调用间隔（仅对摄像头/RTSP 有效，不影响视频文件）
                current_time = time.time()
                is_camera = self.source.source_type in (
                    VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP,
                )
                if is_camera:
                    if current_time - last_camera_time < self.camera_interval:
                        continue
                last_camera_time = current_time

                self._frame_count += 1
                self._report.total_frames = self._frame_count

                # Step 1: 人体检测
                detections = self.detector.detect(frame, self._frame_count)

                # Step 2: 加入帧缓冲区
                self._frame_buffer.append((frame.copy(), detections))

                # Step 3: 需求3 — 持续帧计数
                if detections:
                    self._consecutive_detection_count += 1
                else:
                    self._consecutive_detection_count = 0

                # Step 4: 定期触发行为分析（需求3：需同时满足持续帧条件）
                frame_analysis = None
                sustained_enough = (
                    self._consecutive_detection_count >= self.sustained_detection_frames
                )
                if (
                    self._frame_count % self.process_interval == 0
                    and detections
                    and sustained_enough
                ):
                    frame_analysis = self._analyze_buffer(self._frame_count)
                    self._processed_count += 1

                # Step 5: 可视化
                if self.display:
                    annotated = self._draw_frame(frame, detections, frame_analysis)
                    # 显示持续帧信息
                    cv2.putText(
                        annotated,
                        f"Sustained: {self._consecutive_detection_count}/{self.sustained_detection_frames}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1,
                    )
                    # 缩小视频窗口
                    if self.display_scale != 1.0:
                        h, w = annotated.shape[:2]
                        new_w = int(w * self.display_scale)
                        new_h = int(h * self.display_scale)
                        annotated = cv2.resize(annotated, (new_w, new_h))
                    cv2.imshow("Behavior Recognition Agent", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("用户按 'q' 退出")
                        break
                    elif key == ord("s"):
                        # 手动截图
                        path = os.path.join(self.output_dir, f"screenshot_{self._frame_count}.jpg")
                        save_image(annotated, path)
                        logger.info(f"截图已保存: {path}")

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在退出...")

        finally:
            try:
                self._finalize()
            except Exception as e:
                logger.error(f"清理过程中发生错误: {e}")
                # 确保日志被保存
                if self._camera_log is not None:
                    try:
                        self._camera_log.save()
                        logger.info(f"摄像头行为日志已保存: {self._camera_log.log_path}")
                    except Exception as log_error:
                        logger.error(f"保存摄像头日志失败: {log_error}")

    def _analyze_buffer(self, frame_index: int) -> Optional[FrameAnalysis]:
        """
        分析帧缓冲区中的内容。

        对缓冲区中的每个人物提取关键帧并调用模型分析。
        """
        if not self._frame_buffer:
            return None

        start_time = time.time()
        frame, last_detections = self._frame_buffer[-1]
        h, w = frame.shape[:2]

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            detections=last_detections,
        )

        # 为每个人物提取关键帧并分析
        person_keyframes = self.extractor.extract_multi_person_keyframes(
            list(self._frame_buffer)
        )

        behaviors: list[BehaviorResult] = []

        for person_idx, keyframes_b64 in person_keyframes.items():
            if not keyframes_b64:
                continue

            # 调用千问模型
            result = self.classifier.classify(keyframes_b64)
            behaviors.append(result)

            logger.info(
                f"[帧 {frame_index}] 人物#{person_idx}: "
                f"{result.behavior_label} ({result.behavior_id}) "
                f"[{result.severity.value}]"
            )

            # 保存裁剪图
            if self.save_crops and person_idx < len(last_detections):
                self._save_crop(frame, last_detections[person_idx], frame_index, person_idx)

            # 告警处理
            if result.is_alert():
                self._handle_alert(result, frame_index, person_idx)

            # 记录到报告
            self._record_behavior(result, frame_index)

            # 需求2：记录到摄像头行为日志
            if self._camera_log is not None:
                self._camera_log.add_entry(
                    frame_index=frame_index,
                    person_idx=person_idx,
                    result=result,
                )
                logger.debug(f"日志条目已添加: frame_index={frame_index}, person_idx={person_idx}, behavior={result.behavior_id}")
                logger.debug(f"当前日志条目数: {self._camera_log.entry_count}")

        analysis.behaviors = behaviors
        analysis.processing_time = time.time() - start_time

        # 保存标注帧
        if self.save_annotated:
            behavior_dicts = [
                {"behavior_label": b.behavior_label, "severity": b.severity.value}
                for b in behaviors
            ]
            annotated = draw_detections(frame, last_detections, behavior_dicts)
            path = os.path.join(self.output_dir, "annotated", f"frame_{frame_index:06d}.jpg")
            save_image(annotated, path)

        self._report.frame_analyses.append(analysis)
        return analysis

    def _draw_frame(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        analysis: Optional[FrameAnalysis] = None,
    ) -> np.ndarray:
        """绘制带检测框和行为标签的帧"""
        behavior_dicts = None
        if analysis and analysis.behaviors:
            behavior_dicts = [
                {"behavior_label": b.behavior_label, "severity": b.severity.value}
                for b in analysis.behaviors
            ]

        annotated = draw_detections(frame, detections, behavior_dicts)

        # 添加帧信息
        info = f"Frame: {self._frame_count} | Persons: {len(detections)} | Processed: {self._processed_count}"
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 添加告警指示
        if analysis:
            for b in analysis.behaviors:
                if b.is_alert():
                    alert_text = f"ALERT: {b.behavior_label}!"
                    cv2.putText(
                        annotated, alert_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                    )

        return annotated

    def _save_crop(
        self,
        frame: np.ndarray,
        detection: PersonDetection,
        frame_index: int,
        person_idx: int,
    ):
        """保存人体裁剪图"""
        from utils.image_utils import pad_bbox, crop_region

        h, w = frame.shape[:2]
        bbox = detection.bbox
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        pad_ratio = self.extractor._get_padding(bw, bh, w, h)
        px1, py1, px2, py2 = pad_bbox(
            bbox.x1, bbox.y1, bbox.x2, bbox.y2,
            pad_ratio, w, h,
        )

        crop = crop_region(frame, px1, py1, px2, py2)
        if crop is not None:
            path = os.path.join(
                self.output_dir, "crops",
                f"frame{frame_index:06d}_person{person_idx}.jpg"
            )
            save_image(crop, path)

    def _handle_alert(
        self,
        result: BehaviorResult,
        frame_index: int,
        person_idx: int,
    ):
        """处理告警（含冷却机制）"""
        now = time.time()
        last_alert = self._alert_cooldowns.get(result.behavior_id, 0)

        if now - last_alert < self.alert_cooldown:
            logger.debug(
                f"告警冷却中: {result.behavior_id}, "
                f"剩余 {self.alert_cooldown - (now - last_alert):.0f}s"
            )
            return

        self._alert_cooldowns[result.behavior_id] = now

        alert_msg = (
            f"🚨 告警! [帧 {frame_index}] 人物#{person_idx}: "
            f"{result.behavior_label} ({result.severity.value}) - {result.description}"
        )
        logger.warning(alert_msg)

        # 记录告警
        self._report.alerts.append({
            "frame_index": frame_index,
            "person_idx": person_idx,
            "behavior_id": result.behavior_id,
            "behavior_label": result.behavior_label,
            "severity": result.severity.value,
            "description": result.description,
            "timestamp": time.time(),
        })

        # 回调通知
        if self.alert_callback:
            try:
                self.alert_callback(result, frame_index, person_idx)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

    def _record_behavior(self, result: BehaviorResult, frame_index: int):
        """记录行为到报告统计"""
        bid = result.behavior_id
        self._report.behavior_counts[bid] = self._report.behavior_counts.get(bid, 0) + 1

    def _finalize(self):
        """流水线结束，保存报告"""
        self._report.end_time = time.time()
        self._report.processed_frames = self._processed_count

        # 需求2：保存摄像头行为日志（在释放视频源之前保存）
        if self._camera_log is not None:
            logger.debug(f"保存摄像头日志: 条目数={self._camera_log.entry_count}, log_path={self._camera_log.log_path}")
            self._camera_log.save()
            logger.info(
                f"摄像头行为日志已保存: {self._camera_log.log_path} "
                f"({self._camera_log.entry_count} 条, 保留 {self._camera_log_retention_hours}h)"
            )
        else:
            logger.debug("摄像头日志未初始化，跳过保存")

        # 释放视频源
        try:
            self.source.release()
        except Exception as e:
            logger.error(f"释放视频源失败: {e}")

        # 关闭显示窗口
        if self.display:
            cv2.destroyAllWindows()

        # 保存报告
        if self.save_report:
            report_path = os.path.join(self.output_dir, "analysis_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self._report.summary(), f, ensure_ascii=False, indent=2)
            logger.info(f"分析报告已保存: {report_path}")

            # 保存详细告警记录
            if self._report.alerts:
                alerts_path = os.path.join(self.output_dir, "alerts.json")
                with open(alerts_path, "w", encoding="utf-8") as f:
                    json.dump(self._report.alerts, f, ensure_ascii=False, indent=2)
                logger.info(f"告警记录已保存: {alerts_path} ({len(self._report.alerts)} 条)")

        # 打印摘要
        summary = self._report.summary()
        logger.info("=" * 50)
        logger.info("分析完成! 摘要:")
        logger.info(f"  输入源: {summary['source']}")
        logger.info(f"  运行时长: {summary['duration_seconds']}s")
        logger.info(f"  总帧数: {summary['total_frames']}")
        logger.info(f"  分析帧数: {summary['processed_frames']}")
        logger.info(f"  总检测数: {summary['total_detections']}")
        logger.info(f"  行为统计: {summary['behavior_counts']}")
        logger.info(f"  告警次数: {summary['alert_count']}")
        if self._camera_log is not None:
            logger.info(f"  日志条目: {self._camera_log.entry_count}")
        logger.info("=" * 50)
