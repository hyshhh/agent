# 智能救生行为识别 Agent — 需求与功能文档

## 已完成功能

### ✅ 功能0：基础系统
- 基于 YOLOv8 的人体检测（支持 CPU / GPU）
- 千问多模态模型行为分类（云端 API / 本地 vLLM 两种模式）
- 行为类别：溺水、游泳、翻栏杆、正常步行、水中帮助（可扩展）
- 关键帧提取：对检测到的人体区域加 padding 后裁剪，编码为 base64 输入模型

### ✅ 功能1：视频输入
- 支持视频文件输入（MP4、AVI、MOV 等）
- 支持 USB 摄像头实时输入（cv2.VideoCapture）
- 支持 RTSP 网络摄像头流
- 断线重连机制：连续 10 帧读取失败自动重连

### ✅ 功能2：摄像头持续捕捉帧
- 摄像头模式下持续采集帧并实时分析
- 可配置采集间隔（camera_interval）
- 摄像头行为日志（2 小时滚动，JSON 格式持久化）

### ✅ 功能3：自适应 padding（像素阈值）
- 小目标（检测框面积 < pixel_threshold）→ padding 自动放大到固定最小裁剪面积
- 大目标（检测框面积 >= pixel_threshold）→ 正常使用 padding_ratio 倍率
- 配置接口：`frame_extractor.pixel_threshold`（默认 10000 像素²）

### ✅ 功能4：YOLOv8 微调脚本
- 独立运行脚本 `finetune_yolo.py`
- 用户只需提供三个参数：数据集路径、学习率、训练轮数
- 预配置优化器（AdamW）、数据增强、早停、层冻结等
- 训练完成后自动验证并输出 mAP 指标

### ✅ 功能5：代码调优
- 摄像头间隔控制仅对 camera/rtsp 生效，不影响视频文件处理
- 持续帧检测机制（sustained_detection_frames）
- 告警冷却机制（alert_cooldown）
- 帧缓冲区滑动窗口（process_every_n_frames）

---

## 需求

_（待补充）_
