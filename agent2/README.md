# 智能救生行为识别 Agent

基于人体检测 + 千问多模态大模型的行为识别系统，支持视频文件和摄像头实时输入。

## 功能特性

- **人体检测**：基于 YOLOv8 的高精度人体检测
- **行为识别**：通过千问多模态模型分析关键帧，识别以下行为：
  - 🏊 游泳 (swimming)
  - 🆘 溺水 (drowning)
  - 🚧 翻栏杆 (climbing_fence)
  - 🚶 正常步行 (normal_walking)
  - ❓ 未知行为 (unknown)
- **多输入源**：支持视频文件、USB 摄像头、RTSP 网络摄像头
- **断线重连**：网络摄像头自动重连机制（连续 10 帧失败触发）
- **关键帧提取**：对检测到的人体区域加 padding 后提取连续关键帧
- **可扩展架构**：行为类别可配置扩展

## 项目结构

```
agent/
├── README.md
├── requirements.txt
├── config.yaml                  # 全局配置
├── main.py                      # 主入口
├── core/
│   ├── __init__.py
│   ├── detector.py              # 人体检测器
│   ├── frame_extractor.py       # 关键帧提取器
│   ├── behavior_classifier.py   # 千问模型行为分类器
│   ├── video_source.py          # 视频输入源管理
│   └── pipeline.py              # 主流水线
├── models/
│   ├── __init__.py
│   └── schemas.py               # 数据模型定义
├── utils/
│   ├── __init__.py
│   ├── logger.py                # 日志工具
│   └── image_utils.py           # 图像处理工具
└── output/                      # 输出目录
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 分析视频文件
python main.py --source video --input G:\YZ\云洲\原视频素材\original_video\dangerofdrowning_25.mp4

# 使用 USB 摄像头
python main.py --source camera --camera-id 0

# 使用 RTSP 流
python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream

# 指定配置文件
python main.py --source video --input video.mp4 --config config.yaml
```

## 配置

编辑 `config.yaml` 修改检测参数、模型 API Key 等。

## 环境变量

- `QWEN_API_KEY`：千问 API 密钥（必填）
- `QWEN_API_URL`：千问 API 地址（可选，默认为阿里云百炼平台地址）
