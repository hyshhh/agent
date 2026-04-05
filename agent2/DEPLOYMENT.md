# 智能救生行为识别 Agent — 部署与使用教程

## 目录
1. [环境准备](#1-环境准备)
2. [方式一：云端 API 模式（推荐快速体验）](#2-方式一云端-api-模式)
3. [方式二：本地 vLLM 部署（推荐生产使用）](#3-方式二本地-vllm-部署)
4. [配置说明](#4-配置说明)
5. [运行命令](#5-运行命令)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 基础依赖

```bash
# 克隆项目
git clone https://github.com/hyshhh/agent.git
cd agent/agent2

# 安装 Python 依赖
pip install -r requirements.txt
```

依赖列表：
- `ultralytics` — YOLOv8 人体检测
- `opencv-python` — 视频读取与图像处理
- `openai` — OpenAI 兼容 API 客户端（同时用于云端和本地）
- `PyYAML` — 配置文件解析
- `loguru` — 日志

### 1.2 硬件要求

| 模式 | GPU 显存 | 说明 |
|------|----------|------|
| 云端 API | 不需要 GPU | 仅 YOLOv8 CPU 推理即可 |
| 本地 vLLM（Qwen3-VL-4B） | ≥ 8GB（FP16）| 单卡 RTX 3060 12GB 可跑 |
| 本地 vLLM（Qwen3-VL-4B 量化） | ≥ 6GB（INT8/AWQ）| 更省显存 |

---

## 2. 方式一：云端 API 模式

适合快速体验，不需要 GPU，但需要网络和 API Key。

### 2.1 获取千问 API Key

1. 注册 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 开通模型服务，获取 API Key
3. 将 Key 填入 `config.yaml` 或通过环境变量/命令行传入

### 2.2 修改配置

编辑 `config.yaml`：
```yaml
model_mode: "api"    # ← 云端模式

qwen:
  api_key: "sk-xxxxxxxx"        # 你的千问 API Key
  model: "qwen3-vl-flash"       # 推荐，性价比高
  # model: "qwen-vl-max"        # 效果更好但更贵
```

或通过环境变量：
```bash
export QWEN_API_KEY="sk-xxxxxxxx"
```

### 2.3 运行

```bash
# 分析视频文件
python main.py --source video --input /path/to/video.mp4

# 也可以命令行指定 Key
python main.py --source video --input video.mp4 --api-key sk-xxxxxxxx
```

---

## 3. 方式二：本地 vLLM 部署

完全本地运行，无需联网，数据不出本机，适合隐私敏感场景。

### 3.1 创建 conda 环境

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

### 3.2 安装 vLLM 和 modelscope

```bash
pip install vllm
pip install modelscope
```

> **注意**：vLLM 需要 CUDA 环境。确保你的 NVIDIA 驱动版本 ≥ 525，CUDA ≥ 11.8。

### 3.3 下载模型

```bash
# 下载 Qwen3-VL-4B-Instruct 到本地目录
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /your/local/path/qwen3-vl-4b

# 例如：
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /home/user/models/qwen3-vl-4b
```

> 下载大小约 8GB，请确保磁盘空间充足。

### 3.4 启动 vLLM 推理服务

```bash
# 单卡启动（推荐 12GB+ 显存）
CUDA_VISIBLE_DEVICES=0 vllm serve /your/local/path/qwen3-vl-4b \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 4096 \
  --port 7890

# 双卡启动（显存不足时使用张量并行）
CUDA_VISIBLE_DEVICES=0,1 vllm serve /your/local/path/qwen3-vl-4b \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --port 7890
```

**参数说明：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--api-key` | API 鉴权密钥（和 config.yaml 中 `local_model.api_key` 对应） | 自定义，如 `abc123` |
| `--served-model-name` | 模型名称标识（客户端请求时使用） | 保持 `Qwen/Qwen3-VL-4B-Instruct` |
| `--max-model-len` | 最大上下文长度（prompt + 生成的 token 总数） | `4096` 足够本项目 |
| `--port` | 服务端口 | `7890`（和 config.yaml 对应） |
| `--tensor-parallel-size` | 张量并行 GPU 数量 | 和 CUDA_VISIBLE_DEVICES 的卡数一致 |
| `--gpu-memory-utilization` | GPU 显存占用比例 | 默认 0.9，显存紧张可设 0.95 |

**等待服务启动完成**，看到类似输出表示就绪：
```
INFO:     Uvicorn running on http://0.0.0.0:7890 (Press CTRL+C to quit)
```

### 3.5 验证服务是否正常

另开一个终端测试：
```bash
curl http://localhost:7890/v1/models \
  -H "Authorization: Bearer abc123"
```

应返回模型列表，确认 `Qwen/Qwen3-VL-4B-Instruct` 在其中。

### 3.6 修改项目配置

编辑 `config.yaml`：
```yaml
model_mode: "local"    # ← 切换到本地模式

local_model:
  api_key: "abc123"                      # 和 vllm serve 的 --api-key 一致
  api_url: "http://localhost:7890/v1"    # vLLM 服务地址
  model: "Qwen/Qwen3-VL-4B-Instruct"    # 和 --served-model-name 一致
  max_tokens: 512
  temperature: 0.1
  timeout: 60
```

### 3.7 运行 Agent

```bash
# 分析视频（本地模型）
python main.py --source video --input /path/to/video.mp4

# USB 摄像头实时检测（本地模型）
python main.py --source camera --camera-id 0

# 无头模式（服务器无显示器）
python main.py --source video --input video.mp4 --no-display
```

---

## 4. 配置说明

### 4.1 切换模式的两种方式

**方式 A：修改 config.yaml**
```yaml
model_mode: "local"   # 或 "api"
```

**方式 B：命令行参数（优先级更高）**
```bash
python main.py --source video --input video.mp4 --model-mode local
python main.py --source video --input video.mp4 --model-mode api
```

### 4.2 完整配置文件说明

```yaml
# ===== 模式切换 =====
model_mode: "api"              # "api" = 云端千问, "local" = 本地 vLLM

# ===== 云端 API 配置 =====
qwen:
  api_key: ""                  # 千问 API Key
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen3-vl-flash"     # 模型名
  max_tokens: 100              # 最大输出 token
  temperature: 0.01            # 温度（越低越确定）
  timeout: 30                  # 超时秒数

# ===== 本地模型配置 =====
local_model:
  api_key: "abc123"            # vLLM --api-key
  api_url: "http://localhost:7890/v1"
  model: "Qwen/Qwen3-VL-4B-Instruct"
  max_tokens: 512              # 本地模型可以给大一点
  temperature: 0.1
  timeout: 60                  # 本地推理可能更慢

# ===== 检测器 =====
detector:
  model: "yolov8n.pt"          # YOLOv8 模型（首次自动下载）
  confidence: 0.5              # 检测置信度
  device: "cuda:0"             # "cpu" 或 "cuda:0"
  detect_width: 160            # 检测推理分辨率（越低越快）
  detect_height: 160

# ===== 关键帧提取 =====
frame_extractor:
  padding_ratio: 0.15          # 人体框扩展比例
  keyframe_interval: 1         # 每隔几帧取一帧
  keyframe_count: 1            # 最多取几帧
  min_region_size: 32          # 最小有效像素

# ===== 行为类别（可自行扩展） =====
behavior_classes:
  - id: drowning
    label_cn: 溺水
    severity: critical
    description: "人员在水中失去自主行动能力..."

# ===== 流水线 =====
pipeline:
  process_every_n_frames: 1    # 每 N 帧分析一次
  alert_cooldown: 5            # 同一行为告警冷却秒数
  display: true                # 是否显示画面
  display_scale: 0.1           # 窗口缩放
  camera_interval: 0.1         # 摄像头采集间隔
  sustained_detection_frames: 1 # 连续几帧检测到才触发

# ===== 输出 =====
output:
  save_annotated: true         # 保存标注帧
  save_crops: true             # 保存人体裁剪
  save_report: true            # 保存分析报告
  output_dir: "output"

# ===== 摄像头日志 =====
camera_log:
  enabled: true
  retention_hours: 2.0         # 日志保留时长
  log_filename: "camera_behavior_log.json"
```

---

## 5. 运行命令速查

```bash
# ========== 云端 API 模式 ==========
# 分析视频
python main.py --source video --input video.mp4

# USB 摄像头
python main.py --source camera --camera-id 0

# RTSP 流
python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream

# ========== 本地模型模式 ==========
# 分析视频
python main.py --source video --input video.mp4 --model-mode local

# USB 摄像头
python main.py --source camera --camera-id 0 --model-mode local

# RTSP 流
python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream --model-mode local

# ========== 通用选项 ==========
# 无头模式（不显示窗口）
python main.py --source video --input video.mp4 --no-display

# 详细日志
python main.py --source video --input video.mp4 --verbose

# 指定输出目录
python main.py --source video --input video.mp4 --output ./results

# 指定 API Key
python main.py --source video --input video.mp4 --api-key sk-xxxxx

# 指定配置文件
python main.py --source video --input video.mp4 --config my_config.yaml
```

---

## 6. 常见问题

### Q1: vLLM 启动报错 "CUDA out of memory"
**A:** 显存不足。解决方案：
- 减小 `--max-model-len`（如 `2048`）
- 添加 `--gpu-memory-utilization 0.95`
- 使用量化版本：`modelscope download --model Qwen/Qwen3-VL-4B-Instruct-AWQ`
- 使用双卡：`CUDA_VISIBLE_DEVICES=0,1 ... --tensor-parallel-size 2`

### Q2: 本地模型返回结果不如云端
**A:** 4B 模型本身比云端大模型弱。可以尝试：
- 增加 `keyframe_count`（如 3 帧）给模型更多信息
- 增加 `max_tokens`（如 1024）
- 降低 `temperature`（如 0.01）
- 使用更大模型如 Qwen3-VL-8B-Instruct

### Q3: 摄像头打不开
**A:** 
- USB 摄像头：确认设备 ID（`ls /dev/video*`），通常为 0
- RTSP：用 `ffplay rtsp://xxx` 先验证流是否可用
- 权限：`sudo usermod -aG video $USER` 然后重新登录

### Q4: YOLOv8 模型下载失败
**A:** 手动下载：
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```
放到项目根目录即可。

### Q5: 如何扩展行为类别？
**A:** 在 `config.yaml` 的 `behavior_classes` 中添加：
```yaml
behavior_classes:
  - id: fighting
    label_cn: 打斗
    severity: warning
    description: 人员之间发生肢体冲突
```

### Q6: 云端和本地能同时用吗？
**A:** 不能同时用，但可以随时切换。改 `config.yaml` 的 `model_mode` 或用 `--model-mode` 命令行参数即可。

---

## 输出文件说明

运行结束后，在 `output/` 目录下会生成：

```
output/
├── analysis_report.json           # 完整分析报告（行为统计、时长等）
├── alerts.json                    # 告警记录（仅触发告警时生成）
├── camera_behavior_log.json       # 摄像头模式行为日志（2小时滚动）
├── annotated/                     # 标注后的帧图片
│   ├── frame_000001.jpg
│   └── ...
└── crops/                         # 人体裁剪图
    ├── frame000001_person0.jpg
    └── ...
```
