# 智能救生行为识别 Agent — 从零开始教程

## 目录
1. [项目简介](#1-项目简介)
2. [环境准备](#2-环境准备)
3. [下载项目](#3-下载项目)
4. [安装依赖](#4-安装依赖)
5. [获取 API Key](#5-获取-api-key)
6. [第一次运行](#6-第一次运行)
7. [输入源说明](#7-输入源说明)
8. [配置详解](#8-配置详解)
9. [本地模型部署（可选）](#9-本地模型部署可选)
10. [微调 YOLOv8（可选）](#10-微调-yolov8可选)
11. [常见问题](#11-常见问题)

---

## 1. 项目简介

基于 **YOLOv8 人体检测** + **千问多模态大模型** 的行为识别系统。

工作流程：
```
摄像头/视频 → YOLOv8 检测人体 → 裁剪人体区域关键帧 → 千问模型分析行为 → 输出结果/告警
```

支持的行为：溺水、游泳、翻栏杆、正常步行、水中帮助（可自行扩展）。

---

## 2. 环境准备

### 2.1 系统要求

- **操作系统**：Linux / Windows / macOS
- **Python**：3.10 ~ 3.12
- **GPU**（可选）：有 NVIDIA 显卡可以用 GPU 加速检测，没有也能跑（CPU）。CUDA 11.8+ 即可。

### 2.2 安装 Python

**Linux（Ubuntu/Debian）：**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

**Windows：**
去 [python.org](https://www.python.org/downloads/) 下载安装，勾选 "Add to PATH"。

**macOS：**
```bash
brew install python
```

### 2.3 创建虚拟环境（推荐）

**方式 A：conda（推荐）**

如果没有 conda，先安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)：

```bash
# 下载安装（Linux）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 下载安装（Windows）
# 去 https://docs.conda.io/en/latest/miniconda.html 下载 .exe 安装包
```

创建环境：

```bash
conda create -n agent python=3.10 -y
conda activate agent
```

**方式 B：venv**

```bash
python3 -m venv agent-env
source agent-env/bin/activate    # Linux/macOS
# agent-env\Scripts\activate     # Windows
```

---

## 3. 下载项目

```bash
git clone -b agentv3 https://github.com/hyshhh/agent.git
cd agent/agent2
```

> 没有 git？先安装：`sudo apt install git -y`（Linux）或去 [git-scm.com](https://git-scm.com/) 下载（Windows）。

---

## 4. 安装依赖

```bash
pip install -r requirements.txt
```

如果下载慢，用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 依赖说明

| 包 | 用途 |
|---|---|
| `ultralytics` | YOLOv8 人体检测（首次运行自动下载模型） |
| `opencv-python` | 视频读取、图像处理 |
| `openai` | 调用千问 API（OpenAI 兼容接口） |
| `PyYAML` | 读取 config.yaml |
| `loguru` | 日志输出 |

---

## 5. 获取 API Key

本项目使用阿里云百炼千问模型进行行为分析，需要 API Key。

### 步骤：

1. 打开 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 开通"百炼大模型服务"
4. 在控制台 → API Key 管理 → 创建 Key
5. 复制 Key（格式：`sk-xxxxxxxx`）

### 配置 Key（三选一）：

**方式 A：写入 config.yaml**（推荐）
```yaml
qwen:
  api_key: "sk-你的key"
```

**方式 B：环境变量**
```bash
export QWEN_API_KEY="sk-你的key"
```

**方式 C：命令行参数**
```bash
python main.py --source video --input video.mp4 --api-key sk-你的key
```

---

## 6. 第一次运行

### 6.1 用视频文件测试

最简单的方式，准备一个视频文件（MP4），然后运行：

```bash
python main.py --source video --input /media/ddc/新加卷/hys/qmy/agent/agent2v1/2.mp4 --no-display
```

### 6.2 预期效果

运行后会弹出一个窗口，显示：
- 绿色框：检测到的人体
- 行为标签和置信度
- 告警信息（红色文字）

控制台会输出：
```
17:10:01 | INFO     | pipeline:run | 行为识别流水线启动
17:10:02 | INFO     | pipeline:_analyze_buffer | [帧 5] 人物#0: 游泳 (swimming) [normal]
```

按 `q` 退出，结果自动保存到 `output/` 目录。

### 6.3 无窗口模式（服务器）

如果在没有显示器的服务器上运行，加上 `--no-display`：
```bash
python main.py --source video --input /media/ddc/新加卷/hys/qmy/agent/agent2v1/2.mp4 --no-display
```

---

## 7. 输入源说明

### 7.1 视频文件

```bash
python main.py --source video --input video.mp4
```

支持格式：MP4、AVI、MOV、MKV 等 OpenCV 能读的格式。

### 7.2 USB 摄像头

```bash
python main.py --source camera --camera-id 0
```

- `camera-id 0` 是默认摄像头
- 如果有多个摄像头，试试 `--camera-id 1`、`--camera-id 2`

**Linux 查看摄像头设备：**
```bash
ls /dev/video*
```

### 7.3 RTSP 网络摄像头

```bash
python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream
```

- 替换为你的摄像头 RTSP 地址
- 网络断开会自动重连（连续 10 帧失败触发）

**测试 RTSP 是否可用：**
```bash
ffplay rtsp://192.168.1.100:554/stream
```

---

## 8. 配置详解

编辑 `config.yaml`，以下是关键配置项：

### 8.1 检测器

```yaml
detector:
  model: "yolov8n.pt"       # 模型大小: yolov8n(最快) < yolov8s < yolov8m < yolov8l(最准)
  confidence: 0.5            # 置信度阈值，越低检测越多但误检也越多
  device: "cpu"              # 改成 "cuda:0" 使用 GPU
  detect_width: 160          # 检测分辨率，越低越快
  detect_height: 160
```

### 8.2 关键帧提取

```yaml
frame_extractor:
  padding_ratio: 0.15       # 人体框扩展比例
  keyframe_interval: 1      # 每隔几帧取一帧
  keyframe_count: 1         # 最多取几帧（多帧=更多信息=更好效果=更贵API）
  adaptive_padding: true    # 小目标自动放大padding
  pixel_threshold: 10000    # 小目标阈值（像素²），调大→更积极地扩展小目标
```

### 8.3 行为类别

```yaml
behavior_classes:
  - id: drowning
    label_cn: 溺水
    severity: critical       # critical=危险告警 / warning=警告 / normal=正常
    description: 人员在水中失去自主行动能力...

  # 自行添加新类别：
  - id: fighting
    label_cn: 打斗
    severity: warning
    description: 人员之间发生肢体冲突
```

### 8.4 流水线

```yaml
pipeline:
  process_every_n_frames: 1     # 每N帧分析一次（越大→越省API调用）
  camera_interval: 0.1          # 摄像头采集间隔（秒）
  alert_cooldown: 5             # 同一行为告警冷却时间（秒）
  sustained_detection_frames: 1 # 连续N帧检测到才触发分析
  display: true                 # 是否显示画面
  display_scale: 0.1            # 窗口缩放比例
```

### 8.5 API 模型

```yaml
# 云端（默认）
qwen:
  api_key: "sk-xxx"
  model: "qwen3-vl-flash"      # 性价比高
  # model: "qwen-vl-max"       # 效果更好但更贵
  max_tokens: 100
  temperature: 0.01
```

---

## 9. 本地模型部署（可选）

不需要 API Key，完全本地运行，数据不出本机。

### 9.1 安装 vLLM（单独开启一个环境）

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

### 9.2 下载模型

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /你的模型保存路径
```

> 模型约 8GB，参考 [vLLM 官方支持模型列表](https://docs.vllm.com.cn/en/latest/usage/) 选择模型。

### 9.3 启动服务

**单卡启动：**
```bash
vllm serve /你的模型路径 \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.4
```

**多卡启动（张量并行）：**
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /你的模型路径 \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --port 7890
```

**参数说明：**

| 参数 | 说明 |
|---|---|
| `--api-key` | 自己设置的密钥，客户端请求时需要提供 |
| `--served-model-name` | 模型名称标识，客户端请求时使用 |
| `--max-model-len` | 最大处理长度（输入 prompt + 生成内容的 token 总数） |
| `--port` | 服务端口 |
| `--tensor-parallel-size` | 张量并行数量，和使用的 GPU 卡数保持一致 |
| `--gpu-memory-utilization` | GPU 显存占用比例（如 0.4 = 40%），显存紧张时调小 |

看到 `Uvicorn running on http://0.0.0.0:7890` 就表示启动成功。

### 9.4 测试服务

新开一个终端测试：
```bash
curl http://127.0.0.1:7890/v1/models -H "Authorization: Bearer abc123"
```

> `127.0.0.1` 是本地回环地址（localhost），永远只指向本机。

应返回模型列表，确认 `Qwen/Qwen3-VL-4B-Instruct` 在其中。

### 9.5 切换到本地模式

编辑 `config.yaml`：
```yaml
model_mode: "local"

local_model:
  api_key: "abc123"
  api_url: "http://localhost:7890/v1"
  model: "Qwen/Qwen3-VL-4B-Instruct"
```

或命令行：
```bash
python main.py --source video --input video.mp4 --model-mode local
```

---

## 10. 微调 YOLOv8（可选）

如果你想让检测器更准（比如专门检测泳池场景），可以用自己的数据集微调。

### 10.1 准备数据集

YOLO 格式：
```
dataset/
├── train/
│   ├── images/       # 训练图片
│   └── labels/       # 标注文件 .txt
├── val/
│   ├── images/       # 验证图片
│   └── labels/
└── dataset.yaml      # 配置文件
```

`dataset.yaml` 内容：
```yaml
path: ./dataset
train: train/images
val: val/images
names:
  0: person
  1: drowning
```

### 10.2 开始训练

```bash
python finetune_yolo.py --data dataset.yaml --epochs 50 --lr 0.001
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data` | 数据集配置文件 | 必填 |
| `--epochs` | 训练轮数 | 50 |
| `--lr` | 学习率 | 0.001 |
| `--batch` | 批次大小 | 16 |
| `--device` | 训练设备 | 自动 |
| `--pretrained` | 预训练权重 | yolov8n.pt |

### 10.3 使用微调后的模型

训练完成后，找到最佳权重：
```
runs/finetune/finetune_xxxx/weights/best.pt
```

编辑 `config.yaml`：
```yaml
detector:
  model: "runs/finetune/finetune_xxxx/weights/best.pt"
```

---

## 11. 常见问题

### Q: pip install 报错
```bash
# 用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: YOLOv8 模型下载失败
手动下载放到项目目录：
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

### Q: 摄像头打不开
```bash
# Linux 查看设备
ls /dev/video*
# 加权限
sudo usermod -aG video $USER
```

### Q: CUDA 报错 / 没有 GPU
改 `config.yaml`：
```yaml
detector:
  device: "cpu"
```

### Q: API 调用失败
- 检查 Key 是否正确
- 检查网络是否能访问 `dashscope.aliyuncs.com`
- 检查阿里云账户余额

### Q: 行为识别不准
- 增加 `keyframe_count`（如改成 3），给模型更多信息
- 降低 `temperature`（如 0.01）
- 使用更好的模型：`model: "qwen-vl-max"`

### Q: 运行太慢
- 降低 `detect_width` / `detect_height`（如 160→128）
- 增大 `process_every_n_frames`（如 5）
- 使用 GPU：`device: "cuda:0"`

---

## 输出文件

运行结束后在 `output/` 下：

```
output/
├── analysis_report.json           # 分析报告
├── alerts.json                    # 告警记录
├── camera_behavior_log.json       # 摄像头模式日志
├── annotated/                     # 标注帧图片
└── crops/                         # 人体裁剪图
```
