# agent

水上行为识别系统 — 基于 YOLOv8 + 多模态大模型的实时行为分析

## 快速开始

```bash
cd agent2
pip install -r requirements.txt  # 如有
python main.py --config config.yaml
```

## 配置参数详解

### 模型运行模式

```yaml
model_mode: "api"    # "api" = 阿里云百炼千问 | "local" = 本地 vLLM
```

### 检测器 (detector)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `yolov8n.pt` | YOLOv8 模型路径，首次运行自动下载 |
| `confidence` | `0.5` | 检测置信度阈值，低于此值的框直接丢弃 |
| `device` | `cpu` | 推理设备：`cpu` / `cuda:0` |
| `class_id` | `0` | COCO 数据集类别 ID，`0` = person |
| `detect_width` | `0` | 推理宽度，`0` = 保持原始分辨率 |
| `detect_height` | `0` | 推理高度，`0` = 保持原始分辨率 |
| `nms_iou` | `0.5` | NMS IoU 阈值，控制合并重叠框的严格程度 |

**`nms_iou` 调参指南：**
- `0.1` — 极严格，IoU > 0.1 的重叠框全部合并（可能导致站得近的两个人被合并）
- `0.3~0.5` — 推荐范围
- `0.7` — 很宽松，几乎不合并（同一个人可能保留多个框）

**`confidence` 与 `track_low_thresh` 的关系：**
```
detector.confidence ≤ tracker.track_low_thresh

如果 confidence > track_low_thresh：
  → 低分框根本没进 tracker，第二阶段匹配失效
```

### 跟踪器 (tracker)

```yaml
tracker:
  enabled: true               # true = 启用跟踪 | false = 仅检测（无ID）
  tracker_type: "bytetrack"   # "bytetrack" | "botsort"
```

**三种模式：**

| 模式 | 配置 | 效果 |
|------|------|------|
| 不使用跟踪 | `enabled: false` | 每帧独立检测，无 track_id，无行为连续性 |
| ByteTrack | `enabled: true`, `tracker_type: "bytetrack"` | 纯 IoU 匹配，速度快 |
| BoT-SORT | `enabled: true`, `tracker_type: "botsort"` | IoU + 运动 + 可选外观特征 |

#### 通用参数（bytetrack / botsort 共用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `track_high_thresh` | `0.5` | 高分框门槛。conf ≥ 此值的框参与第1轮匹配 |
| `track_low_thresh` | `0.1` | 低分框门槛。conf 在 [low, high) 之间参与第2轮匹配 |
| `match_thresh` | `0.8` | IoU 匹配阈值，IoU > 此值才算"同一个人" |
| `track_buffer` | `30` | 跟踪丢失后保留帧数，超过此帧数才释放 ID |

**`track_high_thresh` / `track_low_thresh` 两阶段匹配原理：**
```
当前帧所有检测框
    │
    ├─ conf ≥ track_high_thresh      → 第1轮：和已有轨迹匹配（高分框）
    │
    ├─ track_low_thresh ≤ conf < high → 第2轮：补充匹配（低分框）
    │
    └─ conf < track_low_thresh        → 丢弃
```

**`match_thresh` 调参指南：**
- `0.01~0.05` — 极宽松，相邻的人容易被合并成同一个 ID
- `0.3~0.5` — 推荐范围
- `0.8+` — 极严格，框稍微偏移就匹配失败，频繁分配新 ID

**`track_buffer` 调参指南：**
- `5` — 约 0.17 秒（30fps），被遮挡瞬间就丢 ID
- `30` — 约 1 秒，推荐值
- `60+` — 适合长时间遮挡场景

#### BoT-SORT 专有参数

```yaml
with_reid: false    # 是否启用 ReID 外观特征匹配
```

| 值 | 效果 |
|----|------|
| `false` | 纯运动 + IoU 匹配，不额外下载模型，速度快 |
| `true` | 启用 ReID，靠外观特征找回被遮挡目标，需要额外 GPU 显存 |

### 参数之间的依赖关系

```
detector.confidence ──→ 必须 ≤ tracker.track_low_thresh

detector.nms_iou    ──→ YOLO 内部 NMS，过滤重叠框
                         太小：误合并不同人
                         太大：同人多框，tracker 给每个框分配独立 ID

tracker.match_thresh ──→ 框与轨迹的 IoU 匹配门槛
                          太小：不同人被合并
                          太大：频繁换 ID

tracker.track_buffer ──→ 丢帧容忍度
                          太小：遮挡后换 ID
                          太大：真换人后旧 ID 久不释放
```

### 推荐配置

**高精度场景（泳池/仓库监控）：**
```yaml
detector:
  confidence: 0.3
  nms_iou: 0.3
tracker:
  enabled: true
  tracker_type: "bytetrack"
  track_high_thresh: 0.6
  track_low_thresh: 0.3
  match_thresh: 0.4
  track_buffer: 30
```

**高速场景（实时性优先）：**
```yaml
detector:
  confidence: 0.5
  nms_iou: 0.5
tracker:
  enabled: true
  tracker_type: "bytetrack"
  track_high_thresh: 0.5
  track_low_thresh: 0.2
  match_thresh: 0.5
  track_buffer: 15
```

**仅检测不需要跟踪：**
```yaml
tracker:
  enabled: false
```
