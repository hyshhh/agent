#!/usr/bin/env python3
"""
YOLOv8 微调脚本 — 独立运行

用法：
    # 使用 Ultralytics 默认 COCO 预训练权重微调
    python finetune_yolo.py --data https://example.com/dataset.yaml --epochs 50 --lr 0.001

    # 使用本地数据集
    python finetune_yolo.py --data /path/to/dataset.yaml --epochs 100 --lr 0.0005

    # 使用自定义预训练权重
    python finetune_yolo.py --data dataset.yaml --pretrained runs/detect/train/weights/best.pt

    # 指定 GPU
    python finetune_yolo.py --data dataset.yaml --device 0

    # 完整参数
    python finetune_yolo.py \
        --data dataset.yaml \
        --epochs 100 \
        --lr 0.001 \
        --batch 16 \
        --imgsz 640 \
        --device 0 \
        --output runs/finetune

数据集格式 (YOLO 格式)：
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml

    dataset.yaml 内容：
        path: ./dataset
        train: train/images
        val: val/images
        names:
          0: person
          1: drowning
          ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 微调脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --data dataset.yaml --epochs 50 --lr 0.001
  %(prog)s --data dataset.yaml --pretrained best.pt --epochs 100
  %(prog)s --data https://example.com/data.yaml --device 0
        """,
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="数据集配置文件路径或 URL（YOLO 格式的 .yaml）",
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="训练轮数 (默认: 50)",
    )

    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=0.001,
        help="初始学习率 (默认: 0.001)",
    )

    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="批次大小 (默认: 16, -1=自动)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="训练图像尺寸 (默认: 640)",
    )

    parser.add_argument(
        "--pretrained", "-p",
        type=str,
        default="yolov8n.pt",
        help="预训练权重路径 (默认: yolov8n.pt，自动下载 COCO 权重)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="训练设备: '0' / '0,1' / 'cpu' (默认: 自动检测)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="runs/finetune",
        help="输出目录 (默认: runs/finetune)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="数据加载线程数 (默认: 8)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="早停耐心值: 连续 N 轮无改善则停止 (默认: 20, 0=禁用)",
    )

    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="冻结前 N 层 (默认: 0=不冻结, 10=冻结 backbone)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续训练",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8 微调训练")
    print("=" * 60)
    print(f"  数据集:       {args.data}")
    print(f"  预训练权重:   {args.pretrained}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  学习率:       {args.lr}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Image size:   {args.imgsz}")
    print(f"  Device:       {args.device or 'auto'}")
    print(f"  冻结层数:     {args.freeze}")
    print(f"  早停耐心:     {args.patience}")
    print(f"  输出目录:     {args.output}")
    print("=" * 60)

    # 加载模型
    print(f"\n加载预训练模型: {args.pretrained}")
    model = YOLO(args.pretrained)

    # 训练参数（已预调好，用户只需设置 data/lr/epochs）
    train_kwargs = {
        # 数据
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,

        # 训练
        "epochs": args.epochs,
        "lr0": args.lr,               # 初始学习率
        "lrf": 0.01,                  # 最终学习率 = lr0 * lrf
        "warmup_epochs": 3,           # warmup 轮数
        "warmup_bias_lr": 0.1,        # warmup 初始 lr
        "warmup_momentum": 0.8,       # warmup 初始 momentum

        # 优化器
        "optimizer": "AdamW",          # AdamW 效果通常优于 SGD
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # 数据增强（适合微调，不过度增强）
        "mosaic": 0.5,                # 适度 mosaic
        "mixup": 0.1,                 # 轻度 mixup
        "copy_paste": 0.1,            # 复制粘贴增强
        "degrees": 10.0,              # 旋转 ±10°
        "translate": 0.1,             # 平移 10%
        "scale": 0.5,                 # 缩放 50%
        "shear": 2.0,                 # 剪切 ±2°
        "flipud": 0.5,                # 上下翻转
        "fliplr": 0.5,                # 左右翻转
        "hsv_h": 0.015,               # 色调增强
        "hsv_s": 0.7,                 # 饱和度增强
        "hsv_v": 0.4,                 # 亮度增强

        # 正则化
        "dropout": 0.1,               # dropout 防过拟合

        # 早停
        "patience": args.patience,

        # 冻结
        "freeze": [list(range(args.freeze))] if args.freeze > 0 else None,

        # 输出
        "project": args.output,
        "name": f"finetune_{time.strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": False,
        "save": True,
        "save_period": 10,            # 每 10 轮保存一次

        # 设备
        "device": args.device or None,

        # 其他
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "plots": True,                # 生成训练曲线图
    }

    # 移除 None 值
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    # 开始训练
    print("\n开始训练...\n")
    start = time.time()

    try:
        results = model.train(**train_kwargs, resume=args.resume)
    except KeyboardInterrupt:
        print("\n\n训练被中断。使用 --resume 参数可继续训练。")
        sys.exit(1)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"  耗时: {elapsed / 60:.1f} 分钟")
    print(f"  结果目录: {results.save_dir}")
    print(f"  最佳权重: {results.save_dir}/weights/best.pt")
    print(f"  最终权重: {results.save_dir}/weights/last.pt")
    print("=" * 60)

    # 验证
    print("\n在验证集上评估最佳权重...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    val_results = best_model.val(data=args.data, device=args.device or None)

    print(f"\n验证结果:")
    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision:{val_results.box.mp:.4f}")
    print(f"  Recall:   {val_results.box.mr:.4f}")

    print(f"\n使用微调模型:")
    print(f"  在 config.yaml 中设置: detector.model: \"{results.save_dir}/weights/best.pt\"")


if __name__ == "__main__":
    main()
