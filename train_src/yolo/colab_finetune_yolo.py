#!/usr/bin/env python3
"""Colab GPU에서 YOLO11s를 기존 체크포인트로부터 이어서(fine-tune) 학습.

사용법 (Colab):
    !python colab_finetune_yolo.py \
        --weights /content/yolo11s_best.pt \
        --data /content/yolo_train_data/data.yaml \
        --epochs 60 --batch 16

주의: data.yaml의 클래스 수가 기존 체크포인트(nc=3)와 다르면(이 로컬
데이터셋은 nc=2) ultralytics가 분류 헤드를 새로 초기화한다 — 정상 동작.
학습 후 반드시 model.names를 확인해 configs/default.yaml의
detector.class_ids를 'box' 클래스 인덱스로 맞출 것.
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="/content/runs")
    ap.add_argument("--name", default="yolo11s_ft")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        optimizer="AdamW", lr0=0.001, weight_decay=0.0005,
        patience=15, device=0, workers=8,
        cache=True, amp=True, cos_lr=True, seed=42,
        project=args.project, name=args.name, exist_ok=True,
    )

    best = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\n=== 학습 완료: {best} ===")
    m = YOLO(best)
    print("클래스 매핑 (configs/default.yaml의 class_ids를 여기 맞춰 갱신):")
    print(" ", m.names)


if __name__ == "__main__":
    main()
