# auto_match.py — 팀원의 auto_match_v2.py(도커 컨테이너 /tmp에 있다가 마운트
# 사고로 유실됨) 알고리즘을 팀원이 준 정확한 수식대로 재구현.
#
# 목적: 852장 수동 라벨(Roboflow) 이상으로 학습 크롭을 늘리기 위해, YOLO
# 탐지기로 자동 탐지한 bbox를 train_label.json의 GT box_id에 자동 매칭.
#
# 알고리즘 (팀원 원본 설명 그대로):
#   1. 탐지된 박스 = y좌표(cy) 오름차순 정렬
#   2. GT = box_id 순 정렬 (box_id 자체가 이미 이 순서)
#   3. n_miss = n_gt - n_det. 0이면 순서대로 1:1.
#   4. n_miss > 0이면 GT 인덱스 중 n_det개를 고르는 모든 조합(combinations)에
#      대해: ratio_k = det[k] 픽셀폭 / gt[k] 실제 w(cm) 를 계산하고,
#      y좌표(cy) ~ ratio 에 대해 "직선"(선형회귀)을 피팅 (원근법: 카메라에
#      가까울수록 같은 실제 크기라도 픽셀상 크게 보이므로 ratio가 상수가
#      아니라 y에 선형으로 비례하는 게 정상 패턴 — 상수 가정이 아님에 주의).
#      상대 잔차(RMSE/mean)가 가장 작은 조합을 채택.
#   5. 채택된 조합의 잔차가 0.2 초과면 그 프레임 자체 폐기.
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# demo.zip 배치 기준: train_src/regressor/auto_match.py에서 두 단계 위가
# src/가 있는 패키지 루트 (원본 C_regressor_train/에서는 한 단계 위였음)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.detector.onnx_detector import OnnxDetector  # noqa: E402

RESIDUAL_THRESHOLD = 0.2
MAX_COMBOS = 50_000  # 조합 폭발 방지 (n_gt가 크고 n_det가 작을 때)
PAD_PX = 5           # RegressorGeometryEstimator와 동일 (학습/추론 크롭 일관성)
IMG_SIZE = 224
# 직선 피팅(2 파라미터)은 점 2개면 잔차가 항상 0 (퇴화) — n_det<=2에서는
# 어떤 조합이든 완벽히 맞아 매칭 검증이 불가능하고, n_det=3도 자유도 1이라
# 약함. 최소 4점을 요구해 무작위 오염 라벨 유입을 차단.
MIN_DET_FOR_FIT = 4
# 애매성 검사: 점이 적을 때(자유도 낮음)만 적용. 박스가 많은 프레임은 조합이
# 수십 개라 2위가 늘 근소하게 따라붙지만 1위가 맞는다는 게 수동 라벨 대조
# (89크롭 100% 일치)로 실증됨 — 거기에 margin을 걸면 전멸함.
AMBIGUITY_MARGIN = 1.3
AMBIGUITY_MAX_DET = 5      # n_det가 이 이하일 때만 margin 검사
# 영상 단위 전역 일관성 필터: 영상 전체 매칭 샘플로 ratio~y 직선을 재피팅,
# 개별 샘플의 상대 이탈이 이 값을 넘으면 그 샘플만 제거 (프레임 단위 검사보다
# 점이 수백 개라 통계적으로 강함).
# 0.15는 과잉이었음: 카메라가 비스듬해 픽셀 폭에 d 투영이 섞여 박스별로
# ±15~20% 체계적 편차가 정상 (수동 라벨 대조로 확인 — 0.15에서 걸러진 32개
# 중 14개가 사람 라벨과 일치하는 정상 매칭이었음). 오매칭은 다른 박스의
# w(11~43cm)가 붙어 이탈 50%+ → 0.30이면 정상은 통과, 오염만 차단.
GLOBAL_DEV_THRESHOLD = 0.30


def best_match(dets: list, gt_objects: list) -> tuple[float, tuple[int, ...] | None, str]:
    """dets: cy 오름차순 정렬된 Detection 리스트. gt_objects: box_id 순 GT 리스트.
    반환: (최소 잔차, 채택된 GT 인덱스 튜플, 실패사유) — 매칭 불가 시 combo=None.

    전 조합을 numpy로 한 번에 피팅 (조합마다 개별 lstsq 호출하던 최초 버전은
    n_gt가 큰 영상에서 영상당 14분까지 느려짐 — A=[ys,1]가 조합과 무관하게
    고정이라 의사역행렬을 한 번만 구해 전 조합에 동시 적용 가능)."""
    n_det, n_gt = len(dets), len(gt_objects)
    if n_det == 0 or n_det > n_gt:
        return float("inf"), None, "n_det_invalid"
    if n_det < MIN_DET_FOR_FIT:
        return float("inf"), None, "too_few_points"

    n_combos = 1
    for i in range(n_det):
        n_combos = n_combos * (n_gt - i) // (i + 1)
    if n_combos > MAX_COMBOS:
        return float("inf"), None, "combo_overflow"

    ys = np.array([d.bbox.cy for d in dets])                       # (n_det,)
    widths_px = np.array([d.bbox.width for d in dets])              # (n_det,)
    gt_w_all = np.array([o["size_cm"]["w"] for o in gt_objects])    # (n_gt,)

    combo_idx = np.array(list(itertools.combinations(range(n_gt), n_det)))  # (C, n_det)
    gt_w = gt_w_all[combo_idx]                                       # (C, n_det)
    valid = np.all(gt_w > 0, axis=1)
    if not valid.any():
        return float("inf"), None, "no_valid_combo"
    combo_idx, gt_w = combo_idx[valid], gt_w[valid]
    ratios = widths_px[None, :] / gt_w                               # (C, n_det)

    A = np.vstack([ys, np.ones(len(ys))]).T                          # (n_det, 2)
    pinv = np.linalg.pinv(A)                                         # (2, n_det)
    coefs = ratios @ pinv.T                                          # (C, 2)
    preds = coefs @ A.T                                              # (C, n_det)
    residuals = (np.sqrt(np.mean((ratios - preds) ** 2, axis=1))
                 / np.mean(ratios, axis=1))                          # (C,)

    order = np.argsort(residuals)
    best_i = order[0]
    best_residual = float(residuals[best_i])
    best_combo = tuple(int(x) for x in combo_idx[best_i])
    second_residual = float(residuals[order[1]]) if len(order) > 1 else float("inf")

    # 자유도가 낮은(점 적은) 프레임만 2위 조합과의 차이를 요구
    if n_det <= AMBIGUITY_MAX_DET and second_residual < best_residual * AMBIGUITY_MARGIN:
        return best_residual, None, "ambiguous"
    return best_residual, best_combo, ""


def process_video(
    video_path: Path,
    video_label: dict,
    detector: OnnxDetector,
    frame_stride: int,
    crops_dir: Path,
    log,
) -> list[dict]:
    video_id = video_label["video_id"]
    camera = video_label["camera"]
    gt_objects = video_label["objects"]  # box_id 순 (이미 정렬됨 가정)

    cap = cv2.VideoCapture(str(video_path))
    samples = []
    stats = {"frames": 0, "no_det": 0, "over_det": 0, "too_few": 0,
              "ambiguous": 0, "combo_overflow": 0, "residual_fail": 0, "matched": 0}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        stats["frames"] += 1
        fh, fw = frame.shape[:2]

        dets = detector.detect(frame)
        # 좌/우/상단 경계에 잘린 박스는 픽셀 폭이 실제보다 작게 재져 ratio를
        # 오염시키므로 매칭에서 제외 (하단 진입 중인 박스는 폭이 온전하고
        # y순서상 항상 마지막이라 유지 — 기존 수동 데이터셋도 하단컷 포함)
        dets = [d for d in dets
                if d.bbox.x1 > 1 and d.bbox.x2 < fw - 1 and d.bbox.y1 > 1]
        if not dets:
            stats["no_det"] += 1
            frame_idx += 1
            continue
        if len(dets) > len(gt_objects):
            stats["over_det"] += 1
            frame_idx += 1
            continue
        dets = sorted(dets, key=lambda d: d.bbox.cy)

        residual, combo, reason = best_match(dets, gt_objects)
        if combo is None:
            key = {"too_few_points": "too_few", "ambiguous": "ambiguous"}.get(
                reason, "combo_overflow")
            stats[key] += 1
            frame_idx += 1
            continue
        if residual > RESIDUAL_THRESHOLD:
            stats["residual_fail"] += 1
            frame_idx += 1
            continue

        stats["matched"] += 1
        for det, gi in zip(dets, combo):
            gt = gt_objects[gi]
            b = det.bbox
            x1 = max(0, int(b.x1) - PAD_PX)
            y1 = max(0, int(b.y1) - PAD_PX)
            x2 = min(fw, int(b.x2) + PAD_PX)
            y2 = min(fh, int(b.y2) + PAD_PX)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            crop_name = f"{video_id}_f{frame_idx:06d}_id{gt['box_id']}.png"
            ok, buf = cv2.imencode(".png", crop)
            if ok:
                buf.tofile(str(crops_dir / crop_name))

            samples.append({
                "crop": crop_name,
                "video": video_id,
                "frame": frame_idx,
                "box_id": gt["box_id"],
                "w": gt["size_cm"]["w"],
                "d": gt["size_cm"]["d"],
                "h": gt["size_cm"]["h"],
                "bbox_px": [float(b.x1), float(b.y1), float(b.width), float(b.height)],
                "bbox_ratio": [b.cx / fw, b.cy / fh, b.width / fw, b.height / fh],
                "frame_res": [fw, fh],
                "camera": camera,
                "match_residual": residual,
            })
        frame_idx += 1

    cap.release()

    # 영상 단위 전역 일관성 필터: 같은 영상은 카메라가 같으므로 y→배율 관계가
    # 하나의 직선을 이룸. 전체 매칭 샘플로 재피팅해 이탈 샘플만 제거.
    n_dropped = 0
    if len(samples) >= 8:
        ys = np.array([s["bbox_px"][1] + s["bbox_px"][3] / 2 for s in samples])
        ratios = np.array([s["bbox_px"][2] / s["w"] for s in samples])
        A = np.vstack([ys, np.ones(len(ys))]).T
        coef, *_ = np.linalg.lstsq(A, ratios, rcond=None)
        pred = A @ coef
        rel_dev = np.abs(ratios - pred) / np.maximum(pred, 1e-9)
        keep_mask = rel_dev <= GLOBAL_DEV_THRESHOLD
        n_dropped = int((~keep_mask).sum())
        samples = [s for s, k in zip(samples, keep_mask) if k]

    log(f"  {video_id}: frames={stats['frames']} matched={stats['matched']} "
        f"no_det={stats['no_det']} over_det={stats['over_det']} "
        f"too_few={stats['too_few']} ambiguous={stats['ambiguous']} "
        f"combo_overflow={stats['combo_overflow']} residual_fail={stats['residual_fail']} "
        f"global_dropped={n_dropped} kept={len(samples)}")
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--train_label", type=str, required=True)
    p.add_argument("--detector_path", type=str, default="checkpoints/detector_a.onnx")
    p.add_argument("--conf_thresh", type=float, default=0.4)
    p.add_argument("--nms_thresh", type=float, default=0.45)
    p.add_argument("--input_size", type=int, nargs=2, default=[1280, 1280])
    p.add_argument("--class_ids", type=int, nargs="+", default=[2])
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--limit_videos", type=int, default=None)
    p.add_argument("--videos", type=str, nargs="+", default=None,
                   help="특정 video_id만 처리 (예: train_000 train_009)")
    p.add_argument("--resume", action="store_true",
                   help="out_dir/samples_224.json에 이미 있는 영상은 건너뜀")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops_224"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_label, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    videos = label_data["videos"]
    if args.videos:
        wanted = set(args.videos)
        videos = [v for v in videos if v["video_id"] in wanted]
    if args.limit_videos:
        videos = videos[: args.limit_videos]

    all_samples = []
    done_videos = set()
    out_json = out_dir / "samples_224.json"
    if args.resume and out_json.is_file():
        with open(out_json, "r", encoding="utf-8") as f:
            all_samples = json.load(f)["samples"]
        done_videos = {s["video"] for s in all_samples}
        print(f"[resume] {len(done_videos)} videos already done, "
              f"{len(all_samples)} samples loaded", flush=True)
        videos = [v for v in videos if v["video_id"] not in done_videos]

    detector = OnnxDetector(
        model_path=args.detector_path,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        input_size=tuple(args.input_size),
        class_ids=args.class_ids,
    )
    detector.load()

    def log(msg):
        print(msg, flush=True)

    for v in videos:
        video_path = Path(args.video_dir) / f"{v['video_id']}.mp4"
        if not video_path.is_file():
            log(f"  SKIP {v['video_id']}: video file not found at {video_path}")
            continue
        samples = process_video(video_path, v, detector, args.frame_stride, crops_dir, log)
        all_samples.extend(samples)
        # 영상마다 즉시 저장 (Colab 연결 끊김 등으로 중간에 죽어도 그 시점까지는 보존)
        with open(out_dir / "samples_224.json", "w", encoding="utf-8") as f:
            json.dump({"samples": all_samples}, f, ensure_ascii=False, indent=2)

    log(f"DONE. total samples={len(all_samples)} -> {out_dir}")


if __name__ == "__main__":
    main()
