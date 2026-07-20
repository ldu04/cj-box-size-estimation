# 학습 코드 (train_src)

제출 모델 3개(detector, regressor, focalnet — 전부 `checkpoints/`의 ONNX)의
학습 절차와 코드입니다. 학습 환경 의존성은 `requirements.txt` 참고
(추론은 제공 Docker 환경만 사용 — 학습 환경과 별개).

## 구성

```
train_src/
├── yolo/                          # 탐지기(YOLO11s) 학습·변환
│   ├── _train_yolo.sh              # (1차) 학습→export→검증 공용 파이프라인
│   ├── train_yolo11s.sh            # (1차) YOLO11s 실행 래퍼
│   ├── colab_finetune_yolo.py      # (2차, 제출 detector) 체크포인트 이어학습 fine-tune
│   └── export_onnx.py              # best.pt → ONNX(opset17, imgsz=1280) 변환·검증
└── regressor/                     # 크기 회귀 CNN 학습
    ├── model.py                    # 모델 구조 (Conv×4 + FC, 128px 입력)
    ├── auto_match.py               # (제출 모델용) 학습 데이터 자동 생성 — 아래 2-1 참고
    ├── dataset224.py               # auto_match 크롭 로더 + 라벨-안전 증강 + 그룹 분할
    ├── train_baseline_groupsplit.py # (제출 모델) 순열불변·cm공간 손실 학습 스크립트
    ├── train_backbone.py           # train_baseline_groupsplit가 재사용하는 학습 루프
    ├── model_backbone.py           # train_backbone 의존 모듈
    ├── export_single_onnx.py       # 단일 체크포인트 → ONNX 변환·출력 일치 검증
    ├── train_focal.py              # FocalNet 학습
    ├── dataset.py, train.py, train_variant.py, export_ensemble_onnx.py
    │                               # (구버전) 초기 수동크롭·3-seed 앙상블 파이프라인 — 기록용
    └── ...
```

## 1. 탐지기 — 제출 모델: `checkpoints/detector_b.onnx`

2단계로 학습했습니다.

**1차 (기반 체크포인트):**
- 사전학습 모델: ultralytics `yolo11s.pt`
- 데이터: 제공 train 영상 100개에서 추출한 프레임, 수작업 레이블링 (YOLO 형식)
- 학습: `bash yolo/train_yolo11s.sh` (imgsz=1280, epochs=100, batch=32, AdamW, A100 1장)
- 최종 지표: mAP50 0.920 / precision 0.905 / recall 0.872

**2차 (제출 모델 detector_b):**
- 1차 체크포인트에서 이어서, 수작업 레이블링 852장(2-class: '3', 'box')으로 fine-tune
- 학습: `python yolo/colab_finetune_yolo.py --weights {1차 best.pt} --data {data.yaml} --epochs 60 --batch 16`
- 변환: `python yolo/export_onnx.py {best.pt} --out detector_b.onnx` (opset17, imgsz=1280)
- 추론 시 `configs/default.yaml`의 `class_ids: [1]`('box'), `conf_thresh: 0.6`과 세트
  (conf_thresh는 detector별로 재튜닝 필수 — detector_a는 0.4였음)

## 2. 크기 회귀 CNN — 제출 모델: `checkpoints/regressor.onnx`

### 2-1. 학습 데이터 생성 (auto_match)

수동 라벨 크롭(초기 3,418장)만으로는 부족해서, 학습된 탐지기로 train 영상
100개를 자동 탐지하고 각 bbox를 `train_label.json`의 GT box에 자동 매칭해
크롭 약 30,000장(매칭 잔차 필터 통과분)으로 확장:

```bash
python regressor/auto_match.py \
  --video_dir {train_videos_dir} --train_label {train_label.json} \
  --detector_path checkpoints/detector_a.onnx --out_dir {out_dir}
```

매칭 원리: 같은 영상에서 `픽셀폭/실제폭` 비율은 y좌표(원근)에 선형 —
탐지 bbox와 GT 조합 전체에 이 직선을 피팅해 잔차 최소 조합 채택, 잔차
초과 프레임 폐기, 영상 단위 전역 일관성 필터로 오염 라벨 제거.

### 2-2. 학습 (순열불변·cm공간 손실)

공식 채점이 (a) 절대 cm 오차, (b) 축 순열 자유이므로 손실도 동일하게:
SmoothL1을 log가 아닌 cm 공간에서, w/d/h 6가지 순열 중 최소값으로 계산.

```bash
python regressor/train_baseline_groupsplit.py \
  --root_dir {auto_match out_dir} \
  --perm_invariant_loss --cm_space_loss \
  --epochs 40 --seed 42 --split_seed 42 --ckpt_name regressor_permloss.pt
python regressor/export_single_onnx.py \
  --checkpoint regressor_permloss.pt --out regressor.onnx
```

- 분할: (video, box_id) 그룹 단위 85/15 (같은 박스의 크롭이 train/val에 갈리지 않도록)
- 증강: 좌우반전, 밝기/대비 지터, focal 지터(±3%) — 전부 라벨 불변인 것만
- 검증 성적: held-out 박스단위(그룹 중앙값, 순열매칭) MAE 4.10cm
- 참고: 순열불변 손실로 학습한 모델은 출력 축 순서가 (w,d,h)에 고정되지
  않을 수 있음 — 채점이 순열자유라 무해하나, 평가 시에도 순열매칭으로 잴 것

(구버전 기록: 초기 제출은 수동크롭 3,418장 + 3-seed 로그공간 앙상블이었음
— `train.py`/`train_variant.py`/`export_ensemble_onnx.py`로 재현 가능)

## 3. FocalNet (focal 추정기) — 제출 모델: `checkpoints/focalnet.onnx`

- 데이터: 제공 train 영상 100개 × 3프레임(20/50/80% 지점, 320×180 다운샘플)
  + train_label.json의 focal_length_mm
- 분할: 영상 단위 85/15 (같은 영상의 프레임이 train/val에 갈리지 않도록)
- 학습: `python regressor/train_focal.py` (120 epochs, log(focal) 타깃, SmoothL1)
- 검증 성적: 미학습 영상 focal MAE 0.84mm (상대 6.4%)
- ONNX 변환: torch.onnx.export (opset 17, dynamic batch) 후 출력 일치 검증

## 환경

`requirements.txt` 참고 (Python 3.11+, PyTorch 2.x, ultralytics, onnx).
