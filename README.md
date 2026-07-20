# 📦 CCTV 영상 기반 화물 크기 추정 시스템

> **단일 CCTV 영상에서 컨베이어 위 박스의 W×D×H(cm)를 자동 추정**  
> CJ대한통운 미래기술챌린지 2026 — CCTV 영상 기반 화물 객체 분석

**Public 리더보드: 47팀 중 9위**

---

## 프로젝트 소개

트럭 적재율을 높이려면 상자 하나하나의 실제 크기를 알아야 한다.
하지만 물류 현장의 CCTV는 단일 시점 영상 한 대뿐이고, 조명 변화·가림·다양한 상자 크기까지 더해져 정확한 3D 크기 추정이 어렵다.

이 프로젝트는 컨베이어 레일을 따라 이동하는 박스를 탐지·추적하고, CNN 회귀 모델로 개별 상자의 **가로(w)·세로(d)·높이(h)를 cm 단위로 추정**하는 엔드투엔드 파이프라인이다.

- 학습 데이터: Train 영상 100개 (Synthetic CCTV)
- 평가 지표: MAE (낮을수록 우수)
- 제약: 단일 카메라 · 외부 API·다운로드 금지 · ONNX 모델만 허용

---

## 파이프라인 구조

```
입력 영상 (.mp4)
    │
    ▼
① 탐지 (YOLO11s fine-tune → detector_b.onnx)
    │  conf 0.6 이상 bbox 검출 / 0.15~0.6 구간은 기존 트랙 연장에만 사용
    ▼
② 추적 (SORT — Kalman Filter + Hungarian Algorithm)
    │  프레임 간 동일 박스 ID 연결
    ▼
③ 트랙 필터링 (TrackAggregator)
    │  화면 중간에 갑자기 출현한 유령 트랙 제거
    │  출생지(origin) 필터: 하단 진입선 또는 영상 시작 시점만 허용
    ▼
④ 크기 회귀 (CNN → regressor.onnx + focalnet.onnx)
    │  트랙별 최대 5개 크롭의 예측 중앙값
    │  FocalNet으로 영상별 focal length 추정 → 픽셀-cm 변환 보정
    │  100cm 초과 비현실 예측 자동 제거
    ▼
출력 result.json (video_id별 objects 목록)
```

---

## 모델 구성

### 1. 탐지기 — `detector_b.onnx`

2단계 학습으로 구성된 YOLO11s 기반 탐지기다.

| 단계 | 방법 | 데이터 | 주요 지표 |
|------|------|--------|---------|
| 1차 | `yolo11s.pt` fine-tune | Train 영상 100개 수작업 라벨링 | mAP50 0.920 / Precision 0.905 / Recall 0.872 |
| 2차 (제출) | 1차 체크포인트 이어학습 | 수작업 라벨 852장 (2-class: box, 3) | conf 0.6 기준 최적화 |

- 입력 해상도: 1280×1280
- 변환: opset 17, ONNX IR 8

### 2. 크기 회귀 CNN — `regressor.onnx`

#### 학습 데이터 자동 생성 (auto_match)

수동 라벨 크롭 3,418장만으론 부족해 `auto_match.py`로 확장했다.

```
학습된 탐지기 → train 영상 100개 자동 탐지
    → 탐지 bbox × GT box 조합에 픽셀폭/실제폭 직선 피팅
    → 잔차 최소 조합 채택 + 오염 라벨 영상 단위 필터
    → 크롭 약 30,000장 확보
```

#### 손실 함수 설계

채점 방식이 (a) 절대 cm 오차, (b) 축 순열 자유이므로 손실도 동일하게 설계했다.

> SmoothL1을 **cm 공간**에서, **w/d/h 6가지 순열 중 최솟값**으로 계산

- 그룹 분할: (video, box_id) 단위 85/15 — 같은 박스의 크롭이 train/val에 갈리지 않도록
- 증강: 좌우반전·밝기/대비 지터·focal 지터(±3%) — 전부 라벨 불변 변환만 적용
- **Validation MAE: 4.10cm** (held-out 박스 단위, 순열매칭)

### 3. FocalNet — `focalnet.onnx`

영상별 카메라 focal length를 추정해 픽셀-cm 변환 정확도를 높인다.

- 입력: 영상 3프레임(20/50/80% 지점, 320×180 다운샘플)
- 타깃: `log(focal_length_mm)`, SmoothL1
- **검증 성적: focal MAE 0.84mm (상대 오차 6.4%)**

---

## 결과

| 구분 | 순위 |
|------|------|
| Public 리더보드 (50개 영상) | **47팀 중 9위** |
| 최종 평가 (150개 영상) | **39팀 중 6위** |

---

## 실행 방법

```bash
# 추론 실행
python main.py --input /data/test_videos

# 옵션 지정
python main.py --input /data/test_videos \
               --output result.json \
               --config configs/default.yaml
```

`--input` 폴더의 모든 `.mp4`를 처리하고, `result.json`은 `main.py`가 있는 디렉터리에 생성된다.

### 실행 환경

| 항목 | 사양 |
|------|------|
| Python | 3.11 |
| CUDA | 12.1 / cuDNN 9 |
| GPU | NVIDIA A100 40GB |
| ONNX Runtime | 1.20.1 (GPU) |
| OpenCV | 4.10.0 (contrib, headless) |

추론은 제공 Docker에 포함된 패키지만 사용한다. 네트워크 접근·외부 다운로드 없음.

---

## 프로젝트 구조

```
├── main.py                          # 진입점 (--input)
├── configs/default.yaml             # 추론 설정
├── checkpoints/
│   ├── detector_b.onnx              # 탐지기 (제출)
│   ├── regressor.onnx               # 크기 회귀 (제출)
│   ├── focalnet.onnx                # focal 추정 (제출)
│   └── detector_a.onnx              # 이전 탐지기 (기록용)
├── src/
│   ├── pipeline.py                  # VideoProcessor / Pipeline
│   ├── schema.py                    # BBox, Detection, Track, CompetitionResult
│   ├── detector/                    # ONNX YOLO 디코딩 + NMS
│   ├── tracker/                     # SORT (Kalman + Hungarian)
│   ├── aggregator/                  # 트랙 누적·출생지 필터
│   ├── geometry/                    # 크기 회귀 + focal 추정 + 이상치 클램프
│   ├── calibration/                 # 캘리브레이터
│   ├── io/                          # VideoReader / ResultWriter
│   └── validator/                   # result.json 스키마 검증
└── train_src/
    ├── yolo/                        # YOLO11s 학습·fine-tune·ONNX 변환
    └── regressor/                   # auto_match 데이터 생성 + 회귀 CNN 학습
```

학습 절차 상세는 `train_src/README.md` 참고.

---

## 한계 및 향후 발전 방향

- **단일 카메라 한계**: 깊이 정보 없이 픽셀-cm 변환에 의존 → 스테레오 카메라 또는 depth sensor 병행 시 정확도 향상 가능
- **가림(Occlusion) 처리**: 박스끼리 겹치는 구간에서 트랙 단절 발생
- **일반화**: Synthetic 데이터 학습 → 실제 물류 현장 도메인 갭 존재
- [ ] 실제 물류 현장 데이터로 도메인 적응(Domain Adaptation)
- [ ] 멀티 카메라 뷰 퓨전으로 depth 추정 정확도 개선
- [ ] 트랜스포머 기반 탐지기(DETR 계열)로 교체 실험
