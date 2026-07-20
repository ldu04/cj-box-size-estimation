# CJ Logistics — CCTV 박스 크기 추정 챌린지

컨베이어 벨트 위 물류 박스를 CCTV 영상에서 감지·추적하여 W×D×H(cm) 크기를
추정하고 `result.json`을 생성하는 대회 제출용 파이프라인입니다.

## 실행 방법

```bash
# 대회 규격 진입점
python main.py --input /data/test_videos

# 옵션
python main.py --input /data/test_videos \
               --output result.json \
               --config configs/default.yaml
```

`--input` 폴더의 모든 `.mp4`를 처리하고, `result.json`은 `main.py`가 있는
디렉터리에 생성됩니다.

## 추론 구성 (configs/default.yaml)

영상 프레임별로 아래 단계를 거쳐 박스 크기를 추정합니다:

1. **탐지** — `checkpoints/detector_b.onnx` (YOLO11s fine-tune, 2-class,
   conf 0.6): 프레임에서 박스 bbox 검출. ByteTrack식 2단계 confidence
   (0.15~0.6 구간 탐지는 기존 트랙 연장에만 사용)
2. **추적** — SORT (Kalman + Hungarian, filterpy/scipy): 프레임 간 동일
   박스 연결
3. **트랙 필터링** — TrackAggregator 출생지(origin) 필터: 화면 하단
   진입선/영상 시작 시점에 존재하지 않는 중간 출현 트랙은 유령으로 제거
4. **크기 회귀** — `checkpoints/regressor.onnx` (CNN): 트랙별 최대 5개
   크롭의 예측 중앙값. focal은 `checkpoints/focalnet.onnx`가 프레임에서
   추정. 100cm 초과 비현실 예측은 탐지 오탐으로 간주해 제거
5. **출력** — 영상별 objects 목록을 `result.json`으로 저장 (`size_cm`의
   w/d/h만 포함)

모든 컴포넌트는 ABC 기반 플러그인 구조로, `configs/default.yaml`의 `type`
필드로 교체 가능합니다.

## 출력 형식

```json
{
    "videos": [
        {
            "video_id": "test_000",
            "objects": [
                {"size_cm": {"w": 43.0, "d": 26.1, "h": 22.5}},
                {"size_cm": {"w": 30.0, "d": 20.0, "h": 15.0}}
            ]
        }
    ]
}
```

## 제출 패키지 구조

```
demo.zip
├── main.py                          # 진입점 (--input)
├── README.md
├── configs/
│   └── default.yaml                 # 제출 추론 설정
├── checkpoints/                     # ONNX 모델 (opset 17, IR 8)
│   ├── detector_b.onnx              # 탐지기 (제출 사용)
│   ├── regressor.onnx               # 크기 회귀 (제출 사용)
│   ├── focalnet.onnx                # focal 추정 (제출 사용)
│   └── detector_a.onnx              # 이전 탐지기 (참고 보존, 미사용)
├── src/                             # 추론 소스
│   ├── pipeline.py                  # VideoProcessor / Pipeline
│   ├── schema.py                    # BBox, Detection, Track, CompetitionResult
│   ├── detector/onnx_detector.py    # ONNX YOLO 디코딩 + NMS (numpy)
│   ├── tracker/sort_tracker.py      # SORT (filterpy/scipy)
│   ├── aggregator/track_aggregator.py # 트랙 누적·출생지 필터
│   ├── geometry/regressor.py        # 크기 회귀 + focal 추정 + 이상치 클램프
│   ├── calibration/ …               # 캘리브레이터 (기본 dummy)
│   ├── io/ …                        # VideoReader / ResultWriter
│   └── validator/result_validator.py # result.json 스키마 검증
└── train_src/                       # 학습 코드 (README·requirements 포함)
    ├── yolo/                        # 탐지기 학습·fine-tune·ONNX 변환
    └── regressor/                   # auto_match 데이터 생성 + 회귀 CNN 학습
```

학습 절차 상세는 `train_src/README.md` 참고.

## 실행 환경 (대회 제공 Docker 기준)

| 항목 | 사양 |
|------|------|
| Python | 3.11 |
| CUDA | 12.1 / cuDNN 9 |
| GPU | NVIDIA A100 40GB |
| ONNX Runtime | 1.20.1 (GPU) |
| OpenCV | 4.10.0 (contrib, headless) |

추론은 제공 Docker에 포함된 패키지(onnxruntime, cv2, numpy, filterpy,
scipy)만 사용합니다. 네트워크 접근·외부 다운로드 없음.
