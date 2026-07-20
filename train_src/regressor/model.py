"""
model.py -- B 파이프라인(pipeline_dongwook.py)의 Regressor 클래스와
100% 동일하게 유지해야 하는 파일.

절대 임의로 레이어를 바꾸지 마세요. 여기서 한 글자라도 다르면
학습된 state_dict를 B 파이프라인의 Regressor가 load_state_dict()할 때
크기 불일치(RuntimeError)로 바로 깨집니다.

출처: pipeline_dongwook.py의 class Regressor(nn.Module) 그대로 복사.
"""

import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(132, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, img, meta):
        return self.head(torch.cat([self.cnn(img), meta], dim=1))


def build_model():
    return Regressor()


if __name__ == "__main__":
    # 구조 검증: B 코드와 동일한 입력 shape으로 forward 확인
    model = Regressor()
    model.eval()
    dummy_img = torch.randn(2, 3, 128, 128)   # B 코드: cv2.resize(crop,(128,128))
    dummy_meta = torch.randn(2, 4)             # B 코드: 4차원 metadata
    out = model(dummy_img, dummy_meta)
    print("output shape:", out.shape)  # (2, 3) 이어야 함
    assert out.shape == (2, 3)
    print("model.py 구조 검증 OK (132차원 결합 확인됨)")
