"""
사전학습 백본 기반 회귀 모델 (실험).

1차 시도(MobileNetV3-Small, trunk 동결)는 데이터 885박스 시절 실패했음
(train_backbone.py 로그: best 6.77cm, 이후 과적합).

2차 시도: 데이터가 8배(30,403샘플, 915박스)로 늘어난 지금은 과적합 위험이
낮아져 "전체 fine-tune"(freeze_backbone=False)을 재시도할 근거가 생김.
ResNet18 기본 채택(적당한 크기, 검증된 전이학습 성능).

기존 model.py(from-scratch 4-conv CNN)와의 차이:
- ImageNet 사전학습 특징 사용
- 입력 전처리가 다름: BGR/255만 쓰는 기존과 달리 RGB + ImageNet mean/std 정규화
- 원본 224 크롭(dataset224.py, image_size=224)을 그대로 사용 (업샘플 없음)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_for_backbone(img_bgr_01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) BGR/255 텐서 -> 백본 입력 형식.

    RGB 변환 + ImageNet 정규화. 224가 아니면 업샘플하지만, dataset224.py로
    원본 224 크롭을 직접 먹이는 것이 정석 (128 축소본을 업샘플하면 화질 손실).
    """
    rgb = img_bgr_01.flip(1)  # BGR -> RGB (채널 순서 뒤집기)
    if rgb.shape[-1] != 224:
        rgb = F.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


_BACKBONES = {
    "mobilenet_v3_small": lambda: (
        torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        ).features,
        576,
    ),
    "resnet18": lambda: (
        nn.Sequential(*list(
            torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            ).children()
        )[:-1]),  # 마지막 fc 제거, avgpool까지 (이미 (B,512,1,1) 출력)
        512,
    ),
}


class RegressorBackbone(nn.Module):
    def __init__(self, model_name: str = "resnet18", freeze_backbone: bool = False) -> None:
        super().__init__()
        if model_name not in _BACKBONES:
            raise ValueError(f"unknown model_name: {model_name}")
        self.model_name = model_name
        self.features, feat_dim = _BACKBONES[model_name]()
        self.pool = nn.AdaptiveAvgPool2d(1) if model_name != "resnet18" else nn.Identity()

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.features[-1].parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3),
        )

    def forward(self, img_bgr_01: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = preprocess_for_backbone(img_bgr_01)
        feat = self.pool(self.features(x)).flatten(1)
        return self.head(torch.cat([feat, meta], dim=1))

    def param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """동결 안 된(전체 fine-tune) 파라미터는 낮은 lr, head는 높은 lr."""
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]


def build_model(model_name: str = "resnet18", freeze_backbone: bool = False) -> RegressorBackbone:
    return RegressorBackbone(model_name=model_name, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    for name in ("resnet18", "mobilenet_v3_small"):
        model = build_model(name, freeze_backbone=False)
        model.eval()
        dummy_img = torch.rand(2, 3, 224, 224)
        dummy_meta = torch.randn(2, 4)
        out = model(dummy_img, dummy_meta)
        assert out.shape == (2, 3)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{name}: trainable={trainable:,} / total={total:,} OK")
