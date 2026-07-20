# export_single_onnx.py — 단일 Regressor 체크포인트(.pt)를 ONNX로 변환.
# export_ensemble_onnx.py의 앙상블 래퍼 없이, 순열불변손실 등으로 학습한
# 단일 모델(regressor_permloss.pt 등)을 그대로 export할 때 사용.
from __future__ import annotations

import argparse

import torch

from model import Regressor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Regressor()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    torch.onnx.export(
        model, (torch.zeros(1, 3, 128, 128), torch.zeros(1, 4)), args.out,
        input_names=["images", "metas"], output_names=["log_cm"],
        dynamic_axes={"images": {0: "b"}, "metas": {0: "b"}, "log_cm": {0: "b"}},
        opset_version=17, dynamo=False,
    )

    # torch 출력과 ONNX 출력 일치 검증
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    imgs = torch.randn(3, 3, 128, 128)
    metas = torch.randn(3, 4)
    with torch.no_grad():
        torch_out = model(imgs, metas).numpy()
    onnx_out = sess.run(None, {"images": imgs.numpy(), "metas": metas.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"export 완료: {args.out}")
    print(f"torch vs onnx 최대오차: {max_diff:.6e} ({'OK' if max_diff < 1e-4 else 'WARNING: 오차 큼'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
