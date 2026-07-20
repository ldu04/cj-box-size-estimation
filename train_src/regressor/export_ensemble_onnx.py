# export_ensemble_onnx.py — F-const seed 앙상블을 단일 ONNX로 변환.
#
# 3개 체크포인트를 내부에 품고 로그 공간에서 통합(median, 실패 시 mean)하는
# 모듈을 opset17 dynamic-batch ONNX로 export한 뒤, torch 출력과의 일치를 검증한다.
#
# 사용법:
#   python export_ensemble_onnx.py \
#       --checkpoints ../checkpoints/regressor_fconst.pt \
#                     ../checkpoints/regressor_fconst_s43.pt \
#                     ../checkpoints/regressor_fconst_s44.pt \
#       --out ../checkpoints/regressor.onnx
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from model import Regressor


class Ensemble(nn.Module):
    def __init__(self, models, mode: str):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.mode = mode

    def forward(self, images, metas):
        outs = torch.stack([m(images, metas) for m in self.models])
        if self.mode == "median":
            return outs.median(dim=0).values
        return outs.mean(dim=0)


def load(p: str) -> Regressor:
    m = Regressor()
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def try_export(ens: nn.Module, out: str) -> bool:
    try:
        torch.onnx.export(
            ens, (torch.zeros(1, 3, 128, 128), torch.zeros(1, 4)), out,
            input_names=["images", "metas"], output_names=["log_cm"],
            dynamic_axes={"images": {0: "b"}, "metas": {0: "b"}, "log_cm": {0: "b"}},
            opset_version=17, dynamo=False,
        )
        return True
    except Exception as e:
        print(f"export failed ({ens.mode}): {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = [load(p) for p in args.checkpoints]
    print(f"loaded {len(models)} models")

    ens = None
    for mode in ("median", "mean"):
        cand = Ensemble([load(p) for p in args.checkpoints], mode)
        cand.eval()
        if try_export(cand, args.out):
            ens = cand
            print(f"exported with mode={mode} -> {args.out}")
            break
    if ens is None:
        print("ERROR: both median and mean export failed", file=sys.stderr)
        return 1

    import onnx
    import onnxruntime as ort

    mo = onnx.load(args.out)
    print("ir_version:", mo.ir_version,
          "opset:", {o.domain or "ai.onnx": o.version for o in mo.opset_import})
    assert mo.ir_version <= 10, "IR>10은 평가 환경 ORT 1.20.1이 로드 불가"

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for bs in (1, 5):
        g = torch.Generator().manual_seed(bs)
        img = torch.rand(bs, 3, 128, 128, generator=g)
        meta = torch.rand(bs, 4, generator=g)
        with torch.no_grad():
            t_out = ens(img, meta).numpy()
        o_out = sess.run(None, {"images": img.numpy(), "metas": meta.numpy()})[0]
        diff = float(np.abs(t_out - o_out).max())
        print(f"batch={bs}: torch vs onnx max diff = {diff:.2e}")
        assert diff < 1e-4
    print("verification OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
