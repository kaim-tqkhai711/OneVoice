"""band_feats.onnx: STFT band energies as an ONNX graph built with onnx.helper (no torch in the artifact), per docs/BRANCH_B_DESIGN.md section 3.

    audio [1,1,T] -> Pad(256 left, right pad so that N = T//256 frames fit) -> Conv(W [514,1,512], stride 256) -> [1,514,N]
    -> re^2 + im^2 -> power [1,257,N] -> bands:
       out[:, k] for k in 0..3 = sum of power over the band mask (E_50-1000, E_1000-5000, E_0-500, E_500-1500)
       out[:, 4..5]           = max of power over the mask (Pmax_0-2000, Pmax_2000-5000)
    -> [N, 6]
Frame i covers samples [256(i-1), 256(i+1)) of the input: the same alignment as SwiftF0's loudness window. Periodic Hann, n_fft 512, hop 256.
Band edges are applied as [lo, hi) on bin centre frequencies k*16000/512.
"""
from __future__ import annotations

import numpy as np

N_FFT, HOP, SR = 512, 256, 16000
BANDS_SUM = [(50.0, 1000.0), (1000.0, 5000.0), (0.0, 500.0), (500.0, 1500.0)]
BANDS_MAX = [(0.0, 2000.0), (2000.0, 5000.0)]


def band_masks() -> np.ndarray:  # -> [257, 6] float32 0/1
    f = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    m = np.zeros((N_FFT // 2 + 1, 6), np.float32)
    for k, (lo, hi) in enumerate(BANDS_SUM + BANDS_MAX):
        m[:, k] = ((f >= lo) & (f < hi)).astype(np.float32)
    return m


def dft_kernel() -> np.ndarray:  # -> [514, 1, 512] : rows 0..256 = cos part, rows 257..513 = -sin part, each times periodic Hann
    n = np.arange(N_FFT)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)  # periodic Hann
    k = np.arange(N_FFT // 2 + 1)[:, None]
    ang = 2 * np.pi * k * n[None, :] / N_FFT
    return np.concatenate([np.cos(ang) * w, -np.sin(ang) * w], axis=0)[:, None, :].astype(np.float32)


def build(path: str) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W = numpy_helper.from_array(dft_kernel(), "dft_w")
    M = band_masks()
    Msum = numpy_helper.from_array(M[:, :4].copy(), "mask_sum")  # [257, 4]
    Mmax = numpy_helper.from_array(M[:, 4:].T.copy()[:, :, None], "mask_max")  # [2, 257, 1] broadcast over frames
    # Pad pads = [b0,b1,b2,e0,e1,e2] = [0,0,256, 0,0,rpad]
    g_nodes = [
        helper.make_node("Shape", ["audio"], ["shp"]),
        helper.make_node("Gather", ["shp", "idx2"], ["T"], axis=0),
        helper.make_node("Div", ["T", "hop"], ["N"]),
        helper.make_node("Mul", ["N", "hop"], ["NH"]),
        helper.make_node("Sub", ["NH", "T"], ["rpad"]),
        helper.make_node("Concat", ["pad_head", "zeros2", "rpad"], ["pads"], axis=0),
        helper.make_node("Pad", ["audio", "pads"], ["padded"], mode="constant"),
        helper.make_node("Conv", ["padded", "dft_w"], ["spec"], strides=[HOP]),  # [1, 514, N]
        helper.make_node("Split", ["spec", "split_sizes"], ["re", "im"], axis=1),  # each [1,257,N]
        helper.make_node("Mul", ["re", "re"], ["re2"]),
        helper.make_node("Mul", ["im", "im"], ["im2"]),
        helper.make_node("Add", ["re2", "im2"], ["power"]),  # [1,257,N]
        helper.make_node("Squeeze", ["power", "ax0"], ["p2"]),  # [257, N]
        helper.make_node("Transpose", ["p2"], ["pT"], perm=[1, 0]),  # [N, 257]
        helper.make_node("MatMul", ["pT", "mask_sum"], ["sums"]),  # [N, 4]
        helper.make_node("Unsqueeze", ["pT", "ax0"], ["pT3"]),  # [1, N, 257]
        helper.make_node("Transpose", ["mask_max"], ["mmT"], perm=[0, 2, 1]),  # [2,1,257]
        helper.make_node("Mul", ["pT3", "mmT"], ["masked"]),  # [2, N, 257]
        helper.make_node("ReduceMax", ["masked", "ax_last"], ["maxes_k"], keepdims=0),  # [2, N]
        helper.make_node("Transpose", ["maxes_k"], ["maxes"], perm=[1, 0]),  # [N, 2]
        helper.make_node("Concat", ["sums", "maxes"], ["bands"], axis=1),  # [N, 6]
    ]
    inits = [W, Msum, Mmax,
             numpy_helper.from_array(np.array([2], np.int64), "idx2"), numpy_helper.from_array(np.array(HOP, np.int64), "hop"),
             numpy_helper.from_array(np.array([0], np.int64), "ax0"), numpy_helper.from_array(np.array([0, 0, HOP], np.int64), "pad_head"), numpy_helper.from_array(np.array([0, 0], np.int64), "zeros2"),
             numpy_helper.from_array(np.array([257, 257], np.int64), "split_sizes"), numpy_helper.from_array(np.array([2], np.int64), "ax_last")]
    graph = helper.make_graph(g_nodes, "band_feats", [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "T"])],
                              [helper.make_tensor_value_info("bands", TensorProto.FLOAT, ["N", 6])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "models/branch_b/band_feats.onnx")
