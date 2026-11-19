"""SiLU and grouped FP8 with the native BF16 quantizer's rounding contract."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rtp_llm.ops import compute_ops


native_contiguous_silu_fp8 = getattr(compute_ops, "sm90_contiguous_silu_fp8", None)


# Padded expert counts vary with routing even for a fixed input length.
@triton.jit(do_not_specialize=["ROWS"])
def _silu_mul_native_fp8(
    gate_up,
    quantized,
    scales,
    ROWS,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    offsets = rows[:, None] * (2 * WIDTH) + cols[None, :]
    value = tl.load(gate_up + offsets, rows[:, None] < ROWS, other=0).to(tl.float32)
    gate = tl.load(gate_up + offsets + WIDTH, rows[:, None] < ROWS, other=0).to(
        tl.float32
    )
    # Match silu_and_mul's BF16 store before the native quantizer reads it.
    activated = ((gate * tl.sigmoid(gate)) * value).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(activated), axis=1), 1.0e-4)
    scale = tl.div_rn(maximum, 448.0)
    # The native quantizer rounds the inverse and multiplication to BF16.
    inverse = tl.div_rn(1.0, scale).to(tl.bfloat16).to(tl.float32)
    scaled = (activated * inverse[:, None]).to(tl.bfloat16)
    tl.store(
        quantized + rows[:, None] * WIDTH + cols[None, :],
        scaled.to(tl.float8e4nv),
        rows[:, None] < ROWS,
    )
    tl.store(scales + group * ROWS + rows, scale, rows < ROWS)


@gluon.jit(do_not_specialize=["rows"])
def _silu_mul_live_rows(gate_up, expert_ids, expert_end, quantized, scales, rows):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    start = gl.program_id(0) * 16
    row = start + gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    group = gl.program_id(1)
    col = group * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    # Each block is contained in a single 128-row aligned expert segment.
    expert = gl.load(expert_ids + start)
    end = gl.load(expert_end + expert)
    output = quantized + row[:, None] * 640 + col[None, :]
    scale_out = scales + group * rows + row
    if start < end:
        source = gate_up + row[:, None] * 1280 + col[None, :]
        valid = row[:, None] < end
        up = gl.load(source, valid, other=0).to(gl.float32)
        gate = gl.load(source + 640, valid, other=0).to(gl.float32)
        activated = ((gate * (1.0 / (1.0 + gl.exp(-gate)))) * up)
        activated = activated.to(gl.bfloat16).to(gl.float32)
        maximum = gl.maximum(gl.max(gl.abs(activated), 1), 1.0e-4)
        scale = gl.div_rn(maximum, 448.0)
        inverse = gl.div_rn(1.0, scale).to(gl.bfloat16).to(gl.float32)
        scaled = (activated * inverse[:, None]).to(gl.bfloat16)
        gl.store(output, scaled.to(gl.float8e4nv))
        gl.store(scale_out, scale)
    else:
        # The down GEMM can still read padding. Give it defined zero inputs.
        zero = gl.full((16, 128), 0.0, gl.bfloat16, layout).to(gl.float8e4nv)
        gl.store(output, zero)
        gl.store(scale_out, gl.div_rn(1.0e-4, 448.0))


def silu_mul_native_fp8(
    gate_up, *, expert_ids=None, expert_end=None, padding_bounds=None
):
    """Produce FP8 values and MN-major FP32 scales for qualified SM90 MoE."""
    if padding_bounds is not None and native_contiguous_silu_fp8 is not None:
        begin, end = padding_bounds
        return native_contiguous_silu_fp8(
            gate_up,
            expert_ids if begin < end else None,
            expert_end if begin < end else None,
            begin,
            end,
        )
    rows, doubled_width = gate_up.shape
    width = doubled_width // 2
    assert gate_up.dtype == torch.bfloat16 and gate_up.is_contiguous()
    assert doubled_width == 1280 and rows % 128 == 0
    output = torch.empty(
        (rows, width), device=gate_up.device, dtype=torch.float8_e4m3fn
    )
    output_scale = torch.empty_strided(
        (rows, width // 128), (1, rows), device=gate_up.device, dtype=torch.float32
    )
    if expert_ids is not None:
        assert expert_end is not None
        _silu_mul_live_rows[(rows // 16, width // 128)](
            gate_up, expert_ids, expert_end, output, output_scale, rows,
            num_warps=4, enable_fp_fusion=False,
        )
        return output, output_scale

    block_m = 16
    _silu_mul_native_fp8[(triton.cdiv(rows, block_m), width // 128)](
        gate_up,
        output,
        output_scale,
        rows,
        width,
        block_m,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, output_scale
