"""Shard-invariant stochastic rounding."""

from __future__ import annotations

import hashlib

import torch

_I32 = 0x1_0000_0000
_C1 = 0x85EBCA6B - _I32
_C2 = 0xC2B2AE35 - _I32
_GOLDEN = 0x9E3779B9 - _I32


def stable_layer_id(name: str) -> int:
    """Process-stable small integer for a layer name (Python's hash() is randomized)."""
    h = hashlib.blake2b(name.encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") % (1 << 31)


def _lshr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift on int32 (torch's >> is arithmetic, which drags in sign bits)."""
    return (x >> n) & ((1 << (32 - n)) - 1)


def _fmix32(x: int) -> int:
    """murmur3 finalizer on a Python int, used to mix the SEED before it meets the index."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def sr_noise(rows: int, cols: int, *, seed: int, layer_id: int, row0: int = 0, device=None) -> torch.Tensor:
    """16-bit rounding noise for the block [row0, row0+rows) x [0, cols) of a matrix.

    Value at global index (i, j) depends only on (seed, layer_id, i*cols + j), so the
    block returned here is exactly the corresponding sub-block of the full-matrix noise.
    """
    r = torch.arange(row0, row0 + rows, device=device, dtype=torch.int32).unsqueeze(1)
    c = torch.arange(cols, device=device, dtype=torch.int32).unsqueeze(0)
    x = r * int(cols) + c
    mixed = _fmix32(int(seed) & 0xFFFFFFFF)
    if mixed >= 1 << 31:
        mixed -= 1 << 32
    x = x ^ mixed ^ (int(layer_id) * _GOLDEN)
    x = x ^ _lshr(x, 16)
    x = x * _C1
    x = x ^ _lshr(x, 13)
    x = x * _C2
    x = x ^ _lshr(x, 16)
    return x & 0xFFFF


def stochastic_round_(
    dst: torch.Tensor,
    src_fp32: torch.Tensor,
    *,
    seed: int | None = None,
    layer_id: int = 0,
    row0: int = 0,
    col0: int = 0,
    ncols: int | None = None,
    generator=None,
) -> None:
    """Write src_fp32 into dst's dtype with unbiased stochastic rounding: E[SR(x)] = x."""
    if row0 < 0 or col0 < 0:
        raise ValueError(f"row0/col0 must be >= 0, got row0={row0}, col0={col0}")
    if dst.shape != src_fp32.shape:
        raise ValueError(f"shape mismatch: dst {tuple(dst.shape)} vs src {tuple(src_fp32.shape)}")
    if dst.dtype == torch.float32:
        if ncols is not None and col0 + src_fp32.shape[-1] > ncols:
            raise ValueError(f"column shard [{col0}, {col0 + src_fp32.shape[-1]}) exceeds ncols={ncols}")
        dst.copy_(src_fp32)
        return
    if dst.dtype is not torch.bfloat16:
        raise NotImplementedError(f"stochastic rounding not implemented for {dst.dtype}")
    if src_fp32.dtype is not torch.float32:
        raise TypeError(f"stochastic rounding needs an fp32 source, got {src_fp32.dtype}")
    if dst.shape != src_fp32.shape:
        raise ValueError(f"shape mismatch: dst {tuple(dst.shape)} vs src {tuple(src_fp32.shape)}")

    bits = src_fp32.contiguous().view(torch.int32)
    if seed is not None:
        if src_fp32.ndim != 2:
            raise ValueError("shard-invariant rounding expects a 2-D weight matrix")
        width = src_fp32.shape[1] if ncols is None else ncols
        if col0 + src_fp32.shape[1] > width:
            raise ValueError(f"column shard [{col0}, {col0 + src_fp32.shape[1]}) exceeds ncols={width}")
        noise = sr_noise(src_fp32.shape[0], width, seed=seed, layer_id=layer_id, row0=row0, device=src_fp32.device)
        if width != src_fp32.shape[1]:
            noise = noise[:, col0 : col0 + src_fp32.shape[1]]
    else:
        noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32, generator=generator)
    rounded = ((bits + noise) & ~0xFFFF).view(torch.float32)

    finite = torch.isfinite(src_fp32)
    if not bool(finite.all()):
        rounded = torch.where(finite, rounded, src_fp32)
    dst.copy_(rounded.to(torch.bfloat16))
