"""Random projections for LOGRA."""

from __future__ import annotations

import hashlib
import math

import torch

_MASK62 = (1 << 62) - 1


def layer_seed(base_seed: int, round_idx: int, name: str) -> int:
    """Stable per-(round, layer) seed. Deterministic across processes and restarts."""
    h = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")
    return (base_seed * 1_000_003 + round_idx * 7_919 + h) & _MASK62


_KINDS = ("rademacher", "gaussian", "sparse")


def make_projection(
    r: int,
    k: int,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    kind: str = "rademacher",
) -> torch.Tensor:
    """Return A in R^{r x k} with E[A_mi^2] = 1/r."""
    if kind not in _KINDS:
        raise ValueError(f"unknown projection kind {kind!r}; expected one of {sorted(_KINDS)}")
    if r <= 0 or k <= 0:
        raise ValueError(f"make_projection needs r > 0 and k > 0, got r={r}, k={k}")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & _MASK62)
    scale = 1.0 / math.sqrt(r)

    if kind == "rademacher":
        bits = torch.randint(0, 2, (r, k), generator=g, device="cpu", dtype=torch.uint8)
        A = bits.to(dtype).mul_(2.0).sub_(1.0).mul_(scale)
    elif kind == "gaussian":
        A = torch.randn(r, k, generator=g, device="cpu", dtype=torch.float64).mul_(scale).to(dtype)
    else:
        u = torch.rand(r, k, generator=g, device="cpu", dtype=torch.float64)
        A = torch.zeros(r, k, device="cpu", dtype=dtype)
        A[u < 1.0 / 6.0] = -1.0
        A[u > 5.0 / 6.0] = 1.0
        A.mul_(math.sqrt(3.0) * scale)
    return A.to(device=device)


def expected_relative_noise(r: int, kind: str = "rademacher", g: torch.Tensor | None = None) -> float:
    """std/mean of ||A g||^2 / ||g||^2 for an iid-entry sketch with E[A_mi^2] = 1/r."""
    if kind not in _KINDS:
        raise ValueError(f"unknown projection kind {kind!r}; expected one of {sorted(_KINDS)}")
    if r < 1:
        raise ValueError(f"expected_relative_noise needs r >= 1, got {r}")
    base = 2.0 / r
    if kind == "rademacher" and g is not None:
        n2 = float(g.pow(2).sum())
        base *= max(0.0, 1.0 - float(g.pow(4).sum()) / (n2 * n2))
    return math.sqrt(base)


def beta_relative_noise(r: int, k: int) -> float:
    """std/mean of ||Pi g||^2/||g||^2 for an *exact* random orthogonal projector Pi.

    ||Pi g||^2/||g||^2 ~ Beta(r/2, (k-r)/2)  =>  sqrt(2(k-r)/(kr)).
    Reference value for the idealized projector picture only; LOGRA's sketch is iid,
    so use `expected_relative_noise` for predictions.
    """
    if r < 1 or k < 1:
        raise ValueError(f"beta_relative_noise needs r >= 1 and k >= 1, got r={r}, k={k}")
    return math.sqrt(2.0 * (k - r) / (k * r))
