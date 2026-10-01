"""Chunked LM-head log probabilities, recomputed in backward to bound peak memory."""

from __future__ import annotations

import torch


class ChunkedHeadLogProbs(torch.autograd.Function):
    """(log_probs, entropy) of `labels` under softmax(hidden @ weight.T / temperature),
    chunked over rows, logits recomputed in backward. Shapes: hidden [N, H], weight [V, H],
    labels [N] (long; negatives are clamped to 0, callers mask them out). Returns fp32 [N]
    tensors; entropy is a zero-length tensor when not wanted."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature: float, chunk: int, want_entropy: bool):
        n = hidden.shape[0]
        lp = hidden.new_empty(n, dtype=torch.float32)
        ent = hidden.new_empty(n if want_entropy else 0, dtype=torch.float32)
        lab = labels.clamp_min(0)
        for i in range(0, n, chunk):
            z = (hidden[i : i + chunk] @ weight.t()).to(torch.float32)
            if temperature != 1.0:
                z = z / temperature
            lse = torch.logsumexp(z, dim=-1)
            lp[i : i + chunk] = z.gather(-1, lab[i : i + chunk, None]).squeeze(-1) - lse
            if want_entropy:
                p = torch.softmax(z, dim=-1)
                ent[i : i + chunk] = lse - (p * z).sum(-1)
            del z
        ctx.save_for_backward(hidden, weight, lab)
        ctx.temperature, ctx.chunk, ctx.want_entropy = float(temperature), int(chunk), bool(want_entropy)
        return lp, ent

    @staticmethod
    def backward(ctx, g_lp, g_ent):
        hidden, weight, lab = ctx.saved_tensors
        T, chunk = ctx.temperature, ctx.chunk
        need_w = ctx.needs_input_grad[1]
        grad_h = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        n = hidden.shape[0]
        for i in range(0, n, chunk):
            h = hidden[i : i + chunk]
            z = (h @ weight.t()).to(torch.float32)
            if T != 1.0:
                z = z / T
            p = torch.softmax(z, dim=-1)
            # d log p_y / dz = onehot(y) - p
            dz = -p * g_lp[i : i + chunk, None]
            dz.scatter_add_(-1, lab[i : i + chunk, None], g_lp[i : i + chunk, None].to(dz.dtype))
            if ctx.want_entropy and g_ent is not None and g_ent.numel():
                # H = lse(z) - sum_j p_j z_j  ->  dH/dz_k = -p_k (z_k - E_p[z])
                ez = (p * z).sum(-1, keepdim=True)
                dz = dz - g_ent[i : i + chunk, None] * p * (z - ez)
            if T != 1.0:
                dz = dz / T
            dzw = dz.to(weight.dtype)
            if grad_h is not None:
                grad_h[i : i + chunk] = (dzw @ weight).to(hidden.dtype)
            if grad_w is not None:
                grad_w += dzw.t().to(torch.float32) @ h.to(torch.float32)
            del z, p, dz, dzw
        return grad_h, (grad_w.to(weight.dtype) if grad_w is not None else None), None, None, None, None


class HeadOut:
    """What the patched lm_head returns in place of logits: per-position log-probs of the
    requested labels and (optionally) entropies, both [B, S] fp32 and attached to autograd."""

    __slots__ = ("entropy", "log_probs")

    def __init__(self, log_probs, entropy):
        self.log_probs, self.entropy = log_probs, entropy


def chunked_head(hidden, weight, labels, temperature: float = 1.0, chunk: int = 1024, want_entropy: bool = False):
    """hidden [B, S, H], weight [V, H], labels [B, S] -> HeadOut with [B, S] outputs."""
    B, S, H = hidden.shape
    lp, ent = ChunkedHeadLogProbs.apply(
        hidden.reshape(-1, H), weight, labels.reshape(-1), float(temperature), int(chunk), bool(want_entropy)
    )
    return HeadOut(lp.view(B, S), ent.view(B, S) if want_entropy else None)


def reference_head(hidden, weight, labels, temperature: float = 1.0, want_entropy: bool = False):
    """The un-chunked computation, for tests: bf16/fp32 GEMM, fp32 log-softmax."""
    z = (hidden @ weight.t()).to(torch.float32)
    if temperature != 1.0:
        z = z / temperature
    logp = torch.log_softmax(z, dim=-1)
    lp = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1) if want_entropy else None
    return lp, ent


def install_chunked_head(head, state) -> bool:
    """Make `head` (the model's lm_head, an nn.Linear-like module with `.weight`) return a
    HeadOut instead of logits whenever `state.head_labels` is set; otherwise it behaves as
    before. `state` carries head_labels [B, S], head_temperature, head_chunk, head_want_entropy.
    Idempotent. Falls back to the original forward for a tensor-parallel (DTensor) weight."""
    if getattr(head, "_logra_head_patched", False):
        return True
    if not hasattr(head, "weight"):
        return False
    orig_forward = head.forward

    def forward(hidden, *a, **k):
        labels = getattr(state, "head_labels", None)
        w = head.weight
        if labels is None or type(w).__name__ == "DTensor" or hidden.ndim != 3 or labels.shape != hidden.shape[:2]:
            return orig_forward(hidden, *a, **k)
        return chunked_head(
            hidden,
            w,
            labels,
            getattr(state, "head_temperature", 1.0),
            getattr(state, "head_chunk", 1024),
            getattr(state, "head_want_entropy", False),
        )

    head.forward = forward
    head._logra_head_patched = True
    return True
