"""Send the sketch, not the weights."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from .projection import make_projection
from .rounding import stable_layer_id, stochastic_round_

logger = logging.getLogger(__name__)


@dataclass
class LayerDelta:
    """One layer's update, in sketch form. `scaled_u` is already eta * U."""

    name: str
    scaled_u: torch.Tensor
    seed: int
    r: int
    k: int
    kind: str
    sr_seed: int
    layer_id: int = 0

    master_init: torch.Tensor | None = None

    wshape: tuple | None = None

    master_init_path: str | None = None

    def nbytes(self) -> int:
        return self.scaled_u.numel() * self.scaled_u.element_size()

    def projection(self, device) -> torch.Tensor:
        """A [r, k] fp32, regenerated from the seed. Cache it across many small slices of
        the same delta (a MoE tensor is applied one expert at a time)."""
        return make_projection(self.r, self.k, self.seed, device, torch.float32, self.kind)

    def materialize(self, device, dtype, row_slice=None, col_slice=None, row_slices=None, A=None) -> torch.Tensor:
        """dW = -(eta U) @ A, reconstructed from the seed; optionally only some rows and
        columns of it."""
        if A is None:
            A = self.projection(device)
        U = self.scaled_u.to(device, torch.float32)
        if row_slices is not None and row_slice is not None:
            raise ValueError("row_slice and row_slices are mutually exclusive")
        if row_slice is not None:
            lo, hi = row_slice
            d = U.shape[0]
            lo, hi = max(0, lo + d if lo < 0 else lo), max(0, hi + d if hi < 0 else hi)
            row_slices = [(min(lo, d), max(0, min(hi, d) - min(lo, d)))]
        out = self.wshape[1] if self.wshape is not None and len(self.wshape) == 3 else None
        with exact_fp32_matmul():
            if row_slices is None:
                dW = _fixed_shape_gemm(U, A, out)
            else:
                dW = (
                    torch.cat([_fixed_shape_gemm(U, A, out, a, n) for a, n in row_slices], 0)
                    if len(row_slices) != 1
                    else _fixed_shape_gemm(U, A, out, *row_slices[0])
                )
        if col_slice is not None:
            dW = dW[:, col_slice[0] : col_slice[1]]
        return -dW.to(dtype)


def _fixed_shape_gemm(U, A, out=None, row0: int = 0, rows=None) -> torch.Tensor:
    """Rows [row0, row0+rows) of U @ A, computed at the protocol's fixed GEMM shape: the
    whole [d, r] @ [r, k] when `out` is None, otherwise one expert ([out, r] @ [r, k]) at a
    time over the experts the requested rows touch. See LayerDelta.materialize."""
    d = U.shape[0]
    if rows is None:
        rows = d - row0
    if rows <= 0:
        return U.new_empty(0, A.shape[1])
    if out is None:
        return (U @ A)[row0 : row0 + rows]
    e0, e1 = row0 // out, (row0 + rows - 1) // out
    blocks = []
    for e in range(e0, e1 + 1):
        blk = U[e * out : (e + 1) * out] @ A
        lo = max(row0, e * out) - e * out
        hi = min(row0 + rows, (e + 1) * out) - e * out
        blocks.append(blk[lo:hi])
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)


class exact_fp32_matmul:
    """Force the dW GEMM to true IEEE fp32, whatever the ambient backend state is."""

    def __enter__(self):
        self._prev = None
        mm = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        if mm is not None and hasattr(mm, "allow_tf32"):
            self._prev = mm.allow_tf32
            mm.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        mm = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        if self._prev is not None and mm is not None:
            mm.allow_tf32 = self._prev
        return False


def trainer_fp32(L):
    """The fp32 copy of L's weights the trainer updates, or None if it holds no more
    precision than the engine's bf16 already does."""
    w = getattr(L, "W_master", None)
    if w is None:
        w = L.W._local_tensor if hasattr(L.W, "_local_tensor") else L.W.data
        if w.dtype != torch.float32:
            return None
    if w.ndim != 2:
        w = w.reshape(-1, w.shape[-1])
    if w.shape != (int(L.d), int(L.k)):
        return _GATHER_SENTINEL
    return w


_GATHER_SENTINEL = "sharded"


_WARNED: set = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *args)


def pack_deltas(
    layers, sr_seed: int, u_dtype=torch.float32, with_master_init=False, master_source=None
) -> list[LayerDelta]:
    """Trainer side: turn the merge into a wire payload."""
    out = []
    for L in layers:
        u = getattr(L, "last_u", None)
        if u is None:
            raise RuntimeError(f"{L.name}: no cached U; call merge(cache_u=True)")
        if not torch.isfinite(u).all():
            raise ValueError(
                f"{L.name}: eta*U contains non-finite values ({int((~torch.isfinite(u)).sum())} "
                "of {}); the merge produced NaN/Inf and shipping it would corrupt the "
                "engine.".format(u.numel())
            )
        if not torch.equal(u.to(u_dtype).to(u.dtype), u):
            raise ValueError(
                f"{L.name}: quantizing eta*U to {u_dtype} loses information the trainer "
                f"already applied. Set LograConfig.u_wire_dtype={u_dtype} so the merge and "
                "the wire use the same factor."
            )
        seed = getattr(L, "last_seed", None)
        if seed is None:
            raise RuntimeError(
                f"{L.name}: no last_seed; the merge must record the projection it applied "
                "(merge(cache_u=True)) or a reseed would make the engine rebuild a "
                "different A than the trainer used"
            )
        init = None
        if with_master_init is True or (with_master_init and L.name in with_master_init):
            w = trainer_fp32(L)
            if w is _GATHER_SENTINEL:
                w = (master_source or {}).get(L.name)
                if w is None:
                    _warn_once(
                        "master_init_sharded",
                        "[logra-refit] %s is FSDP-sharded and the caller supplied no "
                        "gathered fp32 master; the engine seeds from bf16 and stays "
                        "~1 ULP off on a fraction of elements after a resync.",
                        L.name,
                    )
                    w = None
            if w is not None:
                init = w.detach().to(torch.float32).contiguous()
        wshape = tuple(int(x) for x in getattr(L, "wshape", ()) or ())
        out.append(
            LayerDelta(
                L.name,
                u.to(u_dtype).contiguous(),
                int(seed),
                int(L.r),
                int(L.k),
                str(L.proj_kind),
                int(sr_seed),
                int(
                    getattr(L, "layer_id", None)
                    if getattr(L, "layer_id", None) is not None
                    else stable_layer_id(L.name)
                ),
                init,
                wshape if len(wshape) > 2 else None,
            )
        )
    return out


def payload_bytes(deltas) -> int:
    return sum(d.nbytes() for d in deltas)


def _tensor_digest(t) -> tuple:
    """Everything weight_fingerprint needs from one tensor, in ~24 bytes instead of the
    tensor: its shape, a position-weighted sum over the raw bit patterns, and a float sum."""
    t = t.detach()
    f = t.reshape(-1).to(torch.float32)
    if not f.numel():
        return (tuple(t.shape), None, None)

    bits = f.view(torch.int32).to(torch.int64)
    pos = torch.arange(1, bits.numel() + 1, device=bits.device, dtype=torch.int64)
    acc = int((bits * pos).sum().item()) & ((1 << 63) - 1)

    tot = int(bits.sum().item()) & ((1 << 63) - 1)
    return (tuple(t.shape), acc, tot)


def norm_digest(d):
    """Canonical form for a digest that may have crossed a serialisation boundary."""
    shape, acc, tot = d
    return (tuple(int(x) for x in shape), None if acc is None else int(acc), None if tot is None else int(tot))


def _fingerprint_digests(digests: dict) -> int:
    """Fold per-tensor digests into one order-independent checksum."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for name in sorted(digests):
        shape, acc, tot = digests[name]
        h.update(name.encode())
        h.update(str(shape).encode())
        if acc is not None:
            h.update(acc.to_bytes(8, "little"))
            h.update(tot.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def weight_fingerprint(model_or_layers, n: int = 4096) -> int:
    """Cheap order-independent checksum of the sketched weights."""
    items = model_or_layers if isinstance(model_or_layers, dict) else {L.name: L.W for L in model_or_layers}
    return _fingerprint_digests({name: _tensor_digest(t) for name, t in items.items()})


def _vllm_name_mapper(model):
    """vLLM's own HF-name -> internal-name mapping, the same one `load_weights` applies."""
    mapper = getattr(type(model), "hf_to_vllm_mapper", None) or getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return lambda n: n

    def _map(name: str) -> str:
        try:
            return mapper._map_name(name) or name
        except (AttributeError, KeyError, ValueError):
            return name

    return _map


def _fusion_plan(model) -> dict:
    """HF module suffix -> (vLLM suffix, index within the fused tensor).

    Read off the model's own `packed_modules_mapping`, so a model that fuses
    differently (or not at all) needs no change here.
    """
    plan = {}

    mapping = (
        getattr(model, "packed_modules_mapping", None) or getattr(type(model), "packed_modules_mapping", None) or {}
    )
    for fused, parts in mapping.items():
        for i, part in enumerate(parts):
            plan[part] = (fused, i)
    return plan


WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod")


def canonical_name(name: str, wrappers=WRAPPER_SEGMENTS) -> str:
    """named_modules() path -> the state_dict path vLLM will recognise."""
    parts = [p for p in name.split(".") if p not in wrappers]
    return ".".join(parts)


def is_row_parallel(target: str, model, override=None) -> bool:
    """Does vLLM shard this weight along the INPUT dim (row-parallel) or the OUTPUT dim?"""
    if override is not None:
        return override(target)
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    path = target.removesuffix(".weight")
    m = dict(model.named_modules()).get(path)
    if isinstance(m, RowParallelLinear):
        return True
    if isinstance(m, ColumnParallelLinear):
        return False
    raise ValueError(
        f"cannot determine parallelism for {target!r}: owning module is {type(m).__name__}. "
        "Refusing to guess -- a wrong axis slices the shard incorrectly without any error."
    )


def check_slice_exactness(deltas, tp_size: int, device=None, tol: int = 0) -> list:
    """Which layers' sliced reconstruction is NOT bitwise equal to slicing the full dW?"""
    bad = []
    for d in deltas:
        rows = d.scaled_u.shape[0]
        if rows % tp_size or d.k % tp_size or d.wshape is not None:
            continue
        dev = device or d.scaled_u.device
        full = d.materialize(dev, torch.float32)
        rp = rows // tp_size
        cp = d.k // tp_size
        row_ok = all(
            torch.equal(
                d.materialize(dev, torch.float32, row_slice=(i * rp, (i + 1) * rp)), full[i * rp : (i + 1) * rp]
            )
            for i in range(tp_size)
        )
        col_ok = all(
            torch.equal(
                d.materialize(dev, torch.float32, col_slice=(i * cp, (i + 1) * cp)), full[:, i * cp : (i + 1) * cp]
            )
            for i in range(tp_size)
        )
        if not (row_ok and col_ok):
            bad.append(d.name)
    return bad


_vacc: dict = {}


def _verify_on() -> bool:
    import os

    return os.environ.get("LOGRA_VERIFY_SYNC", "0") == "1"


def take_applied_fingerprint(model):
    """Pop and hash the dW this process reconstructed during the last apply_deltas().

    Returns (fingerprint, n_matrices). (None, 0) when verification is off or nothing was
    applied -- the caller must treat 0 as a FAILURE, never as a pass.
    """
    acc = _vacc.pop(id(model), None)
    if not acc:
        return None, 0
    return _fingerprint_digests(acc), len(acc)


def _master_device() -> str:
    """Where the engine keeps its fp32 accumulator: "cpu" (default) or "cuda".

    Set LOGRA_ENGINE_MASTER_DEVICE=cuda to keep it in HBM, which is faster per refit and
    costs 4 B/param of the memory the method is trying to save.
    """
    return os.environ.get("LOGRA_ENGINE_MASTER_DEVICE", "cpu").lower()


def _engine_master(model, target: str, p, enabled: bool):
    """Per-engine fp32 accumulator for one target tensor, created lazily from its current"""
    if not enabled or p.data.dtype == torch.float32:
        return None
    cache = getattr(model, "_logra_master", None)
    if cache is None:
        cache = {}
        try:
            model._logra_master = cache
        except (AttributeError, TypeError):
            return None
    m = cache.get(target)
    if m is None:
        m = p.data.to(torch.float32).clone()
        if _master_device() == "cpu":
            m = m.cpu()
            try:
                m = m.pin_memory()
            except RuntimeError:
                pass
        cache[target] = m
    return m


def reset_engine_master(model, target: str | None = None) -> int:
    """Drop cached fp32 accumulators so they are rebuilt from the live weights."""
    cache = getattr(model, "_logra_master", None)
    if not cache:
        return 0
    if target is None:
        n = len(cache)
        cache.clear()
        return n
    return 1 if cache.pop(target, None) is not None else 0


def _column_shard_rows(dims, i, shard_rows, tp_size, tp_rank):
    """Single-slice form of _column_shard_slices (kept for callers that expect (r0, rows))."""
    sl = _column_shard_slices(dims, i, shard_rows, tp_size, tp_rank)
    if len(sl) != 1:
        raise ValueError("this part is not a single contiguous slice on this rank; use _column_shard_slices")
    return sl[0]


def _column_shard_slices(dims, i, shard_rows, tp_size, tp_rank, gdn=None):
    """Row slices [(r0, rows), ...] of fused part `i` (full matrix has dims[i] rows) that
    THIS tp rank holds, for a column-parallel (output-split) target whose fused shard on
    this rank has `shard_rows` rows in total. The engine's shard is the concatenation, part
    by part, of these slices."""
    if gdn is not None and len(dims) == 2:
        K, V = int(gdn[0]), int(gdn[1])
        if (
            list(dims) == [2 * K + V, V]
            and K % tp_size == 0
            and V % tp_size == 0
            and shard_rows == (2 * K + 2 * V) // tp_size
        ):
            kr, vr = K // tp_size, V // tp_size
            if i == 0:
                return [(tp_rank * kr, kr), (K + tp_rank * kr, kr), (2 * K + tp_rank * vr, vr)]
            return [(tp_rank * vr, vr)]
    even = [dd // tp_size if dd % tp_size == 0 else None for dd in dims]
    if all(e is not None for e in even) and sum(even) == shard_rows:
        return [(tp_rank * even[i], even[i])]
    if len(dims) == 3 and dims[1] == dims[2] and dims[0] % tp_size == 0:
        q_rows = dims[0] // tp_size
        rest = shard_rows - q_rows
        if rest > 0 and rest % 2 == 0:
            kv_rows = rest // 2
            if kv_rows > 0 and dims[1] % kv_rows == 0:
                num_kv_heads = dims[1] // kv_rows
                if num_kv_heads < tp_size and tp_size % num_kv_heads == 0:
                    replicas = tp_size // num_kv_heads
                    if i == 0:
                        return [(tp_rank * q_rows, q_rows)]
                    g = tp_rank // replicas
                    return [(g * kv_rows, kv_rows)]
    raise ValueError(
        f"cannot derive the tp_rank={tp_rank}/{tp_size} shard of fused part {i} (full row "
        f"counts {dims}, gdn={gdn}) from an engine shard of {shard_rows} rows: neither an even "
        "split, vLLM's replicated-KV-head layout, nor the gated-delta-net layout matches."
    )


def gdn_dims(model, target):
    """(key_dim, value_dim) of the gated-delta-net module that owns `target`, else None.
    Read from the ENGINE module (vLLM's Qwen3NextGatedDeltaNet and its Qwen3.5 twin expose
    both), so the slicing follows what the engine actually built."""
    path = target.removesuffix(".weight")
    owner = path.rsplit(".", 1)[0] if "." in path else ""
    m = dict(model.named_modules()).get(owner)
    K, V = getattr(m, "key_dim", None), getattr(m, "value_dim", None)
    return (int(K), int(V)) if isinstance(K, int) and isinstance(V, int) else None


_HF_TO_VLLM_EXPERTS = {"gate_up_proj": "w13_weight", "down_proj": "w2_weight"}


def _fused_expert_target(params: dict, remap, parent: str, leaf: str):
    """Trainer fused-expert Parameter -> (engine FusedMoE tensor name, "w13" | "w2"), or
    None when this is not one. Decided from what the ENGINE holds: the fused tensor must
    exist, be 3-D, and there must be no dense `<name>.weight` (a dense MLP's down_proj is a
    Linear and takes the ordinary path)."""
    vname = _HF_TO_VLLM_EXPERTS.get(leaf)
    if vname is None:
        return None
    dense = remap(f"{parent}.{leaf}.weight" if parent else f"{leaf}.weight")
    if dense in params:
        return None
    target = remap(f"{parent}.{vname}" if parent else vname)
    prm = params.get(target)
    if prm is None:
        prefix = target[: -len(vname)]
        cands = [n for n, q in params.items() if n.startswith(prefix) and n.endswith("." + vname) and q.ndim == 3]
        if len(cands) > 1:
            raise ValueError(f"{parent}.{leaf}: several engine tensors could take it: {cands}")
        if cands:
            target, prm = cands[0], params[cands[0]]
    if prm is None or prm.ndim != 3:
        return None
    return target, ("w13" if vname == "w13_weight" else "w2")


def resolve_target(name: str, params: dict, plan: dict, remap):
    """Delta name -> (engine tensor name, index within a fused group, kind).

    kind is None for an ordinary (possibly q/k/v- or gate/up-fused) matrix, "w13" or "w2"
    for a fused MoE expert tensor. The same resolution is used by apply_deltas and by the
    layout the verification paths ask the engine for, so the two cannot disagree.
    """
    leaf = name.rsplit(".", 1)[-1]
    parent = name.rsplit(".", 1)[0] if "." in name else ""
    if leaf in plan:
        fused, idx = plan[leaf]
        target = f"{parent}.{fused}.weight" if parent else f"{fused}.weight"
    else:
        target, idx = f"{name}.weight", 0
    target = remap(target)
    if target not in params:
        moe = _fused_expert_target(params, remap, parent, leaf)
        if moe is not None:
            return moe[0], 0, moe[1]
    return target, idx, None


def fused_expert_blocks(kind: str, wshape, p_shape, tp_size: int, tp_rank: int) -> list:
    """This rank's per-expert blocks of a FusedMoE tensor, in the engine's 2-D view."""
    E, out, inn = (int(x) for x in wshape)
    if int(p_shape[0]) != E:
        raise ValueError(
            f"engine holds {int(p_shape[0])} experts, the trainer {E}; expert "
            "parallelism in the engine is not supported by the delta refit"
        )
    if kind == "w13":
        if out % 2:
            raise ValueError(f"gate_up_proj has odd row count {out}")
        I = out // 2
        if I % tp_size:
            raise ValueError(f"intermediate size {I} not divisible by tp_size={tp_size}")
        ir = I // tp_size
        if tuple(int(x) for x in p_shape[1:]) != (2 * ir, inn):
            raise ValueError(
                f"w13 shard {tuple(p_shape)} != expected (E, {2 * ir}, {inn}) "
                f"for gate_up_proj {(E, out, inn)} at tp_size={tp_size}"
            )
        return [
            (e * 2 * ir, 2 * ir, [(e * out + tp_rank * ir, ir), (e * out + I + tp_rank * ir, ir)], None)
            for e in range(E)
        ]
    if kind == "w2":
        if inn % tp_size:
            raise ValueError(f"intermediate size {inn} not divisible by tp_size={tp_size}")
        w = inn // tp_size
        if tuple(int(x) for x in p_shape[1:]) != (out, w):
            raise ValueError(
                f"w2 shard {tuple(p_shape)} != expected (E, {out}, {w}) "
                f"for down_proj {(E, out, inn)} at tp_size={tp_size}"
            )
        return [(e * out, out, [(e * out, out)], (tp_rank * w, w)) for e in range(E)]
    raise ValueError(f"unknown fused expert kind {kind!r}")


def expected_engine_block(parts, shape, rank, tp, gdn=None, kind=None, wshape=None):
    """What a FULL refit would put on this rank for one engine tensor, from the trainer's"""
    shape = tuple(int(x) for x in shape)
    if kind is not None:
        (P,) = parts
        blocks = fused_expert_blocks(kind, wshape, shape, tp, rank)
        blk = torch.cat(
            [
                torch.cat([P[a : a + n] for a, n in rs], 0)[:, cs[0] : cs[0] + cs[1]]
                if cs is not None
                else torch.cat([P[a : a + n] for a, n in rs], 0)
                for _, _, rs, cs in blocks
            ],
            0,
        )
        return blk.reshape(shape)
    rows, cols = shape
    full_rows = sum(p.shape[0] for p in parts)
    full_cols = parts[0].shape[1]
    if cols == full_cols and rows < full_rows:
        dims = [p.shape[0] for p in parts]
        blk = torch.cat(
            [
                torch.cat([p[a : a + n] for a, n in _column_shard_slices(dims, i, rows, tp, rank, gdn=gdn)], 0)
                for i, p in enumerate(parts)
            ],
            0,
        )
    elif cols * tp == full_cols and rows == full_rows:
        blk = torch.cat([p[:, rank * (p.shape[1] // tp) : (rank + 1) * (p.shape[1] // tp)] for p in parts], 0)
    elif (rows, cols) == (full_rows, full_cols):
        blk = torch.cat(parts, 0)
    else:
        raise ValueError(
            f"engine shard {shape} does not correspond to {len(parts)} trainer "
            f"matrices totalling ({full_rows}, {full_cols}) at tp={tp}"
        )
    return blk


class _DigestAcc:
    """_tensor_digest of a large matrix, accumulated from row blocks so the whole
    reconstruction never has to exist at once (a 30B-A3B expert tensor is 1.6 GB fp32).
    Bit-identical to digesting the full tensor: both aggregates are integer sums, the
    position weight is the global row-major index, and two's-complement wrap-around keeps
    the low 63 bits of a sum independent of how it was split."""

    def __init__(self, shape):
        self.shape, self.acc, self.tot, self.n = tuple(int(x) for x in shape), 0, 0, 0

    def add(self, blk, elem_offset: int):
        f = blk.detach().reshape(-1).to(torch.float32)
        if not f.numel():
            return
        bits = f.view(torch.int32).to(torch.int64)
        pos = torch.arange(elem_offset + 1, elem_offset + 1 + bits.numel(), device=bits.device, dtype=torch.int64)
        self.acc += int((bits * pos).sum().item())
        self.tot += int(bits.sum().item())
        self.n += bits.numel()

    def result(self) -> tuple:
        if self.n == 0:
            return (self.shape, None, None)
        return (self.shape, self.acc & ((1 << 63) - 1), self.tot & ((1 << 63) - 1))


def apply_deltas(
    model,
    deltas,
    tp_size: int = 1,
    tp_rank: int = 0,
    row_parallel=None,
    master_weights: bool = True,
    merge_precision: str = "stochastic",
) -> dict:
    """Worker side: regenerate A, rebuild this rank's shard of dW, add it into vLLM."""
    plan = _fusion_plan(model)
    remap = _vllm_name_mapper(model)
    params = dict(model.named_parameters())
    by_target: dict[str, list] = {}
    kinds: dict[str, str | None] = {}

    for d in deltas:
        target, idx, kind = resolve_target(d.name, params, plan, remap)
        by_target.setdefault(target, []).append((idx, d))
        kinds[target] = kind

    stats = {
        "applied": 0,
        "bytes": payload_bytes(deltas),
        "missing": [],
        "tp_size": tp_size,
        "master_weights": bool(master_weights),
    }
    for target, items in by_target.items():
        p = params.get(target)
        if p is None:
            stats["missing"].append(target)
            continue
        kind = kinds[target]
        items.sort(key=lambda t: t[0])
        p2 = p.data if p.data.ndim == 2 else p.data.view(-1, p.data.shape[-1])

        blocks = []
        if kind is not None:
            if len(items) != 1:
                raise ValueError(f"{target}: a fused expert tensor takes exactly one delta, got {len(items)}")
            d = items[0][1]
            if d.wshape is None or len(d.wshape) != 3:
                raise ValueError(
                    f"{d.name} -> {target}: the delta carries no 3-D shape "
                    f"(wshape={d.wshape}); a fused expert tensor needs it"
                )
            E, out, inn = d.wshape
            if d.scaled_u.shape[0] != E * out or d.k != inn:
                raise ValueError(
                    f"{d.name} -> {target}: U is {tuple(d.scaled_u.shape)}, k={d.k}, "
                    f"inconsistent with wshape {tuple(d.wshape)}"
                )
            try:
                blocks = [
                    (d, r0, n, rs, cs)
                    for r0, n, rs, cs in fused_expert_blocks(kind, d.wshape, tuple(p.data.shape), tp_size, tp_rank)
                ]
            except ValueError as e:
                raise ValueError(f"{d.name} -> {target}: {e}") from None
        else:
            dims = [dd.scaled_u.shape[0] for _, dd in items]
            row = 0
            for part_i, (_, d) in enumerate(items):
                rowpar = is_row_parallel(target, model, row_parallel)
                d_full = d.scaled_u.shape[0]
                if rowpar:
                    if d.k % tp_size:
                        raise ValueError(f"{d.name}: k={d.k} not divisible by tp_size={tp_size}")
                    width = d.k // tp_size
                    blocks.append((d, row, d_full, [(0, d_full)], (tp_rank * width, width)))
                    row += d_full
                else:
                    if tp_size == 1:
                        slices = [(0, d_full)]
                    else:
                        try:
                            slices = _column_shard_slices(
                                dims, part_i, p2.shape[0], tp_size, tp_rank, gdn=gdn_dims(model, target)
                            )
                        except ValueError as e:
                            raise ValueError(f"{d.name} -> {target}: {e}") from None
                    rows = sum(n for _, n in slices)
                    blocks.append((d, row, rows, slices, None))
                    row += rows

            if row != p2.shape[0]:
                raise ValueError(
                    f"{target}: deltas cover {row} of {p2.shape[0]} rows "
                    f"({len(items)} part(s) supplied). A fused group must be complete and "
                    "correctly sized, or updates land in the wrong rows."
                )

        master = _engine_master(model, target, p, master_weights)
        master2 = None if master is None else (master if master.ndim == 2 else master.view(-1, master.shape[-1]))
        verify = _verify_on()
        A_cache: dict = {}
        digests: dict = {}
        for d, row0, rows, slices, cols in blocks:
            A = A_cache.get(id(d))
            if A is None:
                A = A_cache[id(d)] = d.projection(p.device)
            if not verify:
                dW = d.materialize(
                    p.device,
                    torch.float32,
                    row_slices=slices,
                    col_slice=None if cols is None else (cols[0], cols[0] + cols[1]),
                    A=A,
                )
            else:
                if kind is None:
                    full = d.materialize(p.device, torch.float32, A=A)
                    base = 0
                    if id(d) not in digests:
                        digests[id(d)] = _tensor_digest(full)
                else:
                    E, out, inn = d.wshape
                    e = row0 // rows
                    base = e * out
                    full = d.materialize(p.device, torch.float32, row_slice=(base, base + out), A=A)
                    acc = digests.get(id(d))
                    if acc is None:
                        acc = digests[id(d)] = _DigestAcc((E * out, inn))
                    acc.add(full, base * inn)
                dW = torch.cat([full[a - base : a - base + n] for a, n in slices], 0)
                if cols is not None:
                    dW = dW[:, cols[0] : cols[0] + cols[1]]
                del full
            sl = p2[row0 : row0 + rows]
            if sl.shape != dW.shape:
                raise ValueError(
                    f"{d.name} -> {target}: shard shape {tuple(sl.shape)} != rebuilt "
                    f"{tuple(dW.shape)} (tp_size={tp_size}, tp_rank={tp_rank}). Either the "
                    "row/column-parallel classification is wrong for this layer, or the "
                    "same delta was supplied twice for this target."
                )
            if master2 is not None:
                msl = master2[row0 : row0 + rows]

                on_host = msl.device.type != p.device.type
                blk_dev = msl.to(p.device, non_blocking=True) if on_host else msl

                tol = blk_dev.abs() * (2.0**-7) + torch.finfo(torch.float32).tiny
                if ((sl.to(torch.float32) - blk_dev).abs() > tol).any():
                    blk_dev.copy_(sl.to(torch.float32))
                    stats["master_resynced"] = stats.get("master_resynced", 0) + 1
                blk_dev += dW
                if d.master_init is not None:
                    mi = d.master_init
                    if mi.ndim != 2:
                        mi = mi.reshape(-1, mi.shape[-1])
                    blk = torch.cat([mi[a : a + n] for a, n in slices], 0).to(blk_dev.device, torch.float32)
                    if cols is not None:
                        blk = blk[:, cols[0] : cols[0] + cols[1]]
                    if blk.shape != blk_dev.shape:
                        raise ValueError(
                            f"{d.name} -> {target}: master_init shard {tuple(blk.shape)} "
                            f"!= master block {tuple(blk_dev.shape)}"
                        )
                    blk_dev.copy_(blk)

                sl.copy_(blk_dev.to(sl.dtype))
                if on_host:
                    msl.copy_(blk_dev, non_blocking=True)
            else:
                if len(slices) > 1 or kind is not None:
                    raise ValueError(
                        f"{d.name} -> {target}: the legacy stochastic-rounding path "
                        "cannot address a multi-slice or fused-expert shard; use "
                        "master_weights=True"
                    )
                c0 = cols[0] if cols is not None else 0
                stochastic_round_(
                    sl,
                    sl.to(torch.float32) + dW,
                    seed=d.sr_seed,
                    layer_id=d.layer_id,
                    row0=slices[0][0],
                    col0=c0,
                    ncols=d.k,
                )
        if verify:
            acc = _vacc.setdefault(id(model), {})
            for _, d in items:
                dg = digests.get(id(d))
                if dg is not None:
                    acc[d.name] = dg.result() if isinstance(dg, _DigestAcc) else dg
        stats["applied"] += len(items)
    return stats
