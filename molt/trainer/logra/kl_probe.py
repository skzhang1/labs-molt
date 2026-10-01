"""Predicted-divergence step: choose the merge scale from the policy's own geometry."""

from __future__ import annotations

import contextlib
import math

import torch
import torch.autograd.forward_ad as fwAD
import torch.distributed as dist

_WARNED = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[LOGRA] KL probe WARNING: {msg}", flush=True)


def _device_for_collectives():
    if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _all_reduce_sum(vals: list[float]) -> list[float]:
    if not dist.is_initialized():
        return list(vals)
    t = torch.tensor(vals, dtype=torch.float64, device=_device_for_collectives())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def all_reduce_min_int(v: int) -> int:
    if not dist.is_initialized():
        return int(v)
    t = torch.tensor([int(v)], dtype=torch.int64, device=_device_for_collectives())
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


@contextlib.contextmanager
def math_sdpa():
    """Force torch's scaled_dot_product_attention onto its MATH backend. The fused kernels
    (flash, mem-efficient, cuDNN, and the CPU flash kernel) have no forward-AD rule and raise
    NotImplementedError under forward mode; MATH is matmul + softmax and differentiates.
    Needed for models that call F.scaled_dot_product_attention directly (NeMo AutoModel's
    custom Qwen2/Qwen3 implementations) and harmless for HF models switched to eager."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        yield
        return
    with sdpa_kernel([SDPBackend.MATH]):
        yield


@contextlib.contextmanager
def eager_attention(model):
    """Flash / SDPA kernels have no forward-AD rule; HF's eager attention is plain
    matmul + softmax and does. Switch for the duration of the probe, then restore. Models
    that bypass HF's attention interface are covered by math_sdpa() instead; both are
    applied by probe_kernels()."""
    cfg = getattr(model, "config", None)
    prev = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    setter = getattr(model, "set_attn_implementation", None)
    switched = False
    if prev is not None and prev != "eager":
        try:
            if callable(setter):
                setter("eager")
            else:
                cfg._attn_implementation = "eager"
            switched = True
        except (AttributeError, ValueError, RuntimeError) as exc:
            _warn_once(
                "attn", f"could not switch attention to eager ({exc!r}); the probe forward may fail under forward AD"
            )
    try:
        yield
    finally:
        if switched:
            try:
                if callable(setter):
                    setter(prev)
                else:
                    cfg._attn_implementation = prev
            except (AttributeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"failed to restore attention implementation {prev!r}") from exc


def _passthrough_checkpoint(fn, *args, **kwargs):
    return fn(*args, **kwargs)


_NEMO_ATTN_OVERRIDES = {"rope_backend": "torch", "rope_fusion": False}


def _plain_fp32_rmsnorm(self, x):
    """Eager twin of NeMo's Float32RMSNorm.forward, whose body is a @torch.compile'd
    function (nemo_automodel/components/models/common/utils.py::_float32_rms_norm_fwd).
    A compiled graph runs on the primal values and returns plain tensors, so the tangent
    of the residual stream is dropped at EVERY norm; measured on the 1.5B run as "no
    tangent at the logits". Same arithmetic, same dtypes (max abs diff 2e-7 on the tiny
    model), differentiable in forward mode."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    return (self.weight * xf).to(dt)


_FORWARD_OVERRIDES = {"Float32RMSNorm": _plain_fp32_rmsnorm}


@contextlib.contextmanager
def no_compiled_kernels():
    """Run @torch.compile'd functions eagerly for the duration (torch >= 2.6). A compiled
    region has no forward-AD rule and silently returns primal-only tensors."""
    stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if stance is None:
        yield
        return
    try:
        with stance("force_eager"):
            yield
    except (AttributeError, ValueError, RuntimeError) as exc:
        if "force_eager" in repr(exc) or "stance" in repr(exc).lower():
            _warn_once("stance", f"torch.compiler.set_stance unavailable ({exc!r}); compiled kernels are not bypassed")
            yield
        else:
            raise


@contextlib.contextmanager
def plain_kernels(model):
    """Point every module at its plain-torch code path where a fused kernel would drop or
    reject the tangent; restore afterwards. Two mechanisms: attribute switches on modules
    that select a kernel at run time, and per-instance forward replacements for modules
    whose forward IS a compiled/fused function."""
    saved = []
    fwd_saved = []
    for m in model.modules():
        for attr, val in _NEMO_ATTN_OVERRIDES.items():
            if hasattr(m, attr) and getattr(m, attr) != val:
                saved.append((m, attr, getattr(m, attr)))
                try:
                    setattr(m, attr, val)
                except (AttributeError, TypeError):
                    saved.pop()
        fn = _FORWARD_OVERRIDES.get(type(m).__name__)
        if fn is not None:
            import types

            fwd_saved.append((m, m.__dict__.get("forward")))
            m.forward = types.MethodType(fn, m)
    try:
        yield
    finally:
        for m, attr, val in saved:
            setattr(m, attr, val)
        for m, old in fwd_saved:
            if old is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = old


@contextlib.contextmanager
def probe_kernels(model):
    """Everything a forward-mode pass needs from the model: HF eager attention, SDPA math
    backend, plain rotary kernels, no activation checkpointing, HF's torch fallback in
    place of fused linear-attention kernels, and the per-expert (eager) MoE loop in place
    of grouped GEMMs."""
    with (
        eager_attention(model),
        math_sdpa(),
        plain_kernels(model),
        no_activation_checkpointing(model),
        no_compiled_kernels(),
        hf_torch_fallback(model),
        eager_experts(model),
    ):
        yield


@contextlib.contextmanager
def eager_experts(model):
    """HF transformers >= 5 dispatches fused MoE experts through `config._experts_implementation`
    ("grouped_mm" / "batched_mm" / "eager"). The grouped GEMM has no forward-mode rule and
    would drop the tangent; the eager loop (one F.linear per hit expert) carries it exactly
    and is what the experts tangent hook mirrors. Switched for the probe only."""
    saved = []
    for cfg in {id(c): c for c in (getattr(m, "config", None) for m in model.modules()) if c is not None}.values():
        cur = getattr(cfg, "_experts_implementation", None)
        if cur is not None and cur != "eager":
            saved.append((cfg, cur))
            cfg._experts_implementation = "eager"
    try:
        yield
    finally:
        for cfg, cur in saved:
            cfg._experts_implementation = cur


@contextlib.contextmanager
def hf_torch_fallback(model=None):
    """Run HF's hybrid linear-attention layers on their pure-torch code path."""
    import sys as _sys

    saved_flags = []
    for name, mod in list(_sys.modules.items()):
        if not (name.startswith("transformers.models.") and ".modeling_" in name):
            continue
        if getattr(mod, "is_fast_path_available", None) is True:
            saved_flags.append(mod)
            mod.is_fast_path_available = False
    saved_inst = []
    saved_globals = []
    if model is not None:
        _GDN_FUNCS = (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "torch_causal_conv1d_update",
        )

        def _torch_body(obj):
            """Follow the hub wrapper's __wrapped__ chain down to a pure-torch body defined in a
            transformers modeling module; None if the object is not such a wrapper."""
            seen, cur = 0, obj
            while callable(cur) and hasattr(cur, "__wrapped__") and seen < 8:
                cur = cur.__wrapped__
                seen += 1
            if seen and callable(cur) and str(getattr(cur, "__module__", "")).startswith("transformers."):
                return cur
            return None

        gdn_modules = {}
        for m in model.modules():
            if all(hasattr(m, a) for a in ("A_log", "dt_bias", "conv1d", "norm")):
                for klass in type(m).__mro__:
                    modname = getattr(klass, "__module__", None)
                    if modname and modname in _sys.modules:
                        gdn_modules[modname] = _sys.modules[modname]
        for mm in gdn_modules.values():
            for fn_name in _GDN_FUNCS:
                obj = getattr(mm, fn_name, None)
                inner = _torch_body(obj)
                if inner is not None and inner is not obj:
                    saved_globals.append((mm, fn_name, obj))
                    setattr(mm, fn_name, inner)
        for m in model.modules():
            if not (hasattr(m, "chunk_gated_delta_rule") and hasattr(m, "recurrent_gated_delta_rule")):
                continue

            def _find(name, module=m):
                for klass in type(module).__mro__:
                    mm_ = _sys.modules.get(getattr(klass, "__module__", ""), None)
                    obj = getattr(mm_, name, None) if mm_ is not None else None
                    if callable(obj):
                        return _torch_body(obj) or obj
                return None

            mm = _sys.modules.get(type(m).__module__)
            torch_chunk = _find("torch_chunk_gated_delta_rule")
            torch_rec = _find("torch_recurrent_gated_delta_rule")
            torch_conv_upd = _find("torch_causal_conv1d_update")
            if torch_chunk is None or torch_rec is None:
                _warn_once(
                    "gdn_fallback",
                    f"{type(m).__name__}: no torch gated-delta-rule twins in "
                    f"{getattr(mm, '__name__', '?')}; leaving its kernels in place",
                )
                continue
            prev = {
                k: getattr(m, k, None)
                for k in (
                    "causal_conv1d_fn",
                    "causal_conv1d_update",
                    "chunk_gated_delta_rule",
                    "recurrent_gated_delta_rule",
                )
            }
            prev_norm = getattr(m, "norm", None)
            m.causal_conv1d_fn = None
            if torch_conv_upd is not None:
                m.causal_conv1d_update = torch_conv_upd
            m.chunk_gated_delta_rule = torch_chunk
            m.recurrent_gated_delta_rule = torch_rec
            new_norm = None
            if prev_norm is not None and type(prev_norm).__name__ == "FusedRMSNormGated":
                gated_cls = None
                for klass in type(m).__mro__:
                    mm_ = _sys.modules.get(getattr(klass, "__module__", ""), None)
                    if mm_ is None:
                        continue
                    gated_cls = next(
                        (
                            getattr(mm_, n)
                            for n in dir(mm_)
                            if n.endswith("RMSNormGated")
                            and n != "FusedRMSNormGated"
                            and isinstance(getattr(mm_, n), type)
                        ),
                        None,
                    )
                    if gated_cls is not None:
                        break
                w = getattr(prev_norm, "weight", None)
                eps = getattr(prev_norm, "eps", None)
                if gated_cls is None or w is None or eps is None:
                    _warn_once(
                        "gdn_norm",
                        f"{type(m).__name__}: cannot rebuild a torch twin of "
                        f"{type(prev_norm).__name__}; its kernel stays in place",
                    )
                else:
                    new_norm = gated_cls(w.shape[0], eps=float(eps))
                    new_norm.weight = w
                    m.norm = new_norm
            saved_inst.append((m, prev, prev_norm if new_norm is not None else None))
    try:
        yield
    finally:
        for mod in saved_flags:
            mod.is_fast_path_available = True
        for mm, fn_name, obj in saved_globals:
            setattr(mm, fn_name, obj)
        for m, prev, prev_norm in saved_inst:
            for k, v in prev.items():
                setattr(m, k, v)
            if prev_norm is not None:
                m.norm = prev_norm


@contextlib.contextmanager
def no_activation_checkpointing(model):
    """Run the wrapped forward directly, bypassing activation checkpointing."""
    hf_flags = []
    wrappers = []
    for m in model.modules():
        if getattr(m, "gradient_checkpointing", False) is True:
            hf_flags.append(m)
            m.gradient_checkpointing = False
        ckpt_fn = getattr(m, "checkpoint_fn", None)
        if ckpt_fn is not None and hasattr(m, "_checkpoint_wrapped_module"):
            wrappers.append((m, ckpt_fn))
            m.checkpoint_fn = _passthrough_checkpoint
    try:
        yield
    finally:
        for m in hf_flags:
            m.gradient_checkpointing = True
        for m, fn in wrappers:
            m.checkpoint_fn = fn


def _first_tensor(output):
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            if torch.is_tensor(o):
                return o, i
    return None, None


def _make_tangent_hook(layer, scale: float, counter: dict):
    """Add Delta_l x = -(scale * U_l)(A_l x) to the module output's tangent."""
    U = layer.last_u
    A = layer.A

    def hook(mod, inputs, output):
        counter["calls"] = counter.get("calls", 0) + 1
        out, idx = _first_tensor(output)
        if out is None or not inputs or not torch.is_tensor(inputs[0]):
            counter["skipped"] = counter.get("skipped", 0) + 1
            return None
        x = fwAD.unpack_dual(inputs[0]).primal
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        t = -scale * ((x2 @ A.T) @ U.T)
        prim = fwAD.unpack_dual(out).primal
        t = t.reshape(prim.shape).to(prim.dtype)
        new = out + fwAD.make_dual(torch.zeros_like(prim), t)
        if idx is None:
            return new
        lst = list(output)
        lst[idx] = new
        return type(output)(lst) if isinstance(output, tuple) else lst

    return hook


def _make_experts_tangent_hook(mod, fused, scale: float, counter: dict):
    """Tangent of a fused-experts module's output w.r.t. its two sketched tensors."""
    gu = next((L for L in fused if L._pname == "gate_up_proj"), None)
    dn = next((L for L in fused if L._pname == "down_proj"), None)
    if gu is None and dn is None:
        return None

    def hook(mod, inputs, output):
        counter["calls"] = counter.get("calls", 0) + 1
        out, idx = _first_tensor(output)
        if out is None or len(inputs) < 3 or not torch.is_tensor(inputs[0]):
            counter["skipped"] = counter.get("skipped", 0) + 1
            return None
        x = fwAD.unpack_dual(inputs[0]).primal
        top_idx = fwAD.unpack_dual(inputs[1]).primal if torch.is_tensor(inputs[1]) else inputs[1]
        top_w = fwAD.unpack_dual(inputs[2]).primal
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        prim = fwAD.unpack_dual(out).primal
        t = torch.zeros(x2.shape[0], prim.shape[-1], device=x2.device, dtype=torch.float32)
        Wgu = fwAD.unpack_dual(mod.gate_up_proj).primal
        Wd = fwAD.unpack_dual(mod.down_proj).primal
        E, two_i, H = Wgu.shape
        top_idx = top_idx.reshape(x2.shape[0], -1)
        top_w = top_w.reshape(x2.shape[0], -1).to(torch.float32)
        gu_A = gu.A if gu is not None else None
        dn_A = dn.A if dn is not None else None
        for e in torch.unique(top_idx).tolist():
            if e >= E:
                continue
            tok, pos = torch.where(top_idx == e)
            xe = x2[tok]
            g, u = (xe @ Wgu[e].to(torch.float32).T).chunk(2, dim=-1)
            if gu is not None:
                dgu = -scale * ((xe @ gu_A.T) @ gu.last_u[e * two_i : (e + 1) * two_i].T)
                dg, du = dgu.chunk(2, dim=-1)
            else:
                dg = du = torch.zeros_like(g)
            h_d = mod.act_fn(fwAD.make_dual(g, dg)) * fwAD.make_dual(u, du)
            h, dh = fwAD.unpack_dual(h_d)
            dy = dh @ Wd[e].to(torch.float32).T
            if dn is not None:
                dy = dy - scale * ((h @ dn_A.T) @ dn.last_u[e * H : (e + 1) * H].T)
            t.index_add_(0, tok, dy * top_w[tok, pos, None])
        t = t.reshape(prim.shape).to(prim.dtype)
        new = out + fwAD.make_dual(torch.zeros_like(prim), t)
        if idx is None:
            return new
        lst = list(output)
        lst[idx] = new
        return type(output)(lst) if isinstance(output, tuple) else lst

    return hook


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:

    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 1)


def _action_positions(T: int, action_mask: torch.Tensor):
    """Logit positions whose next token is a generated action. logits[t] predicts
    token[t+1]; the action span is the last action_mask.shape[1] columns of the
    shifted [:, :-1] log-probs (molt's convention)."""
    A = action_mask.shape[-1]
    start = (T - 1) - A
    pos = torch.arange(start, T - 1, device=action_mask.device)
    return pos, action_mask.bool()


def _stride_select(n: int, k: int, device) -> torch.Tensor:
    """Up to k indices spread evenly over range(n); deterministic."""
    if k <= 0 or n <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    step = max(1, n // k)
    return torch.arange(0, n, step, device=device)[:k]


class KLProbe:
    """Stash a few rows of the current micro-batch, then predict (and optionally
    verify) the KL that the pending merge will produce on them."""

    def __init__(self, n_seqs: int = 2, max_tokens: int = 0, check_tokens: int = 256, temperature: float = 1.0):
        self.n_seqs = int(n_seqs)
        self.max_tokens = int(max_tokens)
        self.check_tokens = int(check_tokens)
        self.temperature = float(temperature)
        self.seqs = None
        self.attn = None
        self.act = None
        self._check = None
        self.last = {}

    def stash(self, sequences, attention_mask, action_mask) -> None:
        n = min(self.n_seqs, sequences.shape[0])
        if n <= 0:
            return

        self.seqs = sequences[:n].detach().clone()
        self.attn = attention_mask[:n].detach().clone()
        self.act = action_mask[:n].detach().clone()

    def ready(self) -> bool:
        return self.seqs is not None

    def _rows(self) -> int:
        """Rows to probe, agreed across ranks (min), so every rank runs the same number
        of forwards."""
        n_local = 0 if self.seqs is None else int(self.seqs.shape[0])
        return all_reduce_min_int(n_local)

    def _slice(self, i: int, device):
        seq, attn, act = self.seqs[i : i + 1], self.attn[i : i + 1], self.act[i : i + 1]
        if self.max_tokens > 0 and seq.shape[1] > self.max_tokens:
            seq, attn = seq[:, -self.max_tokens :], attn[:, -self.max_tokens :]
            act = act[:, -min(act.shape[1], self.max_tokens - 1) :]
        return seq.to(device), attn.to(device), act.to(device)

    def _kept_positions(self, seq, attn, act):
        T = seq.shape[1]
        pos, amask = _action_positions(T, act)
        keep = amask[0] & attn[0, pos + 1].bool()
        return pos[keep]

    @torch.no_grad()
    def predict(self, model, layers, scale: float) -> dict:
        """Predicted KL of the step {-(scale*U_l) A_l} on the stashed rows.

        Returns per-token means over action tokens, reduced over ranks:
          kl_pred      1/2 * Var_{v~pi}(zdot_v)          (full vocabulary; used for alpha)
          kl_pred_tok  1/2 * mean g_t^2, g_t = zdot_{a_t} - E_pi zdot   (sampled-token estimate)
        """
        n = self._rows()
        out = {"kl_pred": float("nan"), "kl_pred_tok": float("nan"), "n_tokens": 0}
        self._check = None
        if n == 0:
            _warn_once("norows", "no micro-batch was stashed before the merge; nothing to probe")
            return out
        device = _model_device(model, layers)

        def _fused(L):
            return len(getattr(L, "wshape", ()) or ()) == 3

        hooked = [
            L for L in layers if getattr(L, "_module", None) is not None and L.last_u is not None and not _fused(L)
        ]
        by_mod: dict = {}
        for L in layers:
            if (
                _fused(L)
                and getattr(L, "_module", None) is not None
                and L.last_u is not None
                and getattr(L, "_pname", None) in ("gate_up_proj", "down_proj")
            ):
                by_mod.setdefault(id(L._module), (L._module, []))[1].append(L)
        n_fused_hooked = sum(len(v[1]) for v in by_mod.values())
        missing = len(layers) - len(hooked) - n_fused_hooked
        if missing:
            _warn_once(
                "fused",
                f"{missing} sketched tensor(s) have no module hook (fused experts in "
                "an unrecognised layout); their contribution is missing from the "
                "predicted KL",
            )
        counter = {}
        handles = [L._module.register_forward_hook(_make_tangent_hook(L, scale, counter)) for L in hooked]
        for mod, Ls in by_mod.values():
            h = _make_experts_tangent_hook(mod, Ls, scale, counter)
            if h is not None:
                handles.append(mod.register_forward_hook(h, prepend=True))
        sum_full = sum_tok = 0.0
        n_tok = 0
        check = {}
        per_row_check = max(1, self.check_tokens // n) if self.check_tokens > 0 else 0
        try:
            with probe_kernels(model), fwAD.dual_level():
                for i in range(n):
                    seq, attn, act = self._slice(i, device)
                    z = model(
                        input_ids=seq, attention_mask=attn, position_ids=_position_ids(attn), use_cache=False
                    ).logits
                    zp, zt = fwAD.unpack_dual(z)
                    if zt is None:
                        _warn_once(
                            "notangent",
                            "forward AD produced no tangent at the logits: "
                            + _describe_tangent_loss(model, counter, len(hooked)),
                        )
                        return out
                    pos = self._kept_positions(seq, attn, act)
                    if pos.numel() == 0:
                        continue
                    tgt = seq[0, pos + 1]
                    f_full, f_tok = _fisher_terms(zp[0], zt[0], pos, tgt, self.temperature)
                    sum_full += f_full
                    sum_tok += f_tok
                    n_tok += int(pos.numel())
                    if per_row_check:
                        sel = pos[_stride_select(pos.numel(), per_row_check, pos.device)]
                        lp = (zp[0][sel].to(torch.float32) / self.temperature).log_softmax(-1)
                        check[i] = (sel, lp)
        finally:
            for h in handles:
                h.remove()
            _reshard(model)
        tot = _all_reduce_sum([sum_full, sum_tok, float(n_tok)])
        if tot[2] > 0:
            out = {"kl_pred": 0.5 * tot[0] / tot[2], "kl_pred_tok": 0.5 * tot[1] / tot[2], "n_tokens": int(tot[2])}
        self._check = check if per_row_check else None
        self.last = dict(out)
        return out

    @torch.no_grad()
    def realised(self, model) -> dict:
        """Exact KL(pi_before || pi_after) on the check positions, after merge() wrote the
        weights. Same eager attention as the prediction, so the two forwards differ only by
        the step (and by the bf16 rendering of it, which is the point of checking).

        Runs the agreed number of forwards on every rank even when this rank has nothing
        to check (rank-local state must not change the collective count)."""
        n = self._rows()
        out = {"kl_real": float("nan"), "n_check": 0}
        if n == 0:
            return out
        device = _model_device(model, None)
        by_row = self._check or {}
        s_kl, n_chk = 0.0, 0
        with probe_kernels(model):
            for i in range(n):
                seq, attn, _act = self._slice(i, device)
                z = model(
                    input_ids=seq, attention_mask=attn, position_ids=_position_ids(attn), use_cache=False
                ).logits[0]
                if i not in by_row:
                    continue
                pos, lp0 = by_row[i]
                lp1 = (z[pos].to(torch.float32) / self.temperature).log_softmax(-1)
                kl = (lp0.exp() * (lp0 - lp1)).sum(-1)
                s_kl += float(kl.sum().item())
                n_chk += int(pos.numel())
        _reshard(model)
        tot = _all_reduce_sum([s_kl, float(n_chk)])
        if tot[1] > 0:
            out = {"kl_real": tot[0] / tot[1], "n_check": int(tot[1])}
        self._check = None
        self.last.update(out)
        return out


def _reshard(model) -> None:
    """A no_grad forward leaves the FSDP2 root's parameters unsharded until the next
    backward (nothing else reshards them). Give the memory back now; the next training
    forward all-gathers again as usual. No-op for a non-FSDP model."""
    fn = getattr(model, "reshard", None)
    if callable(fn):
        try:
            fn()
        except (AttributeError, ValueError, RuntimeError) as exc:
            _warn_once("reshard", f"model.reshard() failed after the probe ({exc!r})")


def _describe_tangent_loss(model, counter, n_hooked) -> str:
    """Why did the tangent not reach the logits? Either the hooks never ran (the sketched
    modules are not on the forward path of this model object) or something between them and
    the head dropped it (a custom autograd.Function -- e.g. a REENTRANT checkpoint wrapper or
    a fused kernel -- runs its forward with forward-grad mode off and returns plain tensors).
    List what is on the module tree so the log says which."""
    import collections

    classes = collections.Counter(type(m).__name__ for m in model.modules())
    ckpt = collections.Counter(
        type(m).__name__
        + (
            "(fn={})".format(getattr(getattr(m, "checkpoint_fn", None), "__name__", "?"))
            if hasattr(m, "checkpoint_fn")
            else ""
        )
        for m in model.modules()
        if hasattr(m, "checkpoint_fn")
        or hasattr(m, "_checkpoint_wrapped_module")
        or getattr(m, "gradient_checkpointing", False)
    )
    impl = getattr(getattr(model, "config", None), "_attn_implementation", "?")
    return (
        f"hooks registered={n_hooked} fired={counter.get('calls', 0)} "
        f"skipped(non-tensor io)={counter.get('skipped', 0)}; attn={impl}; "
        f"checkpoint-like modules={dict(ckpt) or 'none'}; "
        f"module classes={dict(classes.most_common(12))}. The step is left at the target "
        "value this merge."
    )


def _model_device(model, layers):
    if layers:
        for L in layers:
            if L.last_u is not None:
                return L.last_u.device
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _fisher_terms(zp, zt, pos, tgt, temperature, chunk=512):
    """Sum over the given positions of  Var_pi(zdot)  and  (zdot_a - E_pi zdot)^2, on the
    temperature-scaled distribution pi = softmax(z / T). Chunked over positions: a full
    fp32 [T, V] copy of a 152k-vocab model is 2.5 GB at T=4096."""
    s_full = s_tok = 0.0
    for c in range(0, pos.numel(), chunk):
        p = pos[c : c + chunk]
        z = zp[p].to(torch.float32) / temperature
        d = zt[p].to(torch.float32) / temperature
        pi = z.softmax(-1)
        mean = (pi * d).sum(-1, keepdim=True)
        dc = d - mean
        s_full += float((pi * dc * dc).sum().item())
        g = dc.gather(-1, tgt[c : c + chunk].unsqueeze(-1)).squeeze(-1)
        s_tok += float((g * g).sum().item())
    return s_full, s_tok


def step_scale(kl_pred: float, budget: float, alpha_max: float) -> float:
    """alpha = sqrt(budget / kl_pred), capped. Non-finite or zero prediction -> 1.0."""
    if not (math.isfinite(kl_pred) and kl_pred > 0.0) or budget <= 0.0:
        return 1.0
    return float(min(math.sqrt(budget / kl_pred), alpha_max))
