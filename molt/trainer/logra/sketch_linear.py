"""Sketch accumulation via hooks, with no extra parameters and no forward override."""

from __future__ import annotations

import collections
import re

import torch
import torch.distributed as dist
from torch import nn

DEFAULT_TARGETS: tuple[str, ...] = ()


DEFAULT_EXCLUDE: tuple[str, ...] = (r"(^|\.)(\w+_)?(router|gate)(\.|$)",)


LANGUAGE_SUBTREES = ("model.language_model", "language_model", "model.text_model", "text_model", "model.decoder")


def language_prefix(model: nn.Module) -> str:
    """Dotted prefix of the text stack, or "" if the model is text-only."""
    for path in LANGUAGE_SUBTREES:
        try:
            model.get_submodule(path)
        except AttributeError:
            continue
        return path + "."
    return ""


def is_linear_like(mod) -> bool:
    """True for anything computing y = x W^T + b with a 2-D weight."""
    w = getattr(mod, "weight", None)
    return (
        w is not None and getattr(w, "ndim", 0) == 2 and hasattr(mod, "in_features") and hasattr(mod, "out_features")
    )


def _vocab_weight_ids(model: nn.Module) -> set[int]:
    """id() of the input and output embedding weights, including tied ones."""
    ids = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        fn = getattr(model, getter, None)
        if fn is None:
            continue
        try:
            mod = fn()
        except (AttributeError, NotImplementedError):  # Some model families omit this getter.
            continue
        w = getattr(mod, "weight", None)
        if w is not None:
            ids.add(id(w))
    return ids


SKIP_RECOMPUTE_STASH = False
_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)


def _in_backward() -> bool:
    """True while an autograd backward pass is executing on this thread, which for a forward
    hook means: this forward is a checkpoint RECOMPUTATION."""
    return _graph_task_id is not None and _graph_task_id() != -1


def _make_hook(layer):
    def forward_hook(mod, inputs, output):

        if not torch.is_tensor(output):
            if isinstance(output, (tuple, list)):
                output = next((o for o in output if torch.is_tensor(o)), None)
            if not torch.is_tensor(output):
                layer.saw_untensored_output = True
                return

        if not output.requires_grad:
            return
        if SKIP_RECOMPUTE_STASH and _in_backward():
            return
        if not inputs:
            layer.saw_untensored_output = True
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return

        with torch.no_grad():
            Ac = layer.cast_A(x.dtype)
            h = x.reshape(-1, x.shape[-1]) @ Ac.T

        def grad_hook(grad_out):
            go = grad_out.reshape(-1, grad_out.shape[-1])
            layer.S.add_((go.T @ h).to(torch.float32))

        output.register_hook(grad_hook)
        return

    return forward_hook


def _input_embedding(model: nn.Module) -> nn.Module:
    """The module the anchor hooks. Falls back to the first nn.Embedding for
    architectures that do not implement get_input_embeddings()."""
    fn = getattr(model, "get_input_embeddings", None)
    if fn is not None:
        try:
            mod = fn()
            if mod is not None:
                return mod
        except (AttributeError, NotImplementedError):
            pass
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            return mod
    raise RuntimeError(
        "no input embedding found; LOGRA needs one module to anchor the autograd graph "
        "(every parameter is frozen, so without an anchor no graph is built)"
    )


class _GraphAnchor:
    """A grad-requiring leaf kept OUT of the module, so state_dict stays clean.

    Every base parameter is frozen, so without this nothing in the forward requires
    grad, autograd builds no graph, and the gradient hooks never fire. Multiplying the
    embedding output by a scalar 1.0 leaf costs one elementwise op. Its .grad is unused.
    """

    def __init__(self):
        self.tensor = None
        self.handle = None

    def attach(self, model: nn.Module) -> None:
        self.detach()

        def hook(module, inputs, output):
            t = self.tensor
            if t is None or t.dtype != output.dtype or t.device != output.device:
                t = torch.ones((), dtype=output.dtype, device=output.device, requires_grad=True)
                self.tensor = t
            return output * t

        self.handle = _input_embedding(model).register_forward_hook(hook)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


_ANCHORS: dict[int, _GraphAnchor] = {}


def _anchor_for(model: nn.Module) -> _GraphAnchor:
    a = _ANCHORS.get(id(model))
    if a is None:
        a = _ANCHORS[id(model)] = _GraphAnchor()
    return a


class _AnchorProxy:
    """Back-compat shim: GRAPH_ANCHOR.attach(model) / .detach() still work, now per-model."""

    def attach(self, model: nn.Module) -> None:
        _anchor_for(model).attach(model)

    def detach(self, model: nn.Module | None = None) -> None:
        if model is not None:
            a = _ANCHORS.pop(id(model), None)
            if a is not None:
                a.detach()
            return
        for a in list(_ANCHORS.values()):
            a.detach()
        _ANCHORS.clear()

    @property
    def tensor(self):
        return next((a.tensor for a in _ANCHORS.values() if a.tensor is not None), None)

    @property
    def handle(self):
        return next((a.handle for a in _ANCHORS.values() if a.handle is not None), None)


GRAPH_ANCHOR = _AnchorProxy()


def select_modules(
    model: nn.Module,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    restrict_to_language_model: bool = True,
) -> list:
    """The modules LOGRA sketches. Single source of truth: the installer and any
    coverage probe must agree, or a probe reports a number the run does not use."""
    prefix = language_prefix(model) if restrict_to_language_model else ""
    banned = _vocab_weight_ids(model)
    pats = [re.compile(p) for p in exclude]

    hits = []
    for name, mod in model.named_modules():
        if prefix and not name.startswith(prefix):
            continue
        if not is_linear_like(mod):
            continue
        if id(mod.weight) in banned:
            continue
        if any(p.search(name) for p in pats):
            continue
        if targets and not any(name.endswith(t) for t in targets):
            continue
        hits.append((name, mod))
    return hits


def coverage_report(model: nn.Module, **kw) -> dict:
    """What was sketched and what was left out, for auditing a new architecture."""
    chosen = {n for n, _ in select_modules(model, **kw)}
    total = sum(p.numel() for p in model.parameters())
    sketched = 0
    skipped: dict[str, int] = collections.defaultdict(int)

    seen: set[int] = set()
    for name, mod in model.named_modules():
        if name in chosen:
            w = mod.weight
            sketched += w.numel()
            seen.add(id(w))

    try:
        import torch.nn.utils.parametrize as _pz

        for _n, _m in model.named_modules():
            if not _pz.is_parametrized(_m):
                continue
            for _pl in getattr(_m, "parametrizations", {}).values():
                if any(type(f).__name__ == "_SketchParametrization" for f in _pl):
                    orig = _pl.original
                    sketched += orig.numel()
                    seen.add(id(orig))
    except ImportError:
        pass
    for pname, prm in model.named_parameters():
        if id(prm) in seen:
            continue
        seen.add(id(prm))
        leaf = pname.rsplit(".", 1)[-1]
        parent = pname.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in pname else ""
        key = f"{parent}.{leaf}" if parent else leaf
        if prm.ndim != 2:
            key = f"{key} (ndim={prm.ndim})"
        skipped[key] += prm.numel()

    accounted = sketched + sum(skipped.values())
    return {
        "total": total,
        "sketched": sketched,
        "n_modules": len(chosen),
        "language_prefix": language_prefix(model),
        "skipped": dict(skipped),
        "fully_accounted": accounted == total,
        "accounted": accounted,
    }


class _SketchGrad(torch.autograd.Function):
    """Reduce a parameter's gradient to its sketch and drop the dense tensor."""

    @staticmethod
    def forward(ctx, W, layer):
        ctx.layer = layer
        return W

    @staticmethod
    def backward(ctx, gW):
        L = ctx.layer
        g2 = gW.reshape(-1, gW.shape[-1])
        L.S.add_((g2.to(torch.float32) @ L.cast_A(torch.float32).T).to(torch.float32))
        return torch.zeros_like(gW), None


class _SketchParametrization(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self._logra_layer = layer

    def forward(self, W):
        return _SketchGrad.apply(W, self._logra_layer)


def _needs_moment(cfg) -> bool:
    """Does this configuration ever read the first-moment buffer? Adam and Muon always; SGD only
    with momentum that can survive from one merge to the next -- a basis redrawn every merge
    under moment_policy="reset" zeroes it before each use, which makes the update Shat exactly
    (see LograOptimizer.merge), so the buffer is dead weight: 4 GB per rank on a 27B at r=256."""
    if cfg.optimizer == "rowadam":
        return False
    if cfg.optimizer != "sgd":
        return True
    if float(getattr(cfg, "momentum", 0.0)) <= 0:
        return False
    return not (bool(getattr(cfg, "reseed", False)) and getattr(cfg, "moment_policy", "reset") == "reset")


def is_fused_weight(name: str, prm: torch.Tensor, vocab_ids: set, rank: int, exclude: tuple = DEFAULT_EXCLUDE) -> bool:
    """A stack of real weight MATRICES that no linear-like module exposes."""
    if prm.ndim != 3 or not prm.is_floating_point() or id(prm) in vocab_ids:
        return False
    if any(re.search(pat, name) for pat in exclude):
        return False
    return prm.shape[-2] >= rank and prm.shape[-1] >= max(2 * rank, 16)


def install_fused_sketches(model: nn.Module, cfg, round_idx: int = 0, layers=None, exclude: tuple = DEFAULT_EXCLUDE):
    """Sketch fused >=3-D expert weights via a parametrization. Returns new layers.
    `exclude` are the caller's patterns (the same ones the linear-like selection honours)."""
    from torch.nn.utils import parametrize

    from .delta_sync import canonical_name
    from .optimizer import LograLayer
    from .projection import layer_seed

    vocab = _vocab_weight_ids(model)
    covered = {id(L.W) for L in (layers or [])}
    out = []
    for mod_name, mod in list(model.named_modules()):
        for pname, prm in list(mod.named_parameters(recurse=False)):
            full = f"{mod_name}.{pname}" if mod_name else pname
            if id(prm) in covered or not is_fused_weight(full, prm, vocab, cfg.rank, exclude):
                continue
            if parametrize.is_parametrized(mod, pname):
                continue
            L = LograLayer(
                full,
                prm,
                torch.zeros(0),
                cfg.rank,
                layer_seed(cfg.base_seed, round_idx, canonical_name(full)),
                cfg.proj_kind,
                need_v=(cfg.optimizer == "adam"),
                need_m=_needs_moment(cfg),
            )
            L.S = L.S.to(prm.device)
            L._owner_model = model
            L._module = mod
            L._pname = pname
            prm.requires_grad_(True)
            parametrize.register_parametrization(mod, pname, _SketchParametrization(L), unsafe=True)
            L.W = getattr(mod.parametrizations, pname).original
            out.append(L)
    return out


def install_sketches(
    model: nn.Module,
    cfg,
    round_idx: int = 0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    restrict_to_language_model: bool = True,
) -> list:
    """Freeze the model, attach sketch hooks to the target projections, anchor the graph."""
    from .delta_sync import canonical_name
    from .optimizer import LograLayer
    from .projection import layer_seed

    for p in model.parameters():
        p.requires_grad_(False)

    hits = select_modules(model, targets, exclude, restrict_to_language_model)

    fused_possible = bool(getattr(cfg, "sketch_fused_experts", True)) and any(p.ndim == 3 for p in model.parameters())
    if not hits and not fused_possible:
        raise RuntimeError(f"no linear-like modules to sketch (targets={targets}, exclude={exclude})")

    already = [n for n, m in hits if getattr(m, "_logra", None) is not None]
    if already:
        raise RuntimeError(
            f"install_sketches called twice: {len(already)} module(s) already sketched "
            f"(e.g. {already[0]}). A second install leaves two forward hooks per module, "
            "and remove_sketches on one list will not stop the other. Call "
            "remove_sketches() first."
        )

    GRAPH_ANCHOR.attach(model)

    layers = []
    for name, mod in hits:
        L = LograLayer(
            name,
            mod.weight,
            torch.zeros(0),
            cfg.rank,
            layer_seed(cfg.base_seed, round_idx, canonical_name(name)),
            cfg.proj_kind,
            need_v=(cfg.optimizer == "adam"),
            need_m=_needs_moment(cfg),
        )
        L.S = L.S.to(mod.weight.device)
        mod._logra = L
        L._module = mod
        L._owner_model = model
        L.handle = mod.register_forward_hook(_make_hook(L))
        L._owner_model = model
        layers.append(L)

    total_p = sum(p.numel() for p in model.parameters())
    sketched_p = sum(m.weight.numel() for _, m in hits)
    classes = collections.Counter(type(m).__name__ for _, m in hits)
    lp = language_prefix(model) or "(text-only model)"
    print(
        f"[LOGRA] sketching {len(hits)} projections "
        f"({sketched_p / 1e6:.0f}M / {total_p / 1e6:.0f}M params = "
        f"{100 * sketched_p / total_p:.1f}% coverage); classes={dict(classes)}; "
        f"language subtree={lp}",
        flush=True,
    )

    if getattr(cfg, "sketch_fused_experts", True):
        fused = install_fused_sketches(model, cfg, round_idx, layers, exclude=exclude)
        if fused and (not dist.is_initialized() or dist.get_rank() == 0):
            n_p = sum(int(torch.tensor(L.wshape).prod().item()) for L in fused)
            print(
                f"[LOGRA] + {len(fused)} fused expert tensor(s) ({n_p / 1e6:.1f}M params) "
                "sketched via parametrization (dense grad is transient, not stored)",
                flush=True,
            )
        layers = layers + fused
    if not layers:
        raise RuntimeError(
            f"nothing to sketch: no linear-like module and no fused expert tensor "
            f"survived (targets={targets}, exclude={exclude})"
        )
    return layers


def remove_sketches(layers) -> None:
    owners: dict = {}
    for L in layers:
        if getattr(L, "handle", None) is not None:
            L.handle.remove()
            L.handle = None

        mod = getattr(L, "_module", None)
        if mod is not None and getattr(mod, "_logra", None) is L:
            mod._logra = None
        owner = getattr(L, "_owner_model", None)
        if owner is not None:
            owners[id(owner)] = owner
    for m in owners.values():
        GRAPH_ANCHOR.detach(m)


def all_reduce_sketches(layers, average: bool = True) -> None:
    """One collective per merge. Without it each rank consolidates only its own
    shard of the batch."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    ws = dist.get_world_size()
    for L in layers:
        dist.all_reduce(L.S, op=dist.ReduceOp.SUM)
        if average:
            L.S.div_(ws)
