"""LOGRA outer optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .projection import layer_seed, make_projection
from .rounding import stable_layer_id, stochastic_round_


def _sum_across_ranks(x: float) -> float:
    """Sum a scalar over the default process group. Identity when not distributed.

    Every rank must end a merge with the SAME trust-region scale, or the shards of one
    weight matrix are scaled inconsistently and the low-rank payload rank 0 ships no
    longer describes what the other ranks applied.
    """
    import torch.distributed as _dist

    if not _dist.is_available() or not _dist.is_initialized() or _dist.get_world_size() == 1:
        return float(x)
    t = torch.tensor([float(x)], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
    return float(t.item())


def local_rows(W: torch.Tensor) -> tuple[int, int]:
    """(row offset, row count) of this rank's slice of W."""
    try:
        from torch.distributed.tensor import DTensor, Shard
    except ImportError:
        return 0, W.shape[0]
    if not isinstance(W, DTensor):
        return 0, (W.numel() // W.shape[-1] if W.ndim > 1 else W.shape[0])
    placements = W.placements
    if any(isinstance(pl, Shard) and pl.dim != 0 for pl in placements):
        raise NotImplementedError(f"LOGRA merge supports row-sharded (Shard(0)) weights only, got {placements}")
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape, offset = compute_local_shape_and_global_offset(W.shape, W.device_mesh, placements)
    return int(offset[0]), int(shape[0])


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization (Muon). Returns ~U V^T for G = U S V^T.

    Transposed so the Gram matrix is over the SHORT side: our sketches are [d, r] with
    r=16, so this costs O(d r^2) -- Muon is normally expensive on big square matrices and
    is nearly free here.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    flip = X.shape[0] > X.shape[1]
    if flip:
        X = X.T

    n = X.norm()
    X = X / torch.clamp(n, min=torch.finfo(X.dtype).tiny) if n > 0 else X
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if flip else X


@dataclass
class LograConfig:
    rank: int = 16
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0

    merge_tokens: int = 1_000_000
    proj_kind: str = "rademacher"
    reseed: bool = True

    reseed_every: int = 1

    moment_policy: str = "reset"
    max_sketch_norm: float = 1.0

    max_update_ratio: float = 0.001

    update_mode: str = "target"

    sketch_fused_experts: bool = True
    merge_precision: str = "stochastic"

    master_weights: bool = True

    preserve_layer_scale: bool = True

    grad_cos_diag: bool = False

    u_wire_dtype: torch.dtype | None = None

    optimizer: str = "sgd"
    momentum: float = 0.0

    kl_mode: str = "off"
    kl_budget: float = 0.0
    kl_probe_seqs: int = 2
    kl_probe_max_tokens: int = 0
    kl_alpha_max: float = 3.0
    kl_check_tokens: int = 256

    kl_mismatch_adjust: bool = True
    kl_mismatch_ema: float = 0.3
    kl_mismatch_floor: float = 0.5

    kl_refuse_max: int = 20
    kl_real_feedback: bool = False

    kl_budget_final: float = 0.0
    kl_budget_steps: int = 0
    eps_relative: bool = True
    muon_momentum: float = 0.95

    muon_ns_steps: int = 10
    base_seed: int = 1234

    def __post_init__(self):

        for field_name, allowed in (
            ("optimizer", {"adam", "muon", "sgd", "rowadam"}),
            ("proj_kind", {"rademacher", "gaussian", "sparse"}),
            ("merge_precision", {"stochastic", "nearest", "fp32"}),
            ("moment_policy", {"reset", "reproject", "keep"}),
            ("update_mode", {"cap", "target"}),
            ("kl_mode", {"off", "diag", "auto"}),
        ):
            v = getattr(self, field_name)
            if v not in allowed:
                raise ValueError(f"LograConfig.{field_name}={v!r} is not one of {sorted(allowed)}")

        if self.rank < 1:
            raise ValueError(f"LograConfig.rank must be >= 1, got {self.rank}")
        if self.lr <= 0:
            raise ValueError(f"LograConfig.lr must be > 0 (negative is ascent), got {self.lr}")
        if self.merge_tokens < 0:
            raise ValueError(
                f"LograConfig.merge_tokens must be >= 0 (0 = merge every optimizer step), got {self.merge_tokens}"
            )
        if self.max_sketch_norm < 0:
            raise ValueError(f"LograConfig.max_sketch_norm must be >= 0, got {self.max_sketch_norm}")
        if self.max_update_ratio < 0:
            raise ValueError(f"LograConfig.max_update_ratio must be >= 0, got {self.max_update_ratio}")
        if self.reseed_every < 1:
            raise ValueError(f"LograConfig.reseed_every must be >= 1, got {self.reseed_every}")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"LograConfig.momentum must be in [0, 1), got {self.momentum}")
        if not (0.0 <= self.muon_momentum < 1.0):
            raise ValueError(f"LograConfig.muon_momentum must be in [0, 1), got {self.muon_momentum}")
        if self.muon_ns_steps < 1:
            raise ValueError(f"LograConfig.muon_ns_steps must be >= 1, got {self.muon_ns_steps}")
        if self.kl_mode == "auto" and not self.kl_budget > 0:
            raise ValueError(f"LograConfig.kl_mode='auto' needs kl_budget > 0, got {self.kl_budget}")
        if self.kl_budget < 0:
            raise ValueError(f"LograConfig.kl_budget must be >= 0, got {self.kl_budget}")
        if self.kl_mode != "off" and self.kl_probe_seqs < 1:
            raise ValueError(f"LograConfig.kl_probe_seqs must be >= 1, got {self.kl_probe_seqs}")
        if not self.kl_alpha_max > 0:
            raise ValueError(f"LograConfig.kl_alpha_max must be > 0, got {self.kl_alpha_max}")
        if self.kl_check_tokens < 0:
            raise ValueError(f"LograConfig.kl_check_tokens must be >= 0, got {self.kl_check_tokens}")
        if not (0.0 < self.kl_mismatch_ema <= 1.0):
            raise ValueError(f"LograConfig.kl_mismatch_ema must be in (0, 1], got {self.kl_mismatch_ema}")
        if not (0.0 <= self.kl_mismatch_floor <= 1.0):
            raise ValueError(f"LograConfig.kl_mismatch_floor must be in [0, 1], got {self.kl_mismatch_floor}")
        if self.kl_refuse_max < 1:
            raise ValueError(f"LograConfig.kl_refuse_max must be >= 1, got {self.kl_refuse_max}")
        if self.kl_budget_final < 0 or self.kl_budget_steps < 0:
            raise ValueError("LograConfig.kl_budget_final and kl_budget_steps must be >= 0")
        if self.kl_budget_final > 0 and self.kl_budget_steps <= 0:
            raise ValueError("LograConfig.kl_budget_final needs kl_budget_steps > 0")
        if self.kl_probe_max_tokens < 0:
            raise ValueError(f"LograConfig.kl_probe_max_tokens must be >= 0, got {self.kl_probe_max_tokens}")


class LograLayer:
    """One sketched weight matrix: base W (d x k), projection A (r x k), sketch S (d x r)."""

    __slots__ = (
        "B",
        "S",
        "W",
        "W_master",
        "_A_host",
        "_Ac",
        "_C_prev",
        "_module",
        "_owner_model",
        "_pname",
        "d",
        "handle",
        "k",
        "last_seed",
        "last_u",
        "layer_id",
        "m",
        "name",
        "prev_S",
        "proj_kind",
        "r",
        "row0",
        "rows",
        "saw_untensored_output",
        "seed",
        "v",
        "v_row",
        "wshape",
    )

    def __init__(
        self,
        name: str,
        W: torch.Tensor,
        B: torch.Tensor,
        r: int,
        seed: int,
        kind: str,
        need_v: bool = True,
        need_m: bool = True,
    ):
        self.proj_kind = kind
        self.last_u = None
        self.name, self.W, self.B, self.r = name, W, B, r
        self.layer_id = stable_layer_id(name)
        self.last_seed = None

        self.W_master = None

        self.v_row = None

        self.prev_S = None
        self._C_prev = None
        self.saw_untensored_output = False
        self._module = None
        self._pname = None
        self._owner_model = None
        self.wshape = tuple(W.shape)
        self.d = int(torch.tensor(W.shape[:-1]).prod().item()) if W.ndim > 1 else W.shape[0]
        self.k = W.shape[-1]
        self.row0, self.rows = local_rows(W)
        self.seed = seed
        self.handle = None
        self._Ac = None

        self._A_host = None
        self.A = make_projection(r, self.k, seed, "cpu", torch.float32, kind)
        self.S = torch.zeros(self.d, r, device=W.device, dtype=torch.float32)

        self.m = torch.zeros_like(self.S) if need_m else None

        self.v = torch.zeros_like(self.S) if need_v else None

    @property
    def _device(self) -> torch.device:
        """Device the sketch runs on: the weight's (a DTensor reports its local device)."""
        W = self.W
        dev = getattr(W, "device", None)
        if dev is None:
            dev = self.S.device
        return dev

    @property
    def A(self) -> torch.Tensor:
        """The fp32 projection, on the compute device."""
        dev = self._device
        A = self._A_host
        if A.device == dev:
            return A
        return A.to(dev, non_blocking=True)

    @A.setter
    def A(self, value: torch.Tensor) -> None:
        A = value.detach()
        if A.dtype != torch.float32:
            A = A.to(torch.float32)
        if A.device.type != "cpu":
            A = A.to("cpu")
        elif A is value and value.requires_grad:
            A = A.clone()
        if torch.cuda.is_available() and self._device.type == "cuda" and not A.is_pinned():
            try:
                A = A.pin_memory()
            except RuntimeError:
                pass
        self._A_host = A
        self._Ac = None

    def cast_A(self, dtype):
        """Cached copy of A in the compute dtype, on the compute device. This is what the hooks
        multiply by every micro-batch; the fp32 canonical copy stays on the host (see `A`)."""
        Ah = self._A_host
        if self._Ac is None or self._Ac.dtype != dtype or self._Ac.shape != Ah.shape:
            self._Ac = Ah.to(device=self._device, dtype=dtype)
        return self._Ac


class LograOptimizer(torch.optim.Optimizer):
    """Accumulate gradient sketches and consolidate them into base weights."""

    def __init__(self, layers: list[LograLayer], cfg: LograConfig):
        super().__init__([layer.W for layer in layers], {"lr": cfg.lr})
        self.layers, self.cfg = layers, cfg
        self.model = None
        self.probe = None
        self.mismatch_sum = self.mismatch_tokens = 0.0
        self.mismatch_ema = None
        self.last_stats = {}
        self.delta_sync = False

        if (
            cfg.optimizer in ("sgd", "muon")
            and cfg.reseed
            and max(cfg.reseed_every, 1) == 1
            and cfg.moment_policy == "reset"
            and (cfg.momentum if cfg.optimizer == "sgd" else cfg.muon_momentum) > 0
        ):
            import warnings

            warnings.warn(
                f"reseed every merge + moment_policy='reset' zeroes the momentum buffer "
                f"each merge, so optimizer={cfg.optimizer!r} momentum is inert (bias "
                f"correction keeps the step size right, but nothing accumulates across "
                f"merges). Use reseed=False, reseed_every>1, or moment_policy='reproject' "
                f"if you want momentum to average sketch noise.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.last_sr_seed = 0
        self.tokens = 0
        self.round = 0
        self.adam_t = 0

        self.row_t = 0

    @torch.no_grad()
    def accumulate(self) -> None:
        """S += dL/dB, then release the autograd buffer. Call after each backward."""
        for L in self.layers:
            if L.B.grad is not None:
                L.S.add_(L.B.grad.to(torch.float32))
                L.B.grad = None

            if L.B.data.abs().max() != 0:
                raise RuntimeError(f"{L.name}: lora_B drifted from zero; something stepped it during the round")

    def add_tokens(self, n: int) -> None:
        self.tokens += int(n)

    def ready(self) -> bool:
        return self.tokens >= self.cfg.merge_tokens

    @torch.no_grad()
    def sketch_norm(self) -> float:
        return math.sqrt(sum(L.S.pow(2).sum().item() for L in self.layers))

    @torch.no_grad()
    def merge(
        self, dry_run: bool = False, cache_u: bool = False, sr_seed: int | None = None, step_scale_fn=None
    ) -> dict:
        """One outer Adam step, written into the base weights. Returns diagnostics."""
        cfg = self.cfg
        n = max(self.tokens, 1)
        b1, b2 = cfg.betas

        if dry_run:
            saved = [(None if L.m is None else L.m.clone(), None if L.v is None else L.v.clone()) for L in self.layers]
            saved_t = self.adam_t

            saved_wire = None if cache_u else [(L.last_u, L.last_seed) for L in self.layers]

        self.adam_t += 1
        self.row_t += 1

        raw_norm = self.sketch_norm() / n
        scale = 1.0
        if cfg.max_sketch_norm > 0 and raw_norm > cfg.max_sketch_norm:
            scale = cfg.max_sketch_norm / (raw_norm + 1e-12)

        stats = {
            "round": self.round,
            "tokens": self.tokens,
            "sketch_norm": raw_norm,
            "clip_scale": scale,
            "update_norm": 0.0,
        }

        bc1 = 1.0 - b1**self.adam_t
        bc2 = 1.0 - b2**self.adam_t

        eff_sr_seed = (
            int(sr_seed) if sr_seed is not None else (cfg.base_seed * 2_654_435_761 + self.round * 40_503) & 0x7FFFFFFF
        )
        self.last_sr_seed = eff_sr_seed

        cos_ip, cos_n2_cur, cos_n2_prev, cos_layers, cos_by_layer = 0.0, 0.0, 0.0, [], {}
        for L in self.layers:
            Shat = L.S.mul(scale / n)
            if cfg.grad_cos_diag:
                if L.prev_S is not None:
                    if L._C_prev is None:
                        ip = (Shat * L.prev_S).sum().item()
                    else:
                        ip = (Shat * (L.prev_S @ L._C_prev)).sum().item() / (L.k / L.r)
                    n2c, n2p = Shat.pow(2).sum().item(), L.prev_S.pow(2).sum().item()
                    cos_ip += ip
                    cos_n2_cur += n2c
                    cos_n2_prev += n2p
                    if n2c > 0 and n2p > 0:
                        cos_layers.append(ip / math.sqrt(n2c * n2p))
                        cos_by_layer[L.name] = cos_layers[-1]
                if not dry_run:
                    L.prev_S = Shat.detach().clone()
                    L._C_prev = None
            if cfg.optimizer == "sgd":
                if cfg.momentum > 0 and L.m is not None:
                    if L.m is None:
                        L.m = torch.zeros_like(L.S)
                    L.m.mul_(cfg.momentum).add_(Shat, alpha=1.0 - cfg.momentum)

                    U = L.m / (1.0 - cfg.momentum**self.adam_t)
                else:
                    U = Shat
            elif cfg.optimizer == "rowadam":
                row = Shat.pow(2).mean(dim=1, keepdim=True)
                if L.v_row is None:
                    L.v_row = torch.zeros_like(row)
                L.v_row.mul_(b2).add_(row, alpha=1.0 - b2)
                vr = (L.v_row / (1.0 - b2 ** max(self.row_t, 1))).sqrt()
                eps_r = (cfg.eps * vr.mean()).clamp_min(torch.finfo(vr.dtype).tiny) if cfg.eps_relative else cfg.eps
                D = 1.0 / (vr + eps_r)
                D = D / D.pow(2).mean().sqrt().clamp_min(torch.finfo(D.dtype).tiny)
                U = D * Shat
            elif cfg.optimizer == "muon":
                L.m.mul_(cfg.muon_momentum).add_(Shat)
                O = newton_schulz(L.m, cfg.muon_ns_steps)

                U = O * math.sqrt(L.d)
            else:
                L.m.mul_(b1).add_(Shat, alpha=1.0 - b1)
                L.v.mul_(b2).addcmul_(Shat, Shat, value=1.0 - b2)
                if not torch.isfinite(L.v).all():
                    raise FloatingPointError(
                        f"{L.name}: Adam's second moment overflowed fp32 (|Shat|max="
                        f"{Shat.abs().max().item():.3e}); the update would silently become "
                        "zero. Lower the learning rate, enable max_sketch_norm, or use "
                        "optimizer='muon' (scale-invariant, no second moment)."
                    )
                vhat = (L.v / bc2).sqrt_()

                eps = (cfg.eps * vhat.mean()).clamp_min(torch.finfo(vhat.dtype).tiny) if cfg.eps_relative else cfg.eps
                U = (L.m / bc1) / (vhat + eps)

            if cfg.preserve_layer_scale and cfg.optimizer != "sgd":
                U = U * (Shat.norm() / U.norm().clamp_min(torch.finfo(U.dtype).tiny))

            if cfg.weight_decay:
                Wl = L.W._local_tensor if hasattr(L.W, "_local_tensor") else L.W.data
                if Wl.ndim != 2:
                    Wl = Wl.view(-1, Wl.shape[-1])
                U = U + cfg.weight_decay * (Wl.to(torch.float32) @ L.A.T)

            scaled = U.mul(cfg.lr)

            L.last_u = scaled.detach().clone()

            L.last_seed = int(L.seed)
            Ul = scaled[L.row0 : L.row0 + L.rows] if L.rows != L.d else scaled
            stats["update_norm"] += (Ul @ L.A).pow(2).sum().item()

        if cfg.grad_cos_diag and cos_layers:
            stats["grad_cos_prev"] = cos_ip / math.sqrt(cos_n2_cur * cos_n2_prev)
            stats["grad_cos_prev_layer_mean"] = sum(cos_layers) / len(cos_layers)
            stats["grad_cos_prev_layer_min"] = min(cos_layers)
            stats["grad_cos_prev_layers"] = cos_by_layer

        usq = _sum_across_ranks(stats["update_norm"])
        raw_update = math.sqrt(usq)
        stats["update_norm"] = raw_update
        stats["update_scale"] = 1.0
        if cfg.max_update_ratio > 0:
            wsq = 0.0
            for L in self.layers:
                st = L.W._local_tensor if hasattr(L.W, "_local_tensor") else L.W.data
                wsq += st.to(torch.float32).pow(2).sum().item()
            wsq = _sum_across_ranks(wsq)

            stats["weight_norm"] = math.sqrt(wsq)
            budget = cfg.max_update_ratio * math.sqrt(wsq)
            if budget > 0 and raw_update > 0 and (cfg.update_mode == "target" or raw_update > budget):
                stats["update_scale"] = budget / (raw_update + 1e-30)
                stats["update_norm"] = budget

        if step_scale_fn is not None and not dry_run:
            alpha = float(step_scale_fn(self, stats))

            if not (math.isfinite(alpha) and alpha >= 0.0):
                raise FloatingPointError(f"step_scale_fn returned {alpha!r}; refusing to merge")
            stats["kl_alpha"] = alpha
            stats["update_scale"] *= alpha
            stats["update_norm"] *= alpha

        for L in self.layers:
            scaled = L.last_u
            if stats["update_scale"] != 1.0:
                scaled = scaled * stats["update_scale"]
            if cfg.u_wire_dtype is not None:
                scaled = scaled.to(cfg.u_wire_dtype).to(torch.float32)
            L.last_u = scaled
            Ul = scaled[L.row0 : L.row0 + L.rows] if L.rows != L.d else scaled

            from .delta_sync import _fixed_shape_gemm, exact_fp32_matmul

            A_dev = L.A
            with exact_fp32_matmul():
                if len(L.wshape) == 3:
                    dW = _fixed_shape_gemm(scaled, A_dev, L.wshape[1], L.row0, L.rows)
                elif cache_u and L.rows != L.d:
                    dW = (scaled @ A_dev)[L.row0 : L.row0 + L.rows]
                else:
                    dW = Ul @ A_dev

            if not dry_run:
                storage = L.W._local_tensor if hasattr(L.W, "_local_tensor") else L.W.data
                if storage.ndim != 2:
                    storage = storage.view(-1, storage.shape[-1])
                if cfg.master_weights and storage.dtype != torch.float32:
                    if getattr(L, "W_master", None) is None:
                        L.W_master = storage.to(torch.float32).clone()
                    L.W_master -= dW

                    storage.copy_(L.W_master.to(storage.dtype))
                else:
                    target = storage.to(torch.float32) - dW
                    if cfg.merge_precision == "stochastic":
                        stochastic_round_(storage, target, seed=eff_sr_seed, layer_id=L.layer_id, row0=L.row0)
                    else:
                        storage.copy_(target.to(storage.dtype))
            if not cache_u:
                L.last_u = None

        if dry_run:
            for L, (m0, v0) in zip(self.layers, saved):
                if m0 is not None and L.m is not None:
                    L.m.copy_(m0)
                elif m0 is None:
                    L.m = None
                if L.v is not None and v0 is not None:
                    L.v.copy_(v0)
            self.adam_t = saved_t
            if saved_wire is not None:
                for L, (u0, s0) in zip(self.layers, saved_wire):
                    L.last_u, L.last_seed = u0, s0
        else:
            self._reseed_and_reset()
        return stats

    @torch.no_grad()
    def _reseed_and_reset(self) -> None:
        cfg = self.cfg
        self.round += 1
        do_reseed = cfg.reseed and (self.round % max(cfg.reseed_every, 1) == 0)
        for L in self.layers:
            if do_reseed:
                new_seed = layer_seed(cfg.base_seed, self.round, L.name)
                A_new = make_projection(L.r, L.k, new_seed, L.A.device, torch.float32, cfg.proj_kind)
                if cfg.moment_policy == "reproject":
                    C = math.sqrt(L.r / L.k) * (L.A @ A_new.T)
                    if L.m is not None:
                        L.m = L.m @ C
                    if L.v is not None:
                        L.v = L.v @ C.pow(2)
                elif cfg.moment_policy == "reset":
                    if L.m is not None:
                        L.m.zero_()
                    if L.v is not None:
                        L.v.zero_()
                    self.adam_t = 0
                if cfg.grad_cos_diag and L.prev_S is not None:
                    L._C_prev = L.A @ A_new.T
                L.A, L.seed, L._Ac = A_new, new_seed, None
            L.S.zero_()
        self.tokens = 0

    def state_dict(self) -> dict:
        """Everything needed to resume. The Adam moments live here, not on the wire:
        the inference engine needs only the update U, but a restarted trainer needs the
        moments that produce future ones. A is not stored -- it regenerates from its seed."""
        return {
            "round": self.round,
            "adam_t": self.adam_t,
            "row_t": self.row_t,
            "tokens": self.tokens,
            "mismatch_ema": self.mismatch_ema,
            "layers": {
                L.name: {
                    "S": L.S.detach().to("cpu", copy=True),
                    "m": None if L.m is None else L.m.detach().to("cpu", copy=True),
                    "v": None if L.v is None else L.v.detach().to("cpu", copy=True),
                    "v_row": None if L.v_row is None else L.v_row.detach().to("cpu", copy=True),
                    "W_master": None if L.W_master is None else L.W_master.detach().to("cpu", copy=True),
                    "seed": L.seed,
                }
                for L in self.layers
            },
        }

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        from .projection import make_projection

        if (
            self.cfg.optimizer == "rowadam"
            and sd["round"] > 0
            and ("row_t" not in sd or any(e.get("v_row") is None for e in sd["layers"].values()))
        ):
            raise ValueError("RowAdam checkpoint lacks row_t/v_row; exact resume is impossible")
        self.round, self.adam_t, self.tokens = sd["round"], sd["adam_t"], sd["tokens"]
        self.row_t = int(sd.get("row_t", 0))
        self.mismatch_ema = sd.get("mismatch_ema")
        for L in self.layers:
            e = sd["layers"].get(L.name)
            if e is None:
                raise KeyError(f"checkpoint has no state for {L.name}")
            L.S.copy_(e["S"].to(L.S.device))
            if L.m is not None:
                if e.get("m") is not None:
                    L.m.copy_(e["m"].to(L.m.device))
                else:
                    L.m.zero_()

            if L.v is not None:
                if e["v"] is None:
                    L.v.zero_()
                else:
                    L.v.copy_(e["v"].to(L.v.device))
            vr = e.get("v_row")
            L.v_row = None if vr is None else vr.to(L.S.device, torch.float32, copy=True)
            wm = e.get("W_master")
            if wm is None:
                L.W_master = None
            else:
                L.W_master = wm.to(L.W.device, torch.float32)
            L.seed = int(e["seed"])
            L.A = make_projection(L.r, L.k, L.seed, "cpu", torch.float32, L.proj_kind)

    @torch.no_grad()
    def discard(self) -> None:
        """Throw away the round without touching W (used when the eval gate rejects)."""
        self._reseed_and_reset()

    @torch.no_grad()
    def observe(self, sequences, attention_mask, action_mask, log_probs, rollout_log_probs, truncated=None):
        """Collect action-token counts and trainer/rollout mismatch before backward."""
        self.tokens += int(action_mask.sum())
        if self.probe is not None:
            self.probe.stash(sequences, attention_mask, action_mask)
        if rollout_log_probs is None:
            return
        mask = action_mask.bool()
        if truncated is not None:
            mask = mask & ~torch.as_tensor(truncated, device=mask.device, dtype=torch.bool).reshape(-1, 1)
        delta = rollout_log_probs.detach().float() - log_probs.detach().float()
        self.mismatch_sum += float(torch.where(mask, delta, 0.0).sum())
        self.mismatch_tokens += float(mask.sum())

    @torch.no_grad()
    def step(self, closure=None):
        from .sketch_linear import all_reduce_sketches

        if closure is not None:
            raise ValueError("LOGRA receives sketches from backward; closures are unsupported")
        if self.model is None:
            raise RuntimeError("LOGRA must be attached to its policy model before stepping")
        totals = torch.tensor(
            [self.tokens, self.mismatch_sum, self.mismatch_tokens], dtype=torch.float64, device=self.layers[0].S.device
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        self.tokens = int(totals[0])
        valid = float(totals[2])
        raw = max(float(totals[1]) / valid, 0.0) if valid else float("nan")
        if valid:
            rate = self.cfg.kl_mismatch_ema
            self.mismatch_ema = raw if self.mismatch_ema is None else (1 - rate) * self.mismatch_ema + rate * raw
        self.mismatch_sum = self.mismatch_tokens = 0.0
        all_reduce_sketches(self.layers, average=False)
        self.last_stats = self.merge(
            cache_u=self.delta_sync, step_scale_fn=self.kl_scale if self.probe is not None else None
        )
        self.last_stats.update(kl_mismatch_raw=raw, kl_mismatch_valid_tokens=valid)

    def kl_scale(self, optimizer, stats):
        from .kl_probe import all_reduce_min_int, step_scale

        if not all_reduce_min_int(int(self.probe.ready())):
            raise RuntimeError("LOGRA predicted KL requires probe inputs on every training rank")
        pred = self.probe.predict(self.model, self.layers, scale=stats["update_scale"])
        if not math.isfinite(pred["kl_pred"]):
            raise RuntimeError("LOGRA KL prediction is not finite; refusing an uncontrolled update")
        cfg = self.cfg
        fraction = min(self.round / max(cfg.kl_budget_steps - 1, 1), 1.0)
        scheduled = cfg.kl_budget
        if cfg.kl_budget_final > 0 and cfg.kl_budget_steps > 0:
            scheduled = (
                cfg.kl_budget_final + (cfg.kl_budget - cfg.kl_budget_final) * (1 + math.cos(math.pi * fraction)) / 2
            )
        budget = scheduled
        if cfg.kl_mismatch_adjust and self.mismatch_ema is not None:
            budget = max(scheduled - self.mismatch_ema, cfg.kl_mismatch_floor * scheduled)
        alpha = step_scale(pred["kl_pred"], budget, cfg.kl_alpha_max)
        stats.update(pred)
        stats.update(
            kl_budget_sched=scheduled,
            kl_budget_eff=budget,
            kl_beta=self.mismatch_ema if self.mismatch_ema is not None else float("nan"),
            kl_alpha_raw=alpha,
            kl_pred_unscaled=pred["kl_pred"],
        )
        if cfg.kl_mode == "auto":
            stats["kl_pred"] *= alpha * alpha
            stats["kl_pred_tok"] *= alpha * alpha
            return alpha
        return 1.0
