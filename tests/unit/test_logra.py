"""Numerical contracts for gradient compression and checkpoint continuation."""

import copy

import pytest
import torch

from molt.trainer.logra.optimizer import LograConfig, LograOptimizer
from molt.trainer.logra.sketch_linear import install_sketches


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(31, 16)
        self.proj = torch.nn.Linear(16, 12, bias=False)
        self.lm_head = torch.nn.Linear(12, 31, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, tokens):
        return self.lm_head(self.proj(self.embed_tokens(tokens)).tanh())


@pytest.mark.parametrize("projection", ["rademacher", "gaussian", "sparse"])
def test_sketch_matches_projected_dense_gradient(projection):
    torch.manual_seed(3)
    dense = TinyPolicy()
    compressed = copy.deepcopy(dense)
    cfg = LograConfig(rank=4, proj_kind=projection, optimizer="rowadam")
    layers = install_sketches(compressed, cfg)
    for _ in range(3):
        tokens = torch.randint(31, (2, 7))
        target = torch.randint(31, (2, 7))
        for model in (dense, compressed):
            torch.nn.functional.cross_entropy(model(tokens).flatten(0, 1), target.flatten()).backward()
    assert len(layers) == 1
    expected = dense.proj.weight.grad @ layers[0].A.T
    torch.testing.assert_close(layers[0].S, expected, atol=2e-7, rtol=2e-5)
    assert all(parameter.grad is None for parameter in compressed.parameters())


def test_resume_preserves_moments_projection_and_next_update():
    torch.manual_seed(12)
    model = TinyPolicy()
    cfg = LograConfig(rank=4, optimizer="rowadam", max_sketch_norm=0, max_update_ratio=6e-5)
    optimizer = LograOptimizer(install_sketches(model, cfg), cfg)
    optimizer.model = model
    for _ in range(3):
        model(torch.randint(31, (2, 7))).square().sum().backward()
        optimizer.step()
    optimizer.mismatch_ema = 1.2e-5
    model_state = copy.deepcopy(model.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    resumed = TinyPolicy()
    resumed.load_state_dict(model_state)
    resumed_optimizer = LograOptimizer(install_sketches(resumed, cfg), cfg)
    resumed_optimizer.model = resumed
    resumed_optimizer.load_state_dict(state)
    tokens = torch.randint(31, (2, 7))
    for current, opt in ((model, optimizer), (resumed, resumed_optimizer)):
        current(tokens).square().sum().backward()
        opt.step()
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert optimizer.row_t == resumed_optimizer.row_t
    assert optimizer.round == resumed_optimizer.round
    assert optimizer.mismatch_ema == resumed_optimizer.mismatch_ema
    torch.testing.assert_close(optimizer.layers[0].v_row, resumed_optimizer.layers[0].v_row, rtol=0, atol=0)


def test_mismatch_excludes_truncated_responses():
    model = TinyPolicy()
    cfg = LograConfig(rank=4, optimizer="rowadam")
    optimizer = LograOptimizer(install_sketches(model, cfg), cfg)
    optimizer.model = model
    mask = torch.tensor([[True, True, False], [True, True, True]])
    lp = torch.zeros(2, 3)
    rollout_lp = torch.tensor([[0.1, 0.3, 100.0], [100.0, 100.0, 100.0]])
    optimizer.observe(torch.ones(2, 4), None, mask, lp, rollout_lp, [False, True])
    optimizer.step()
    assert optimizer.mismatch_ema == pytest.approx(0.2)
    assert optimizer.last_stats["kl_mismatch_valid_tokens"] == 2


def test_predicted_kl_controls_the_applied_update():
    from types import SimpleNamespace

    from molt.trainer.logra.kl_probe import KLProbe

    class ProbePolicy(TinyPolicy):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=super().forward(input_ids))

    torch.manual_seed(4)
    model = ProbePolicy()
    cfg = LograConfig(
        rank=4, optimizer="rowadam", max_sketch_norm=0, max_update_ratio=0.02, kl_mode="auto", kl_budget=1e-5
    )
    optimizer = LograOptimizer(install_sketches(model, cfg), cfg)
    optimizer.model = model
    optimizer.probe = KLProbe(n_seqs=1, check_tokens=0)
    tokens = torch.randint(31, (1, 7))
    mask = torch.ones(1, 6, dtype=torch.bool)
    before = model(tokens).logits
    optimizer.observe(tokens, torch.ones_like(tokens), mask, torch.zeros(1, 6), None)
    before.square().sum().backward()
    optimizer.step()
    after = model(tokens).logits
    p = before.detach().double().log_softmax(-1)[:, :-1]
    q = after.detach().double().log_softmax(-1)[:, :-1]
    realised = (p.exp() * (p - q)).sum(-1).mean().item()
    assert optimizer.last_stats["kl_pred"] == pytest.approx(cfg.kl_budget, rel=1e-6)
    assert realised == pytest.approx(cfg.kl_budget, rel=0.06)


@pytest.mark.parametrize("entropy", [False, True])
def test_chunked_head_matches_dense_logits_and_input_gradient(entropy):
    from molt.models.chunked_head import chunked_head, reference_head

    torch.manual_seed(8)
    hidden = torch.randn(2, 7, 12, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(31, 12)
    labels = torch.randint(31, (2, 7))
    chunked = chunked_head(hidden, weight, labels, temperature=0.7, chunk=3, want_entropy=entropy)
    lp, ent = reference_head(other, weight, labels, temperature=0.7, want_entropy=entropy)
    torch.testing.assert_close(chunked.log_probs, lp, atol=2e-6, rtol=2e-6)
    a, b = chunked.log_probs.sum(), lp.sum()
    if entropy:
        torch.testing.assert_close(chunked.entropy, ent, atol=2e-6, rtol=2e-6)
        a, b = a + chunked.entropy.sum(), b + ent.sum()
    a.backward()
    b.backward()
    torch.testing.assert_close(hidden.grad, other.grad, atol=2e-5, rtol=2e-5)


def test_annealed_budget_endpoints_and_mismatch_floor():
    from types import SimpleNamespace

    model = TinyPolicy()
    cfg = LograConfig(
        rank=4, optimizer="rowadam", kl_mode="auto", kl_budget=2e-4, kl_budget_final=2e-5, kl_budget_steps=300
    )
    optimizer = LograOptimizer(install_sketches(model, cfg), cfg)
    optimizer.model = model
    optimizer.probe = SimpleNamespace(
        ready=lambda: True,
        predict=lambda *args, **kwargs: {"kl_pred": 2e-4, "kl_pred_tok": 2e-4},
    )
    optimizer.mismatch_ema = 3e-4
    for step, budget in ((0, 2e-4), (299, 2e-5), (350, 2e-5)):
        optimizer.round = step
        stats = {"update_scale": 1.0}
        scale = optimizer.kl_scale(optimizer, stats)
        assert stats["kl_budget_sched"] == pytest.approx(budget)
        assert stats["kl_budget_eff"] == pytest.approx(budget / 2)
        assert stats["kl_pred"] == pytest.approx(budget / 2)
        assert 2e-4 * scale**2 == pytest.approx(budget / 2)
