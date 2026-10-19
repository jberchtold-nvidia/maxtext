# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.
"""K3 native math and explicitly approximate TE adapter correctness."""

from pathlib import Path
from unittest import mock
import math
import numpy as np
import pytest
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.models import kimi_k3 as k3
from maxtext.layers import moe
from maxtext.utils import maxtext_utils

pytestmark = pytest.mark.cpu_only
BASE = str(Path(__file__).resolve().parents[2] / "src/maxtext/configs/base.yml")


def config(**kwargs):
  overrides = dict(
      model_name="kimi-k3-tiny-te-surrogate",
      override_model_config=True,
      base_emb_dim=64,
      base_mlp_dim=128,
      base_moe_mlp_dim=32,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      moe_expert_input_dim=32,
      num_experts=4,
      num_experts_per_tok=2,
      q_lora_rank=16,
      kv_lora_rank=16,
      qk_nope_head_dim=8,
      qk_rope_head_dim=8,
      v_head_dim=8,
      kimi_kda_heads=2,
      kimi_kda_head_dim=8,
      kimi_chunk_size=4,
      max_target_length=8,
      per_device_batch_size=1,
      log_config=False,
      hardware="cpu",
      base_output_directory="/tmp/k3-unit",
  )
  overrides.update(kwargs)
  return pyconfig.initialize([None, BASE], **overrides)


def native_moe(inputs, gate, wi, wo, top_k, bias):
  """Test-only dense oracle; production must always dispatch through TE."""
  logits = jnp.einsum("bsh,he->bse", inputs.astype(jnp.float32), gate.astype(jnp.float32))
  scores = jax.nn.sigmoid(logits)
  indices = jax.lax.top_k(scores + bias, top_k)[1]
  selected = jnp.take_along_axis(scores, indices, axis=-1)
  weights = selected / (selected.sum(-1, keepdims=True) + 1e-20)
  # Preserve expert GEMM BF16 rounding independently of dispatch.
  gate_up = jnp.einsum("bsh,ehm->bsem", inputs, wi)
  a, b = jnp.split(gate_up, 2, axis=-1)
  expert = jnp.einsum("bsem,emh->bseh", jax.nn.silu(a) * b, wo)
  expert = jnp.take_along_axis(expert, indices[..., None], axis=2)
  return jnp.sum(expert.astype(jnp.float32) * weights[..., None], axis=2).astype(inputs.dtype)


def oracle_adapter(self, inputs, gate, wi, wo, *args):
  del args
  output = native_moe(inputs, gate, wi, wo, self.num_experts_per_tok, self.gate.bias[...])
  return output, None, (jnp.array(1), jnp.array(4096))


@pytest.mark.parametrize("length,chunk", [(1, 4), (9, 4), (17, 8)])
def test_kda_chunk_outputs_and_all_gradients(length, chunk):
  shape = (2, length, 2, 4)
  keys = jax.random.split(jax.random.key(2), 5)
  q, k = (k3.l2norm(jax.random.normal(keys[i], shape)) for i in range(2))
  v = jax.random.normal(keys[2], shape)
  g = -5 * jax.nn.sigmoid(jax.random.normal(keys[3], shape))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1]))
  seg = jnp.ones((2, length), jnp.int32).at[0, length // 2 :].set(2).at[1, -1].set(0)

  def run(fn, *args):
    return jnp.sum(fn(*args, seg) ** 2)

  reference = lambda *a: run(k3.kda_recurrent, *a)
  candidate = lambda *a: run(lambda *a: k3.kda_chunkwise(*a, chunk_size=chunk), *a)
  expected = jax.value_and_grad(reference, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
  actual = jax.jit(jax.value_and_grad(candidate, argnums=(0, 1, 2, 3, 4)))(q, k, v, g, beta)
  for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
    np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def test_situ_and_convolution():
  x = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
  y = jnp.array([80.0, 2.0, 0.0, -3.0, -100.0])
  expected = 4 * np.tanh(np.array(x) / 4) / (1 + np.exp(-np.array(x, dtype=np.float64))) * 25 * np.tanh(np.array(y) / 25)
  np.testing.assert_allclose(k3.situ(x, y), expected, atol=1e-5)
  value = jnp.arange(12, dtype=jnp.float32).reshape(1, 6, 2)
  segments = jnp.array([[1, 1, 2, 2, 0, 0]])
  output = k3.causal_convolution(value, jnp.ones((3, 2)), segments)
  np.testing.assert_allclose(output[0, 2], jax.nn.silu(value[0, 2]))
  np.testing.assert_array_equal(output[0, 4:], 0)


def test_quantile_pooling_and_selection_bias():
  scores = jnp.array([[[0.9, 0.2, 0.1, 0.8], [0.8, 0.1, 0.4, 0.2], [0.7, 0.6, 0.2, 0.1], [0.8, 0.3, 0.2, 0.1]]])
  bias = jnp.array([0.2, 0.0, -0.1, -0.1])
  valid = jnp.array([[1, 1, 1, 0]], bool)
  hist = k3.quantile_histogram(scores, bias, 2, valid)
  pooled = sum(k3.quantile_histogram(scores[:, i : i + 1], bias, 2, valid[:, i : i + 1]) for i in range(4))
  np.testing.assert_array_equal(hist, pooled)
  np.testing.assert_array_equal(hist.sum(-1), 3)
  estimated = k3.bias_from_histogram(hist, 2)
  np.testing.assert_allclose(estimated.mean(), 0, atol=1e-7)
  cutoff = np.sort(np.array(scores + bias), axis=-1)[..., -3]
  # Histograms use nearest-rank quantiles with bin-center error <= half a bin.
  margins = np.array(scores - cutoff[..., None])[0, :3]
  exact = -np.sort(margins, axis=0)[1]
  exact -= exact.mean()
  np.testing.assert_allclose(estimated, exact, atol=4 / 256)


def test_exact_configuration_rejected():
  with pytest.raises(ValueError, match="Exact Kimi-K3 is blocked"):
    config(model_name="kimi-k3-2.8t", kimi_te_surrogate=False)


@pytest.mark.parametrize("scan,layers", [(False, 4), (True, 4), (True, 13), (True, 37)])
def test_backbone_gradients_and_balance(scan, layers):
  full = list(range(4, layers + 1, 4))
  cfg = config(
      scan_layers=scan,
      base_num_decoder_layers=layers,
      kimi_full_attn_layers=full,
      max_target_length=4,
      remat_policy="full",
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with (
      jax.set_mesh(mesh),
      logical_axis_rules(cfg.logical_axis_rules),
      mock.patch.object(moe.RoutedMoE, "_te_moe_block", oracle_adapter),
  ):
    model = k3.KimiBackbone(cfg, mesh, nnx.Rngs(3))
    x = jax.random.normal(jax.random.key(4), (1, 4, 64), dtype=jnp.bfloat16)
    seg = jnp.ones((1, 4), jnp.int32)

    def loss(m):
      return jnp.mean(m(x, seg, True).astype(jnp.float32) ** 2)

    value, grads = nnx.value_and_grad(loss)(model)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(nnx.state(grads, nnx.Param)))
    states = nnx.state(model, k3.BalanceHistogram)
    assert sum(np.asarray(h).sum() for h in jax.tree.leaves(states)) > 0
    nnx.pop(model, nnx.Intermediate)
    k3.finalize_balancing(model)
    assert all(np.asarray(h).sum() == 0 for h in jax.tree.leaves(nnx.state(model, k3.BalanceHistogram)))


def test_mla_causality_and_residual_reference():
  cfg = config()
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with jax.set_mesh(mesh), logical_axis_rules(cfg.logical_axis_rules):
    mla = k3.KimiMLA(cfg, mesh, nnx.Rngs(11))
    x = jax.random.normal(jax.random.key(5), (1, 8, 64), dtype=jnp.bfloat16)
    seg = jnp.ones((1, 8), jnp.int32)
    expected = mla(x, seg)
    changed = mla(x.at[:, 4:].set(20), seg)
    np.testing.assert_allclose(expected[:, :4], changed[:, :4], atol=1e-5)
    mix = k3.AttentionResidual(cfg, mesh, nnx.Rngs(7))
    history = jnp.stack((x, x * 2, x * 100), axis=2)
    result = mix(x * 3, history, jnp.array(2))
    values = np.stack(
        (np.array(x, dtype=np.float32), np.array(x * 2, dtype=np.float32), np.array(x * 3, dtype=np.float32)), axis=2
    )
    keys = values / np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + cfg.normalization_layer_epsilon)
    scores = np.sum(
        keys * np.array(mix.scale[...], dtype=np.float32) * np.array(mix.query[...], dtype=np.float32), axis=-1
    )
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    reference = np.sum(values * probs[..., None], axis=2)
    np.testing.assert_allclose(np.array(result, dtype=np.float32), reference, atol=2e-2, rtol=2e-2)


def test_full_transformer_optimizer_and_checkpoint_resume(tmp_path):
  """Round-trip optimizer, params, and non-optimizer QB state through Orbax."""
  import optax
  import orbax.checkpoint as ocp
  from maxtext.models.models import Transformer

  cfg = config(max_target_length=4)
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with (
      jax.set_mesh(mesh),
      logical_axis_rules(cfg.logical_axis_rules),
      mock.patch.object(moe.RoutedMoE, "_te_moe_block", oracle_adapter),
  ):
    model = Transformer(cfg, mesh, None, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    state = nnx.Dict(model=model, optimizer=optimizer)
    ids = jnp.array([[1, 2, 3, 4]], jnp.int32)
    pos = jnp.arange(4)[None]
    seg = jnp.ones_like(ids)

    @nnx.jit
    def step(state):
      def loss(m):
        logits = m(ids, pos, decoder_segment_ids=seg, enable_dropout=False)
        return jnp.mean(jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, ids[..., None], axis=-1)[..., 0])

      loss, grads = nnx.value_and_grad(loss)(state.model)
      state.optimizer.update(state.model, grads)
      k3.finalize_balancing(state.model)
      nnx.pop(state.model, nnx.Intermediate)
      return loss

    assert np.isfinite(step(state))
    graph, variables = nnx.split(state)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(tmp_path / "checkpoint", variables)
    checkpointer.wait_until_finished()
    restored = nnx.merge(graph, checkpointer.restore(tmp_path / "checkpoint", target=variables))
    expected = step(state)
    actual = step(restored)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    for a, b in zip(jax.tree.leaves(nnx.state(state)), jax.tree.leaves(nnx.state(restored))):
      if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
        a, b = jax.random.key_data(a), jax.random.key_data(b)
      np.testing.assert_array_equal(a, b)
    checkpointer.close()
