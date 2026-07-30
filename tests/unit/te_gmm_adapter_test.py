# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MaxText's TransformerEngine grouped GEMM adapter."""

import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import pytest


_QUANTIZATIONS_PATH = Path(__file__).parents[2] / "src" / "maxtext" / "layers" / "quantizations.py"
_SPEC = importlib.util.spec_from_file_location("_local_maxtext_quantizations", _QUANTIZATIONS_PATH)
quantizations = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quantizations)


def test_te_gmm_adapter_local_ep_axis(monkeypatch):
  """TE GMM receives local EP-shaped inputs without changing MaxText's flat contract."""
  seen_input_shapes = []

  def fake_make_grouped_dense_cls(quantization_recipe, quantization_checkpoint_name=None):
    assert quantization_recipe is not None
    assert quantization_checkpoint_name == "quantization"

    def fake_grouped_dense(inputs, kernel, group_sizes):
      del group_sizes
      seen_input_shapes.append(inputs.shape)
      return jnp.zeros((*inputs.shape[:-1], kernel.shape[-1]), dtype=inputs.dtype)

    return fake_grouped_dense

  fake_te = ModuleType("transformer_engine")
  fake_te_jax = ModuleType("transformer_engine.jax")
  fake_te_flax = ModuleType("transformer_engine.jax.flax")
  fake_te_flax.make_grouped_dense_cls = fake_make_grouped_dense_cls
  fake_te_jax.flax = fake_te_flax
  fake_te.jax = fake_te_jax
  monkeypatch.setitem(sys.modules, "transformer_engine", fake_te)
  monkeypatch.setitem(sys.modules, "transformer_engine.jax", fake_te_jax)
  monkeypatch.setitem(sys.modules, "transformer_engine.jax.flax", fake_te_flax)
  monkeypatch.setattr(
      quantizations.TransformerEngineQuantization,
      "_get_recipe",
      staticmethod(lambda recipe_name: object()),
  )

  quant = quantizations.TransformerEngineQuantization(SimpleNamespace(quantization="te_mxfp8"))
  kernel = jnp.ones((2, 16, 32), dtype=jnp.bfloat16)
  group_sizes = jnp.array([64, 64], dtype=jnp.int32)

  flat_inputs = jnp.ones((128, 16), dtype=jnp.bfloat16)
  flat_output = quant.gmm(flat_inputs, kernel, None, group_sizes, None, "te_mxfp8")
  assert seen_input_shapes[-1] == (1, 128, 16)
  assert flat_output.shape == (128, 32)

  ep_local_inputs = jnp.ones((1, 128, 16), dtype=jnp.bfloat16)
  ep_local_output = quant.gmm(ep_local_inputs, kernel, None, group_sizes, None, "te_mxfp8")
  assert seen_input_shapes[-1] == (1, 128, 16)
  assert ep_local_output.shape == (1, 128, 32)


def test_moe_block_quantizer_set_keeps_distinct_token_and_expert_group_counts():
  quant = quantizations.TransformerEngineQuantization(SimpleNamespace(quantization="te_mxfp8"))
  quantizer_set = quant.get_moe_block_quantizer_set(
      "te_mxfp8", n_token_groups=64, n_expert_groups=32
  )

  assert len(quantizer_set.x.quantizers) == 64
  assert len(quantizer_set.dgrad.quantizers) == 64
  assert len(quantizer_set.kernel.quantizers) == 32


def test_moe_block_quantizer_sets_are_independent():
  quant = quantizations.TransformerEngineQuantization(SimpleNamespace(quantization="te_mxfp8"))
  fc1_quantizer_set, fc2_quantizer_set = quant.get_moe_block_quantizer_sets(
      "te_mxfp8", n_token_groups=64, n_expert_groups=32
  )

  assert fc1_quantizer_set is not fc2_quantizer_set
  for quantizer_set in (fc1_quantizer_set, fc2_quantizer_set):
    assert len(quantizer_set.x.quantizers) == 64
    assert len(quantizer_set.dgrad.quantizers) == 64
    assert len(quantizer_set.kernel.quantizers) == 32


def test_moe_block_quantizer_set_rejects_invalid_group_counts():
  quant = quantizations.TransformerEngineQuantization(SimpleNamespace(quantization="te_mxfp8"))
  with pytest.raises(ValueError, match="group counts must be positive"):
    quant.get_moe_block_quantizer_set("te_mxfp8", n_token_groups=0, n_expert_groups=32)


def test_moe_block_mxfp8_shape_validation_uses_grouped_gemm_k_dim():
  quant = quantizations.TransformerEngineQuantization(SimpleNamespace(quantization="te_mxfp8"))

  quant.validate_moe_block_quantization_shapes("te_mxfp8", hidden_dim=1792, intermediate_dim=2048, fsdp_size=2)
  quant.validate_moe_block_quantization_shapes("te_no_quant", hidden_dim=1793, intermediate_dim=2049, fsdp_size=2)

  with pytest.raises(ValueError, match="local hidden/K.*128-aligned"):
    quant.validate_moe_block_quantization_shapes("te_mxfp8", hidden_dim=1800, intermediate_dim=2048, fsdp_size=2)
  with pytest.raises(ValueError, match="intermediate.*128-aligned"):
    quant.validate_moe_block_quantization_shapes("te_mxfp8", hidden_dim=1792, intermediate_dim=2050, fsdp_size=2)
  with pytest.raises(ValueError, match="must be divisible by fsdp_size"):
    quant.validate_moe_block_quantization_shapes("te_mxfp8", hidden_dim=1793, intermediate_dim=2048, fsdp_size=2)

  # FSDP on the expert dimension leaves the grouped-GEMM K dimension whole.
  quant.validate_moe_block_quantization_shapes(
      "te_mxfp8", hidden_dim=7168, intermediate_dim=2048, fsdp_size=16, shard_exp_on_fsdp=True
  )
