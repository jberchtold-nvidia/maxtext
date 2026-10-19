# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.
"""Four-process TE MoEBlock output/input/router/expert-gradient parity check.

Launch with MOE_PARITY=1 tests/end_to_end/gpu/kimi/run_kimi_k3.sh.
The native oracle is for validation only, never production training.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.layers import moe, initializers, quantizations
from maxtext.utils import max_utils, maxtext_utils, maxtext_utils_nnx, sharding


def native_oracle(x, gate, wi, wo, bias, top_k):
  scores = jax.nn.sigmoid(jnp.einsum("bsh,he->bse", x.astype(jnp.float32), gate.astype(jnp.float32)))
  ids = jax.lax.top_k(scores + bias, top_k)[1]
  selected = jnp.take_along_axis(scores, ids, -1)
  weights = selected / (selected.sum(-1, keepdims=True) + 1e-20)
  probabilities = jnp.sum(jax.nn.one_hot(ids, scores.shape[-1]) * weights[..., None], axis=-2)
  a, b = jnp.split(jnp.einsum("bsh,ehm->bsem", x, wi), 2, axis=-1)
  intermediate = (jax.nn.silu(a) * b) * probabilities[..., None].astype(x.dtype)
  return jnp.einsum("bsem,emh->bsh", intermediate, wo)


def main():
  import os

  root = Path(__file__).resolve().parents[4]
  cfg = pyconfig.initialize(
      [None, str(root / "src/maxtext/configs/base.yml")],
      model_name="kimi-k3-tiny-te-surrogate",
      hardware="gpu_multiprocess",
      log_config=False,
      ici_expert_parallelism=2,
      ici_fsdp_parallelism=2,
      per_device_batch_size=1,
      max_target_length=32,
      base_output_directory="/tmp/k3-parity",
      override_model_config=True,
      te_gmm_quantization=os.environ.get("TE_GMM_QUANTIZATION", "te_no_quant"),
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with (
      jax.set_mesh(mesh),
      logical_axis_rules(cfg.logical_axis_rules),
      max_utils.maybe_get_transformer_engine_context(cfg),
  ):
    max_utils.maybe_bootstrap_te_moe(cfg, mesh, {"inputs": jax.ShapeDtypeStruct((4, 32), jnp.int32)})
    quant = quantizations.TransformerEngineQuantization(
        SimpleNamespace(quantization="te_no_quant", te_comm_gemm_overlap=cfg.te_comm_gemm_overlap)
    )

    def create():
      return moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          rngs=nnx.Rngs(0),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
      )

    abstract = nnx.eval_shape(create)
    routed = maxtext_utils_nnx.create_nnx_sharded_model(abstract, create, mesh)
    bias = jnp.linspace(-0.05, 0.05, cfg.num_experts, dtype=jnp.float32)
    routed.gate.bias[...] = bias
    spec = sharding.create_sharding(
        mesh, ("activation_batch", "activation_norm_length", None), rules=cfg.logical_axis_rules
    )
    x = jax.jit(
        lambda: jax.random.normal(jax.random.key(2), (4, 32, cfg.moe_expert_input_dim), dtype=cfg.dtype),
        out_shardings=spec,
    )()
    gate, wi, wo = routed.gate.kernel[...], routed.wi[...], routed.wo[...]

    def te(x, gate, wi, wo):
      return routed._te_moe_block(x, gate, wi, wo, None, None, None)[0]

    def reference(x, gate, wi, wo):
      return native_oracle(x, gate, wi, wo, bias, cfg.num_experts_per_tok)

    def measured(fn):
      def loss(*args):
        out = fn(*args)
        return jnp.mean(out.astype(jnp.float32) ** 2), out

      return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))(x, gate, wi, wo)

    (loss, output), grad = measured(te)
    (expected_loss, expected), expected_grad = measured(reference)
    errors = {}
    for name, a, b in zip(
        ("output", "input_grad", "router_grad", "wi_grad", "wo_grad"), (output, *grad), (expected, *expected_grad)
    ):
      a, b = a.astype(jnp.float32), b.astype(jnp.float32)
      error = jnp.sqrt(jnp.sum((a - b) ** 2) / jnp.maximum(jnp.sum(b**2), 1e-20))
      errors[name] = float(error)
      if not math.isfinite(errors[name]) or errors[name] > 0.03:
        raise AssertionError(f"{name} relative RMS error {errors[name]} exceeds 0.03")
    if jax.process_index() == 0:
      print(
          json.dumps(dict(loss=float(loss), reference_loss=float(expected_loss), relative_rms_errors=errors), indent=2),
          flush=True,
      )
  jax.distributed.shutdown()


if __name__ == "__main__":
  main()
