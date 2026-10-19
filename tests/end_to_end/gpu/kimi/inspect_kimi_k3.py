# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.
"""Abstract full-size parameter/AdamW accounting; allocates no weights."""

import json
import math
from pathlib import Path
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
from maxtext.configs import pyconfig
from maxtext.models.models import Transformer
from maxtext.utils import maxtext_utils, train_utils, sharding


def inspect():
  root = Path(__file__).resolve().parents[4]
  cfg = pyconfig.initialize(
      [None, str(root / "src/maxtext/configs/base.yml")],
      model_name="kimi-k3-2.8t-te-surrogate",
      hardware="gpu",
      log_config=False,
      ici_expert_parallelism=2,
      ici_fsdp_parallelism=2,
      per_device_batch_size=1,
      max_target_length=16,
      base_output_directory="/tmp/kimi-k3-abstract",
      mu_dtype="bfloat16",
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)

  def create():
    model = Transformer(cfg, mesh, None, rngs=nnx.Rngs(0))
    _, tx = train_utils.create_training_optimizer(cfg, model, mesh)
    return model, nnx.Optimizer(model, tx, wrt=nnx.Param)

  with jax.set_mesh(mesh), logical_axis_rules(cfg.logical_axis_rules):
    model, optimizer = nnx.eval_shape(create)
  params = nnx.state(model, nnx.Param)
  parameter_count = sum(math.prod(x.shape) for x in jax.tree.leaves(params))
  parameter_bytes = sum(math.prod(x.shape) * x.dtype.itemsize for x in jax.tree.leaves(params))
  optimizer_bytes = sum(
      math.prod(x.shape) * x.dtype.itemsize for x in jax.tree.leaves(nnx.state(optimizer, nnx.OptState))
  )
  checked = 0
  for _, var in jax.tree_util.tree_leaves_with_path(params, is_leaf=lambda x: isinstance(x, nnx.Variable)):
    axes = var.get_metadata().get("out_sharding")
    if axes is None:
      continue
    spec = sharding.logical_to_spec(axes, cfg.logical_axis_rules)
    for dimension, partition in zip(var.shape, spec):
      axes = (partition,) if isinstance(partition, str) else (partition or ())
      divisor = math.prod(mesh.shape[a] for a in axes)
      if dimension % divisor:
        raise ValueError(f"Invalid shape/sharding: {var.shape}, {spec}")
    checked += 1
  exact_router_delta = (cfg.emb_dim - cfg.moe_expert_input_dim) * cfg.num_experts * (cfg.num_decoder_layers - 1)
  print(
      json.dumps(
          dict(
              model="kimi-k3-2.8t-te-surrogate",
              abstract_only=True,
              surrogate_parameters=parameter_count,
              exact_router_parameter_delta=exact_router_delta,
              implied_exact_text_parameters=parameter_count + exact_router_delta,
              parameter_bytes=parameter_bytes,
              optimizer_bytes=optimizer_bytes,
              ideal_four_gpu_state_bytes_per_gpu=(parameter_bytes + optimizer_bytes) / 4,
              validated_parameter_shardings=checked,
              limitations="State-only ideal lower bound: excludes activations, gradients, dispatch, gathers, workspaces, replication and padding.",
          ),
          indent=2,
      )
  )


if __name__ == "__main__":
  inspect()
