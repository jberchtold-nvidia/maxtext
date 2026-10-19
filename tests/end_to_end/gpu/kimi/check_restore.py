# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.
"""Verify every addressable shard immediately after restoring a K3 checkpoint.

Launch RESTORE_PARITY=/path/to/step/items tests/end_to_end/gpu/kimi/run_kimi_k3.sh.
"""

import os
from pathlib import Path
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import numpy as np
from maxtext.configs import pyconfig
from maxtext.common import checkpointing, train_state_nnx
from maxtext.models.models import Transformer
from maxtext.utils import maxtext_utils, maxtext_utils_nnx, train_utils
from compare_resume import flatten, restore_numpy


def main():
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  root = Path(__file__).resolve().parents[4]
  cfg = pyconfig.initialize(
      [None, str(root / "src/maxtext/configs/base.yml")],
      model_name="kimi-k3-tiny-te-surrogate",
      hardware="gpu_multiprocess",
      log_config=False,
      ici_expert_parallelism=2,
      ici_fsdp_parallelism=2,
      per_device_batch_size=1,
      max_target_length=64,
      steps=20,
      base_output_directory="/tmp/k3-restore-check",
      mu_dtype="bfloat16",
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  path = Path(os.environ["RESTORE_PARITY"])
  with (
      jax.set_mesh(mesh),
      logical_axis_rules(cfg.logical_axis_rules),
  ):

    def create():
      model = Transformer(cfg, mesh, None, rngs=maxtext_utils_nnx.create_nnx_rngs(cfg))
      _, tx = train_utils.create_training_optimizer(cfg, model, mesh)
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    abstract = nnx.eval_shape(create)
    restored = checkpointing._load_full_state_from_path(
        path=str(path),
        checkpoint_conversion_fn=None,
        source_checkpoint_layout="orbax",
        abstract_unboxed_pre_state=nnx.state(abstract),
        checkpoint_storage_concurrent_gb=1,
        use_ocdbt=True,
        use_zarr3=True,
        maxtext_config=cfg,
    )
    actual = train_state_nnx.to_checkpoint_dict(restored)
    expected = dict(flatten(restore_numpy(path)))

    # Preserve distributed arrays here; flatten() would convert global arrays to numpy.
    def leaves(tree, prefix=()):
      if isinstance(tree, dict):
        for key, value in tree.items():
          yield from leaves(value, prefix + (str(key),))
      elif isinstance(tree, (tuple, list)):
        for key, value in enumerate(tree):
          yield from leaves(value, prefix + (str(key),))
      elif tree is not None:
        yield "/".join(prefix), tree

    actual = dict(leaves(actual))
    assert actual.keys() == expected.keys()
    count = 0
    for name, value in actual.items():
      if isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
      if isinstance(value, jax.Array):
        for shard in value.addressable_shards:
          np.testing.assert_array_equal(np.asarray(shard.data), expected[name][shard.index], err_msg=name)
      else:
        np.testing.assert_array_equal(value, expected[name], err_msg=name)
      count += 1
    print(f"rank={jax.process_index()} restored {count} arrays exactly, including optimizer and QB state", flush=True)
  jax.distributed.shutdown()


if __name__ == "__main__":
  main()
