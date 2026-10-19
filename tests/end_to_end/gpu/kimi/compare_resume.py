# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.
"""Compare saved K3 training continuations and optimizer/nonparameter state.

Example: JAX_PLATFORMS=cpu python tests/end_to_end/gpu/kimi/compare_resume.py \
  /tmp/k3-checkpoint-baseline /tmp/k3-checkpoint-resume
"""

import argparse
import json
from pathlib import Path
import jax
import numpy as np
import orbax.checkpoint as ocp


def restore_numpy(path):
  with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(path)
    tree = metadata.item_metadata if hasattr(metadata, "item_metadata") else metadata
    restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), tree)
    return checkpointer.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))


def flatten(tree, prefix=()):
  if isinstance(tree, dict):
    for key, value in tree.items():
      yield from flatten(value, prefix + (str(key),))
  elif isinstance(tree, (list, tuple)):
    for key, value in enumerate(tree):
      yield from flatten(value, prefix + (str(key),))
  elif tree is not None:
    yield "/".join(prefix), np.asarray(tree)


def compare(baseline, resumed):
  def metrics(root):
    path = root / "metrics_rank_0.jsonl"
    if not path.exists():
      path = root / "metrics.jsonl"  # Older launcher wrote all ranks to one file.
    return {int(v["step"]): v for v in map(json.loads, path.read_text().splitlines())}

  expected, actual = metrics(baseline), metrics(resumed)
  assert actual and max(actual) == max(expected), "Resume did not reach the baseline final step"
  keys = ("learning/loss", "learning/raw_grad_norm", "learning/param_norm", "diag/max_abs_grad")
  for step, record in actual.items():
    for key in keys:
      np.testing.assert_allclose(record[key], expected[step][key], rtol=2e-2, atol=2e-2, err_msg=f"{step}: {key}")
    assert record["learning/te_moe_capacity_overflow"] == 0
  final = max(expected)

  def checkpoint(root, step):
    return root / "logdir/checkpoints" / str(step) / "items"

  a = dict(flatten(restore_numpy(checkpoint(baseline, final))))
  b = dict(flatten(restore_numpy(checkpoint(resumed, final))))
  assert a.keys() == b.keys()
  exact_arrays = 0
  max_state_rms_error = 0.0
  max_parameter_rms_error = 0.0
  for name in a:
    if np.array_equal(a[name], b[name]):
      exact_arrays += 1
      continue
    if not np.issubdtype(a[name].dtype, np.inexact) and str(a[name].dtype) != "bfloat16":
      np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    left, right = a[name].astype(np.float64), b[name].astype(np.float64)
    error = float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))
    max_state_rms_error = max(max_state_rms_error, error)
    # Checkpoint restoration itself is checked exactly by check_restore.py.
    # Subsequent TE BF16 training need not be bitwise deterministic: sparse
    # router decisions can amplify rounding in small optimizer moments. Report
    # all-state differences, but gate continuation on model parameters/metrics.
    if name.startswith("params/"):
      max_parameter_rms_error = max(max_parameter_rms_error, error)
      assert error < 2e-2, f"{name}: BF16 parameter relative RMS error {error} exceeds 0.02"
  max_loss_difference = max(abs(actual[i]["learning/loss"] - expected[i]["learning/loss"]) for i in actual)
  initial = dict(flatten(restore_numpy(checkpoint(baseline, min(expected)))))
  changed = {}
  for name, value in a.items():
    if (
        name.startswith("params/")
        and "/routed/" in name
        and (name.endswith("/gate/kernel") or name.endswith("/wi") or name.endswith("/wo"))
    ):
      changed[name] = int(np.count_nonzero(value != initial[name]))
      assert changed[name] > 0, f"No router/expert parameter update: {name}"
  assert changed, "No routed parameters found in checkpoint"
  print(
      json.dumps(
          dict(
              compared_steps=sorted(actual),
              checkpoint_arrays_compared=len(a),
              checkpoint_arrays_identical=exact_arrays,
              max_state_relative_rms_error=max_state_rms_error,
              max_parameter_relative_rms_error=max_parameter_rms_error,
              max_loss_difference=max_loss_difference,
              changed_routed_parameters=changed,
          ),
          indent=2,
      )
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("baseline", type=Path)
  parser.add_argument("resumed", type=Path)
  args = parser.parse_args()
  compare(args.baseline, args.resumed)
