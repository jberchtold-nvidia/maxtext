# Copyright 2025-2026 Google LLC
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

"""Functions for gradient accumulation (GA)"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from maxtext.common.common_types import ShardMode
from maxtext.utils.sharding import maybe_shard_with_name

_QW_CACHE_KEY = "quantized_kernel_cache"


def gradient_accumulation_loss_and_grad(
    _loss_fn,
    config,
    model,
    params,
    params_shardings,
    data,
    dropout_rng,
    extra_dpo_args,
):
  """
  Calculates gradients using gradient accumulation.

  When quantized-weight caching is active, the function runs a two-phase
  approach:
  - Phase 1: first micro-step with params WITHOUT the cache collection.
    All CachingTE*Wrapper modules see ``has_variable(...) == False`` →
    quantize fresh (JIT trace A).
  - Phase 2: remaining K-1 micro-steps with params WITH the populated
    cache collection.  All modules see ``has_variable(...) == True`` →
    load from cache (JIT trace B via lax.scan).

  The different pytree structures naturally produce separate JIT traces.
  """

  def _maybe_shard_with_name(inputs, sharding_names):
    return maybe_shard_with_name(inputs, sharding_names, config.shard_mode, debug_sharding=config.debug_sharding)

  # For more efficient DP/ZeRO-1 + GA
  if config.shard_mode == ShardMode.EXPLICIT and config.ici_data_parallelism > 1:
    ga_params_shardings = jax.tree.map(update_sharding_for_reduced, params_shardings)
    grad_shardings = jax.tree.map(update_sharding_for_unreduced, params_shardings)
  else:
    ga_params_shardings = grad_shardings = params_shardings

  if config.shard_optimizer_over_data:
    def convert_to_bf16(param):
      if param.dtype == jnp.float32:
        return param.astype(jnp.bfloat16)
      return param
    ga_params = jax.tree_util.tree_map(convert_to_bf16, params)
  else:
    ga_params = params

  # Detect and strip cache collection BEFORE sharding (no sharding spec for it).
  _has_qw_cache = _QW_CACHE_KEY in ga_params
  if not _has_qw_cache:
    _has_qw_cache = (
        hasattr(config, "quantization")
        and config.quantization is not None
        and config.quantization.startswith("te_")
        and config.gradient_accumulation_steps > 1
    )
  if _QW_CACHE_KEY in ga_params:
    ga_cache = ga_params[_QW_CACHE_KEY]
    ga_params_no_cache = {k: v for k, v in ga_params.items() if k != _QW_CACHE_KEY}
  else:
    ga_cache = None
    ga_params_no_cache = ga_params

  ga_params_no_cache = jax.tree.map(_maybe_shard_with_name, ga_params_no_cache, ga_params_shardings)
  grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)

  def reshape_to_microbatch_accumulations(batch_arr):
    num_microbatches = config.gradient_accumulation_steps
    microbatch_shape = (batch_arr.shape[0] // num_microbatches, num_microbatches) + batch_arr.shape[1:]
    reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
    return jnp.swapaxes(reshaped_batch_arr, 0, 1)

  data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)

  # -----------------------------------------------------------------------
  # Two-phase GA with quantized-weight caching
  # -----------------------------------------------------------------------
  if _has_qw_cache and config.gradient_accumulation_steps > 1:
    # PHASE 1: first micro-step — params WITHOUT cache → fresh quantize
    # (JIT trace A: has_variable returns False for all caching modules)
    first_data = jax.tree.map(lambda d: d[0], data)
    (_, first_aux), first_grads = grad_func(
        model, config, first_data, dropout_rng, ga_params_no_cache,
        *extra_dpo_args, is_train=True,
    )

    # Extract populated cache from the mutable output
    first_intermediates = first_aux.get("intermediate_outputs", {})
    if isinstance(first_intermediates, tuple) and len(first_intermediates) > 1:
      first_intermediates = first_intermediates[1]
    populated_cache = first_intermediates.get(_QW_CACHE_KEY, ga_cache)

    # PHASE 2: remaining K-1 micro-steps — params WITH cache → use cached
    # (JIT trace B: has_variable returns True for all caching modules)
    ga_params_with_cache = {**ga_params_no_cache, _QW_CACHE_KEY: populated_cache}

    remaining_data = jax.tree.map(lambda d: d[1:], data)

    # Pad first_grads with zero entries for the cache collection so
    # the pytree structure matches ga_params_with_cache.
    cache_zero_grads = jax.tree_util.tree_map(jnp.zeros_like, populated_cache)
    first_grads_padded = {**first_grads, _QW_CACHE_KEY: cache_zero_grads}

    init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params_with_cache)
    init_grad_and_loss = {
        "loss": first_aux["total_loss"],
        "grad": jax.tree_util.tree_map(lambda fg, ig: fg + ig, first_grads_padded, init_grad),
        "total_weights": first_aux["total_weights"],
        "moe_lb_loss": first_aux["moe_lb_loss"],
        "indexer_loss": first_aux["indexer_loss"],
        "mtp_loss": first_aux["mtp_loss"],
        "ga_params": ga_params_with_cache,
    }

    def accumulate_cached(acc, micro_data):
      p = acc["ga_params"]
      # stop_gradient on cache arrays so optimizer doesn't update them.
      p = {
          k: (jax.lax.stop_gradient(v) if k == _QW_CACHE_KEY else v)
          for k, v in p.items()
      }
      (_, aux), cur_grads = grad_func(
          model, config, micro_data, dropout_rng, p,
          *extra_dpo_args, is_train=True,
      )
      acc["loss"] += aux["total_loss"]
      acc["moe_lb_loss"] += aux["moe_lb_loss"]
      acc["indexer_loss"] += aux["indexer_loss"]
      acc["mtp_loss"] += aux["mtp_loss"]
      acc["grad"] = jax.tree_util.tree_map(lambda x, y: x + y, cur_grads, acc["grad"])
      acc["total_weights"] += aux["total_weights"]
      return acc, aux

    grad_and_loss, aux = jax.lax.scan(
        accumulate_cached, init_grad_and_loss, remaining_data,
        length=config.gradient_accumulation_steps - 1,
    )

    aux = jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None], rest], axis=0),
        first_aux, aux,
    )

  else:
    # ----- Standard GA (no caching) ------------------------------------
    def accumulate_gradient(acc_grad_and_loss, micro_data):
      ga_params = acc_grad_and_loss["ga_params"]
      (_, aux), cur_batch_gradient = grad_func(
          model, config, micro_data, dropout_rng, ga_params,
          *extra_dpo_args, is_train=True,
      )
      acc_grad_and_loss["loss"] += aux["total_loss"]
      acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
      acc_grad_and_loss["indexer_loss"] += aux["indexer_loss"]
      acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"]
      )
      acc_grad_and_loss["total_weights"] += aux["total_weights"]
      return acc_grad_and_loss, aux

    init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params_no_cache)
    init_grad = jax.tree.map(_maybe_shard_with_name, init_grad, grad_shardings)
    init_grad_and_loss = {
        "loss": 0.0,
        "grad": init_grad,
        "total_weights": 0,
        "moe_lb_loss": 0.0,
        "indexer_loss": 0.0,
        "mtp_loss": 0.0,
        "ga_params": ga_params_no_cache,
    }

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data,
        length=config.gradient_accumulation_steps,
    )

  # --- Post-accumulation: normalize and return ----------------------------
  loss = (
      grad_and_loss["loss"] / grad_and_loss["total_weights"]
      + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
      + grad_and_loss["indexer_loss"] / config.gradient_accumulation_steps
      + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
  )
  raw_grads = grad_and_loss["grad"]
  # Strip cache entries from gradients (zero due to stop_gradient).
  if _QW_CACHE_KEY in raw_grads:
    raw_grads = {k: v for k, v in raw_grads.items() if k != _QW_CACHE_KEY}
  raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, params_shardings)
  raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], raw_grads)
  aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)

  return loss, aux, raw_grads


def update_sharding_for_reduced(sharding: NamedSharding) -> NamedSharding:
  return sharding.update(spec=sharding.spec.update(reduced={"data"}))


def update_sharding_for_unreduced(sharding: NamedSharding) -> NamedSharding:
  return sharding.update(spec=sharding.spec.update(unreduced={"data"}))
