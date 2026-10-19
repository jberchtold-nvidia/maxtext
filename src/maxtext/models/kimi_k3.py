# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Native K3 text backbone with an explicitly approximate TE routed MoE.

Reference: moonshotai/Kimi-K3@c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721.
See docs/guides/kimi_k3_te_compatibility.md before changing surrogate semantics.
"""

import math
from types import SimpleNamespace
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.layers import initializers, linears, moe, quantizations
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_logging


def situ(gate, up, beta=4.0, linear_beta=25.0):
  """K3 SiTU-GLU, including the independently soft-capped up branch."""
  a, b = gate.astype(jnp.float32), up.astype(jnp.float32)
  return (beta * jnp.tanh(a / beta) * jax.nn.sigmoid(a) * linear_beta * jnp.tanh(b / linear_beta)).astype(gate.dtype)


def l2norm(x):
  x = x.astype(jnp.float32)
  return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def kda_recurrent(q, k, v, log_decay, beta, segment_ids):
  """Independent time-recurrent oracle; state uses [key, value] ordering."""
  state = jnp.zeros(q.shape[:1] + q.shape[2:] + (v.shape[-1],), jnp.float32)

  def step(carry, xs):
    state, last_segment = carry
    qt, kt, vt, gt, bt, seg = xs
    active = seg != 0
    state = jnp.where((active & (seg == last_segment))[:, None, None, None], state, 0.0)
    state = state * jnp.exp(gt)[..., None]
    delta = bt[..., None] * (vt - jnp.einsum("bhk,bhkv->bhv", kt, state))
    state = state + kt[..., None] * delta[..., None, :]
    output = jnp.einsum("bhk,bhkv->bhv", qt, state) / math.sqrt(q.shape[-1])
    return (state, seg), jnp.where(active[:, None, None], output, 0.0)

  xs = tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, v, log_decay, beta, segment_ids))
  _, output = jax.lax.scan(step, (state, jnp.zeros(q.shape[0], jnp.int32)), xs)
  return jnp.swapaxes(output, 0, 1).astype(v.dtype)


def kda_chunkwise(q, k, v, log_decay, beta, segment_ids, chunk_size=64):
  """Chunk-parallel delta rule using triangular solves and FP32 state.

  Segments never interact, including inside a chunk. Chunk rematerialization
  bounds backward storage without retaining one key/value matrix per token.
  """
  b, length, heads, dim = q.shape
  pad = (-length) % chunk_size

  def chunks(x):
    x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
    return jnp.swapaxes(x.reshape(b, -1, chunk_size, *x.shape[2:]), 0, 1)

  xs = tuple(chunks(x) for x in (q, k, v, log_decay, beta, segment_ids))
  initial = (jnp.zeros((b, heads, dim, v.shape[-1]), jnp.float32), jnp.zeros(b, jnp.int32))

  def chunk(carry, xs):
    state, last_segment = carry
    q, k, v, g, beta, seg = xs
    q, k, v, g, beta = (x.astype(jnp.float32) for x in (q, k, v, g, beta))
    q, k, v, g = (jnp.transpose(x, (0, 2, 1, 3)) for x in (q, k, v, g))
    beta = jnp.swapaxes(beta, 1, 2)
    cumulative = jnp.cumsum(g, axis=2)
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] != 0)
    causal = jnp.arange(chunk_size)[:, None] >= jnp.arange(chunk_size)[None, :]
    mask = same[:, None] & causal[None, None]
    # Future/padded entries are masked before exp to avoid overflow in backward.
    difference = cumulative[:, :, :, None, :] - cumulative[:, :, None, :, :]
    decay = jnp.exp(jnp.where(mask[..., None], difference, 0.0)) * mask[..., None]
    kk = jnp.einsum("bhik,bhijk,bhjk->bhij", k, decay, k)
    qk = jnp.einsum("bhik,bhijk,bhjk->bhij", q, decay, k)
    from_state = (seg == last_segment[:, None]) & (seg != 0)
    prefix_decay = jnp.exp(cumulative) * from_state[:, None, :, None]
    incoming = jnp.einsum("bhik,bhik,bhkv->bhiv", k, prefix_decay, state)
    system = jnp.eye(chunk_size) + jnp.tril(kk * beta[..., None], -1)
    rhs = beta[..., None] * (v - incoming)
    updates = jax.lax.linalg.triangular_solve(system, rhs, left_side=True, lower=True, unit_diagonal=True)
    output = (
        jnp.einsum("bhik,bhik,bhkv->bhiv", q, prefix_decay, state) + jnp.einsum("bhij,bhjv->bhiv", qk, updates)
    ) / math.sqrt(dim)
    output = jnp.where((seg != 0)[:, None, :, None], output, 0.0)
    tail_decay = decay[:, :, -1]
    next_state = jnp.einsum("bhjk,bhjk,bhjv->bhkv", tail_decay, k, updates)
    next_state += state * prefix_decay[:, :, -1, :, None]
    return (next_state, seg[:, -1]), jnp.transpose(output, (0, 2, 1, 3)).astype(xs[2].dtype)

  _, output = jax.lax.scan(jax.checkpoint(chunk), initial, xs)
  return jnp.swapaxes(output, 0, 1).reshape(b, -1, heads, v.shape[-1])[:, :length]


def causal_convolution(x, kernel, segment_ids):
  """Depthwise causal convolution; kernel follows PyTorch's oldest-first order."""
  length = x.shape[1]
  result = jnp.zeros_like(x)
  for offset in range(kernel.shape[0]):
    lag = kernel.shape[0] - 1 - offset
    if lag >= length:
      continue
    source = jnp.pad(x[:, : length - lag], ((0, 0), (lag, 0), (0, 0)))
    segments = jnp.pad(segment_ids[:, : length - lag], ((0, 0), (lag, 0)))
    valid = (segments == segment_ids) & (segment_ids != 0)
    result += source * kernel[offset].astype(x.dtype) * valid[..., None]
  return jax.nn.silu(result)


def quantile_histogram(scores, bias, top_k, valid, bins=256):
  """Detached global-array histogram, additive across microbatches.

  Fixed [-2,2] bins allow exact pooling across microbatches without rebinning.
  For sigmoid scores, QB's centered biases span at most one, so the margins
  stay within this range. JAX global reductions pool token shards, not replicas.
  """
  scores, bias = jax.lax.stop_gradient((scores, bias))
  cutoff = jax.lax.top_k(scores + bias, top_k + 1)[0][..., -1]
  margins = scores - cutoff[..., None]
  indices = jnp.clip(jnp.floor((margins + 2.0) * (bins / 4.0)).astype(jnp.int32), 0, bins - 1)
  experts = scores.shape[-1]
  indices = indices.reshape(-1, experts) + jnp.arange(experts)[None] * bins
  weights = jnp.broadcast_to(valid.reshape(-1, 1), indices.shape).astype(jnp.int32)
  return jnp.bincount(indices.ravel(), weights=weights.ravel(), length=experts * bins).reshape(experts, bins)


def bias_from_histogram(histogram, top_k):
  """Approximate the (1-k/E) quantile using bin centers; keep empty state unchanged at caller."""
  bins = histogram.shape[-1]
  count = histogram.sum(-1)
  target = jnp.maximum(jnp.ceil(count * (1.0 - top_k / histogram.shape[-2])).astype(jnp.int32), 1)
  index = jnp.argmax(jnp.cumsum(histogram, axis=-1) >= target[..., None], axis=-1)
  bias = -(-2.0 + (index.astype(jnp.float32) + 0.5) * (4.0 / bins))
  return bias - jnp.mean(bias, axis=-1, keepdims=True)


class BalanceHistogram(nnx.Variable):
  """Non-optimizer state accumulated through GA and saved in checkpoints."""


def finalize_balancing(model, overflow=False):
  """Apply pooled QB exactly once after an optimizer step, not during forward/remat."""
  values, updates = [], []
  for _, node in nnx.iter_graph(model):
    if isinstance(node, KimiMoE):
      hist = node.histogram[...]
      new_bias = bias_from_histogram(hist, node.config.num_experts_per_tok)
      has_tokens = jnp.any(hist > 0, axis=(-2, -1))
      update = has_tokens & ~jnp.asarray(overflow)
      old_bias = node.routed.gate.bias[...]
      node.routed.gate.bias[...] = jnp.where(update[..., None], new_bias, old_bias)
      values.append(node.routed.gate.bias[...])
      updates.append(node.routed.gate.bias[...] - old_bias)
      node.histogram[...] = jnp.zeros_like(hist)
  return values, updates


class NativeLinear(nnx.Module):
  """Native MaxText dense primitive with FSDP logical parameter axes."""

  def __init__(self, config, mesh, rngs, input_dim, output_dim):
    self.proj = linears.DenseGeneral(
        in_features_shape=input_dim,
        out_features_shape=output_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=initializers.nd_dense_init(0.02, "fan_in", "normal"),
        kernel_axes=("embed", "mlp"),
        mesh=mesh,
        rngs=rngs,
        shard_mode=config.shard_mode,
        quant=None,
    )

  def __call__(self, x):
    return self.proj(x)


def norm(config, rngs, features):
  return RMSNorm(
      num_features=features,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      kernel_axes=("norm",),
      epsilon=config.normalization_layer_epsilon,
      rngs=rngs,
  )


class NativeFFN(nnx.Module):

  def __init__(self, config, mesh, rngs, hidden, intermediate):
    self.gate = NativeLinear(config, mesh, rngs, hidden, intermediate)
    self.up = NativeLinear(config, mesh, rngs, hidden, intermediate)
    self.down = NativeLinear(config, mesh, rngs, intermediate, hidden)
    self.beta, self.linear_beta = config.kimi_situ_beta, config.kimi_situ_linear_beta

  def __call__(self, x):
    return self.down(situ(self.gate(x), self.up(x), self.beta, self.linear_beta))


class KimiMoE(nnx.Module):

  def __init__(self, config, mesh, rngs):
    self.config = config
    latent = config.moe_expert_input_dim
    self.down = NativeLinear(config, mesh, rngs, config.emb_dim, latent)
    self.up = NativeLinear(config, mesh, rngs, latent, config.emb_dim)
    self.latent_norm = norm(config, rngs, latent)
    self.shared = NativeFFN(config, mesh, rngs, config.emb_dim, config.shared_experts * config.moe_mlp_dim)
    # K3-TE-001: correct K3 routes on the original hidden state. Existing TE
    # requires the same input for routing and experts; route on latent input.
    # K3-TE-002: RoutedMoE's SwiGLU substitutes for two-branch SiTU-GLU.
    # Both deviations and restoration tests: docs/guides/kimi_k3_te_compatibility.md.
    te_quant = quantizations.TransformerEngineQuantization(
        SimpleNamespace(quantization="te_no_quant", te_comm_gemm_overlap=config.te_comm_gemm_overlap)
    )
    self.routed = moe.RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        intermediate_dim=config.moe_mlp_dim,
        mesh=mesh,
        rngs=rngs,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=te_quant,
    )
    self.histogram = BalanceHistogram(
        jnp.zeros((config.num_experts, config.kimi_qb_bins), jnp.int32), out_sharding=(None, None)
    )

  def __call__(self, x, segments, collect_balance):
    latent = self.down(x)
    if collect_balance:
      # Detached duplicate logits are statistics only; TE still owns dispatch.
      scores = jax.nn.sigmoid(
          jnp.einsum("bsh,he->bse", latent.astype(jnp.float32), self.routed.gate.kernel[...].astype(jnp.float32))
      )
      hist = quantile_histogram(
          scores, self.routed.gate.bias[...], self.config.num_experts_per_tok, segments != 0, self.config.kimi_qb_bins
      )
      self.histogram[...] += hist
    routed, _, recv = self.routed(latent)
    tokens, capacity = recv
    self.sow(nnx.Intermediate, "te_moe_capacity_overflow", jnp.atleast_1d(jnp.any(tokens > capacity)))
    self.sow(nnx.Intermediate, "te_moe_total_recv_tokens", jnp.atleast_1d(jnp.max(tokens)))
    self.sow(nnx.Intermediate, "te_moe_recv_capacity_per_rank", jnp.atleast_1d(capacity))
    # Match reference order: combine -> latent RMSNorm -> up -> add shared.
    return self.up(self.latent_norm(routed)) + self.shared(x)


class KimiKDA(nnx.Module):

  def __init__(self, config, mesh, rngs):
    self.config = config
    self.heads, self.dim = config.kimi_kda_heads, config.kimi_kda_head_dim
    width = self.heads * self.dim
    for name in ("q", "k", "v", "gate"):
      setattr(self, name, NativeLinear(config, mesh, rngs, config.emb_dim, width))
    self.f_a = NativeLinear(config, mesh, rngs, config.emb_dim, self.dim)
    self.f_b = NativeLinear(config, mesh, rngs, self.dim, width)
    self.beta = NativeLinear(config, mesh, rngs, config.emb_dim, self.heads)
    self.out = NativeLinear(config, mesh, rngs, width, config.emb_dim)
    self.out_norm = norm(config, rngs, self.dim)
    for name in ("q_conv", "k_conv", "v_conv"):
      shape = (config.kimi_conv_size, width)
      setattr(
          self,
          name,
          nnx.Param(
              jax.random.normal(rngs.params(), shape, dtype=config.weight_dtype) / math.sqrt(config.kimi_conv_size),
              out_sharding=(None, None),
          ),
      )
    self.a_log = nnx.Param(
        jnp.log(jax.random.uniform(rngs.params(), (self.heads,), minval=1, maxval=16)), out_sharding=(None,)
    )
    self.dt_bias = nnx.Param(jnp.zeros((self.heads, self.dim), jnp.float32), out_sharding=(None, None))

  def __call__(self, x, segments):
    shape = (*x.shape[:2], self.heads, self.dim)
    q, k, v = (
        causal_convolution(getattr(self, name)(x), getattr(self, name + "_conv")[...], segments).reshape(shape)
        for name in ("q", "k", "v")
    )
    q, k = l2norm(q), l2norm(k)
    g = self.f_b(self.f_a(x)).reshape(shape).astype(jnp.float32)
    # Safe KDA decay from the reference FLA gate: lower_bound*sigmoid(A*(g+dt_bias)).
    log_decay = self.config.kimi_gate_lower_bound * jax.nn.sigmoid(
        jnp.exp(self.a_log[...])[None, None, :, None] * (g + self.dt_bias[...])
    )
    beta = jax.nn.sigmoid(self.beta(x).astype(jnp.float32))
    output = kda_chunkwise(q, k, v, log_decay, beta, segments, self.config.kimi_chunk_size)
    output = self.out_norm(output) * jax.nn.sigmoid(self.gate(x).reshape(shape))
    return self.out(output.reshape(*x.shape[:2], -1))


class KimiMLA(nnx.Module):

  def __init__(self, config, mesh, rngs):
    self.config = config
    h, d = config.num_query_heads, config.qk_nope_head_dim + config.qk_rope_head_dim
    self.q_a = NativeLinear(config, mesh, rngs, config.emb_dim, config.q_lora_rank)
    self.q_b = NativeLinear(config, mesh, rngs, config.q_lora_rank, h * d)
    self.kv_a = NativeLinear(config, mesh, rngs, config.emb_dim, config.kv_lora_rank + config.qk_rope_head_dim)
    # Reference's MLA latent RMSNorm uses its module default 1e-6, not global 1e-5.
    norm_config = SimpleNamespace(dtype=config.dtype, weight_dtype=config.weight_dtype, normalization_layer_epsilon=1e-6)
    self.q_norm = norm(norm_config, rngs, config.q_lora_rank)
    self.kv_norm = norm(norm_config, rngs, config.kv_lora_rank)
    self.kv_b = NativeLinear(config, mesh, rngs, config.kv_lora_rank, h * (config.qk_nope_head_dim + config.v_head_dim))
    self.gate = NativeLinear(config, mesh, rngs, config.emb_dim, h * config.v_head_dim)
    self.out = NativeLinear(config, mesh, rngs, h * config.v_head_dim, config.emb_dim)

  def __call__(self, x, segments):
    cfg = self.config
    b, s = x.shape[:2]
    h, d = cfg.num_query_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    q = self.q_b(self.q_norm(self.q_a(x))).reshape(b, s, h, d)
    kv = self.kv_a(x)
    latent, shared_key = jnp.split(kv, [cfg.kv_lora_rank], axis=-1)
    kv = self.kv_b(self.kv_norm(latent)).reshape(b, s, h, cfg.qk_nope_head_dim + cfg.v_head_dim)
    key, value = jnp.split(kv, [cfg.qk_nope_head_dim], axis=-1)
    key = jnp.concatenate((key, jnp.broadcast_to(shared_key[:, :, None], (b, s, h, cfg.qk_rope_head_dim))), axis=-1)
    mask = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    mask &= jnp.arange(s)[:, None] >= jnp.arange(s)[None, :]
    # Native XLA attention; no TE/cuDNN/Pallas attention backend.
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), key.astype(jnp.float32)) / math.sqrt(d)
    probabilities = jax.nn.softmax(jnp.where(mask[:, None], scores, -1e30), axis=-1).astype(x.dtype)
    output = jnp.einsum("bhqk,bkhv->bqhv", probabilities, value).reshape(b, s, -1)
    output = output * jax.nn.sigmoid(self.gate(x))
    return self.out(output) * (segments != 0)[..., None]


class AttentionResidual(nnx.Module):

  def __init__(self, config, mesh, rngs):
    self.eps = config.normalization_layer_epsilon
    self.scale = nnx.Param(jnp.ones(config.emb_dim, config.weight_dtype), out_sharding=("norm",))
    self.query = nnx.Param(
        jax.random.normal(rngs.params(), (config.emb_dim,), dtype=config.weight_dtype) * 0.02, out_sharding=("norm",)
    )

  def __call__(self, current, history, count):
    values = jnp.concatenate((history, current[:, :, None]), axis=2).astype(jnp.float32)
    keys = values * jax.lax.rsqrt(jnp.mean(values**2, axis=-1, keepdims=True) + self.eps)
    scores = jnp.sum(keys * (self.scale[...].astype(jnp.float32) * self.query[...].astype(jnp.float32)), axis=-1)
    valid = (jnp.arange(history.shape[2] + 1) < count) | (jnp.arange(history.shape[2] + 1) == history.shape[2])
    weights = jax.nn.softmax(jnp.where(valid, scores, -1e30), axis=-1)
    return jnp.sum(weights[..., None] * values, axis=2).astype(current.dtype)


class KimiLayer(nnx.Module):

  def __init__(self, config, mesh, rngs, layer_idx, model_mode="train", quant=None):
    del model_mode, quant
    self.config = config
    self.boundary = layer_idx % config.kimi_attn_res_block_size == 0
    self.attn_res = AttentionResidual(config, mesh, rngs)
    self.ffn_res = AttentionResidual(config, mesh, rngs)
    self.attn_norm = norm(config, rngs, config.emb_dim)
    self.ffn_norm = norm(config, rngs, config.emb_dim)
    self.attention = (KimiMLA if layer_idx + 1 in config.kimi_full_attn_layers else KimiKDA)(config, mesh, rngs)
    self.is_moe = layer_idx >= config.first_num_dense_layers
    self.ffn = (
        KimiMoE(config, mesh, rngs) if self.is_moe else NativeFFN(config, mesh, rngs, config.emb_dim, config.mlp_dim)
    )

  def __call__(self, carry, segments, collect_balance):
    current, history, count = carry
    # Reference mixes the previous partial block before opening a new block.
    mixed = self.attn_res(current, history, count)
    if self.boundary:
      history = history.at[:, :, count].set(current)
      count = count + 1
      current = jnp.zeros_like(current)
    current = current + self.attention(self.attn_norm(mixed), segments)
    mixed = self.ffn_norm(self.ffn_res(current, history, count))
    output = self.ffn(mixed, segments, collect_balance) if self.is_moe else self.ffn(mixed)
    return current + output, history, count


class KimiBlock(nnx.Module):

  def __init__(self, config, mesh, rngs, start, length):
    self.layers = nnx.List([KimiLayer(config, mesh, rngs, i) for i in range(start, start + length)])

  def __call__(self, carry, segments, collect_balance):
    for layer in self.layers:
      # Functional remat threads all nonparameter state, including QB and overflow.
      graph, params, other = nnx.split(layer, nnx.Param, ...)

      def call(params, other, carry):
        module = nnx.merge(graph, params, other)
        result = module(carry, segments, collect_balance)
        return result, nnx.state(module, nnx.Not(nnx.Param))

      result, other = (
          jax.checkpoint(call)(params, other, carry)
          if layer.config.remat_policy != "none"
          else call(params, other, carry)
      )
      nnx.update(layer, other)
      carry = result
    return carry


class KimiBackbone(nnx.Module):

  def __init__(self, config, mesh, rngs):
    self.config = config
    size, n = config.kimi_attn_res_block_size, config.num_decoder_layers
    prefix = min(size, n)
    # Keep the final block static: the last layer may be a terminal MLA exception.
    self.middle_count = max((n - prefix - 1) // size, 0) if config.scan_layers else 0
    self.first = KimiBlock(config, mesh, rngs, 0, prefix)
    if self.middle_count:
      self.middle = nnx.vmap(
          lambda r: KimiBlock(config, mesh, r, prefix, size),
          in_axes=0,
          out_axes=nnx.StateAxes({nnx.Param: 0, ...: 0}),
          transform_metadata={nnx.PARTITION_NAME: None},
      )(rngs.fork(split=self.middle_count))
    self.tail = KimiBlock(config, mesh, rngs, prefix + self.middle_count * size, n - prefix - self.middle_count * size)
    self.output_res = AttentionResidual(config, mesh, rngs)
    max_logging.log(
        "Kimi-K3 TE SURROGATE: latent-input router; routed SwiGLU. "
        "See docs/guides/kimi_k3_te_compatibility.md (K3-TE-001, K3-TE-002)."
    )

  def __call__(self, x, segments, collect_balance):
    n = math.ceil(self.config.num_decoder_layers / self.config.kimi_attn_res_block_size)
    carry = (x, jnp.zeros((*x.shape[:2], n, x.shape[-1]), x.dtype), jnp.array(0, jnp.int32))
    carry = self.first(carry, segments, collect_balance)
    if self.middle_count:
      graph, params, other = nnx.split(self.middle, nnx.Param, ...)

      def body(carry, xs):
        p, o = xs
        block = nnx.merge(graph, p, o)
        carry = block(carry, segments, collect_balance)
        return carry, nnx.state(block, nnx.Not(nnx.Param))

      carry, other = jax.lax.scan(body, carry, (params, other))
      nnx.update(self.middle, other)
    carry = self.tail(carry, segments, collect_balance)
    return self.output_res(*carry)
