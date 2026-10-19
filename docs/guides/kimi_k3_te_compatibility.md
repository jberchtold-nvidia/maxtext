# Kimi-K3 text training with the existing TE MoEBlock

This implementation is a **TE surrogate**, not exact Kimi-K3. All model code,
configuration, launchers, and compatibility changes live in MaxText. TE is unchanged.

## Reference and scope

The architecture source is
[moonshotai/Kimi-K3 at c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721](https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721),
particularly `config.json` and `modeling_kimi_linear.py`. The training balancing
rule comes from the [K3 technical report, section 2.3.3](https://arxiv.org/html/2607.24653v1).
KDA safe decay matches FLA's `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`.

Implemented scope: random-initialized text training with native KDA, gated NoPE
MLA, block attention residuals, normalized latent MoE, native dense/shared SiTU
FFNs, and quantile balancing. Routed experts use the existing full TE MoEBlock.
Vision, tokenizer/weight import, inference caches, Muon, and the complete published
training recipe are outside this implementation. Reference HF MoE training itself
raises `NotImplementedError`, so its forward math, independent differentiable
oracles, and the paper define the correctness tests.

## Architectural deviations required by TE

These IDs are also comments at the affected model code and configs. They must
remain visible in startup logs and resolved configuration. Removing a restriction
requires capability checks and passing the restoration tests below.

### K3-TE-001: router consumes latent input

**Correct K3:** compute router logits from the original 7168-dimensional hidden
state; independently down-project activations to 3584 dimensions before dispatch
to experts. Router weights have shape `[7168, 896]`.

**Current implementation:** down-project first; TE computes routing and experts
from the same 3584-dimensional input. Router weights have shape `[3584, 896]`.
Native quantile-balancing statistics use that same surrogate router. The outer
latent projections, post-combine latent RMSNorm, shared experts, and expert sizes
remain intact. Router gradients now pass through the latent down-projection;
this changes the model and is not checkpoint-compatible with the official router.

**Why:** `RoutedMoE.__call__` rejects separate `gate_inputs` with `te_moe_block`.
TE `jax/moe.py::moe` accepts a single `x` shared by routing and expert computation.

**Future TE capability:** independently shaped router and expert inputs, including
custom-VJP gradients, FSDP/EP partitioning, and checkpoint/rematerialization support.
MaxText can then restore full-width router parameters and pass original hidden
states separately. This requires a new checkpoint/model version; existing latent
router weights cannot simply be reinterpreted as full-width weights.

**Restoration tests:** compare routing IDs, normalized probabilities, forward
outputs, and gradients for original hidden states, latent inputs, both projection
matrices, router weights, and expert weights against a test oracle. Exercise
EP=2/FSDP=2 and nonzero expert-selection biases.

### K3-TE-002: routed SwiGLU replaces SiTU-GLU

**Correct K3:** for expert gate/up projections `a,b`, compute
`4*tanh(a/4)*sigmoid(a) * 25*tanh(b/25)` in FP32 before casting back.

**Current implementation:** TE computes `silu(a)*b` in routed experts. Native
dense and shared FFNs still use the correct two-branch SiTU. Expert parameter
shapes are unchanged, but outputs and derivatives differ, especially at large
magnitudes. This implementation does not claim official weight/logit parity.

**Why:** TE's unfused FFN applies `activation_type` to the gate branch only and
multiplies it by an unchanged up branch. A unary callback cannot express the
up-branch soft cap. Its cuDNN fusion path also requires supported GLU semantics.

**Future TE capability:** two-input activation with independently configurable
SiTU gate/up soft caps, FP32 activation evaluation, correct custom-VJP residuals,
and support or an explicit fallback in BF16 and MXFP8 expert execution.

**Restoration tests:** exact scalar activation/gradient checks, saturated positive
and negative inputs, complete expert output/input/weight gradient parity, and
EP/FSDP forward/backward parity at both expert precisions.

## Behavior retained without TE changes

- TE owns routing, dispatch/combine, grouped expert GEMMs, and backward.
- TE's sigmoid top-k implementation normalizes selected *unbiased* scores in
  forward and backward. Bias affects selection only. MaxText's K3-specific
  validation permits this; it does not double-normalize TE output or alter
  restrictions for existing DSv3/DSv4 configurations.
- Quantile balancing is native MaxText state outside the TE custom VJP. Detached
  duplicate router logits supply statistics only. Histogram counts pool global
  token shards and accumulate across microbatches; the next bias is installed
  once after the optimizer step, with no gradient through histogram updates.
- The latent output order is combine, RMSNorm, up-projection, then shared addition.
- Non-MoE computation is native JAX/XLA. `quantization` is empty; only routed
  experts may select TE MXFP8. The router's initialization bypasses standalone
  TE dense kernels through `unquantized_modules: [gate]`.

## Numerical and engineering choices

These choices are distinct from architectural deviations:

- BF16 parameters/activations, FP32 recurrent state, attention scores, and balancing
  statistics; MaxText initialization and AdamW rather than the published recipe.
- KDA uses chunkwise triangular solves with chunk size 64 and rematerialization;
  a separate token recurrence provides an output/gradient oracle.
- QB uses 256 fixed bins over `[-2,2]` and nearest-rank bin-center quantiles. Fixed
  bounds allow additive pooling across accumulation steps without rebinning.
  Sigmoid scores are in `[0,1]`; centered exact QB biases span at most one,
  bounding margins in this interval. Test quantiles within bin-width error.
  This differs from the plan's dynamic-range histogram proposal, not from its
  intended global-batch balancing behavior.
- Full-size scanning keeps the first 12 layers and final nine layers static and
  scans six 12-layer blocks with `param_scan_axis=0`. A fixed-size residual-history buffer preserves block
  boundary semantics. `scan_layers=false` uses the same residual algorithm.
- Unpacked sequences are required. Native kernels additionally test padding and
  independent segments. Tensor/context/pipeline parallelism beyond the validated
  EP/FSDP layout is not a scaling claim.

## Configurations and local runs

- `kimi-k3-2.8t`: full text architectural specification; intentionally fails before
  allocation on existing TE. It must never silently become a surrogate.
- `kimi-k3-2.8t-te-surrogate`: full dimensions with the two explicit deviations.
- `kimi-k3-tiny-te-surrogate`: reduced dimensions for development.

The full configuration retains 93 layers, 896 routed experts, top-16 selection,
7168 hidden width, 3584 latent width, 3072 expert FFN width, two shared experts,
and vocabulary 163840. The full-size surrogate router has fewer parameters than
correct K3. Abstract shape/memory accounting must label that difference.
The 1M maximum position setting records model capacity; it is not a demonstrated
1M-context native-XLA training result.

From the MaxText repository root:

```bash
DRY_RUN=1 tests/end_to_end/gpu/kimi/run_kimi_k3.sh
PRESET=tiny tests/end_to_end/gpu/kimi/run_kimi_k3.sh
# Default: production widths, four layers, 32 experts, top-16, sequence 1024,
# batch one/GPU, EP=2/FSDP=2, 20 steps, BF16 routed experts.
tests/end_to_end/gpu/kimi/run_kimi_k3.sh
TE_GMM_QUANTIZATION=te_mxfp8 STEPS=5 tests/end_to_end/gpu/kimi/run_kimi_k3.sh
CHECKPOINT=true tests/end_to_end/gpu/kimi/run_kimi_k3.sh
```

`OUTPUT`, `COORDINATOR_PORT`, `BATCH_PER_GPU`, and `EXTRA_MAXTEXT_ARGS` are
optional overrides. Each run saves rank logs, overrides, deviation IDs, and exit
status. Worst-case receive buffers are used to prevent silent token dropping.
For memory limits reduce sequence length to 512, then select `PRESET=tiny`.
One GPU per process is required by the existing TE bootstrap. That bootstrap
requires at least two ranks, so complete single-GPU MoE training is unavailable;
use component tests for one-device iteration and four processes for training.

Unit tests:

```bash
JAX_PLATFORMS=cpu DECOUPLE_GCLOUD=TRUE PYTHONPATH=src \
  python -m pytest tests/unit/kimi_k3_test.py -q
```

## Validation record

Validation results and any remaining runtime limitations are recorded here during
implementation. A passing surrogate run is not evidence of exact K3 correctness,
production-scale convergence, or multi-node feasibility.

Full-size abstract initialization passed: 2,779,188,957,280 surrogate parameters,
295,436,288 additional parameters needed for full-width exact routers, and 902
parameter-sharding validations. With BF16 Adam momentum, parameters occupy
5,558,379,623,552 bytes and optimizer state 11,116,757,538,124 bytes. Even ideal
four-way state sharding needs 4,168,784,290,419 bytes/GPU, before gradients,
activations, replication, dispatch buffers, and workspaces. This is a shape and
lower-bound memory check, not evidence that the full model fits this node.

The installed TE MoEBlock uses `ep_axis`/`data_parallelism_axes`, whereas the local
TE checkout uses `mesh_resource`. The MaxText adapter detects the supported
signature and preserves the full MoEBlock path. This is API compatibility, not
an architectural deviation. Unsupported requested quantization-before-gather
features fail explicitly on older TE.

Run isolated BF16 TE output/input/router/expert-gradient parity on the same mesh:

```bash
MOE_PARITY=1 OUTPUT=/tmp/k3-moe-parity tests/end_to_end/gpu/kimi/run_kimi_k3.sh
PYTHONPATH=src DECOUPLE_GCLOUD=TRUE python tests/end_to_end/gpu/kimi/inspect_kimi_k3.py
```

The parity oracle applies normalized unbiased sigmoid weights before the expert
down-projection, matching the TE adapter's early-weighting choice. It uses native
ops only in the test process; production always retains TE routing and dispatch.

Hardware validation (GB200×4, one process/GPU, EP=2/FSDP=2):

| Check | Result |
|---|---|
| Tiny BF16 training, sequence 64 | Two steps passed; loss 7.390 → 7.375. |
| Production widths, four layers, 32 experts, sequence 1024 | 20 BF16 steps passed; loss 12.514 → 0.003 on repeated synthetic data. |
| Production widths with routed-expert MXFP8 | Five steps passed; loss 12.514 → 4.596. |
| BF16 isolated TE/native parity | Relative RMS errors: output 0.00354, input gradient 0.00517, router gradient 0.01127, first expert weight gradient 0.00601, second expert weight gradient 0.00478. |
| Existing TE-bootstrap / DeepSeek routing-bias regressions | 13 tests and five subtests passed. |

Production-width slice has 8.361B parameters. Compiler-estimated memory was
22.2 GB for BF16 and 20.8 GB for MXFP8 per device; these are compiler estimates,
not measured peak HBM. Excluding first-step setup, the late BF16 steps took about
0.15 s. No performance-tuning claim is made. Worst-case dispatch capacity was
used and no overflow was reported. MXFP8 and BF16 run lengths differ, so their
learning-rate schedules are not a paired numerical-loss comparison.

Evidence logs are under `/tmp/k3-four-v2`, `/tmp/k3-production-bf16-final`,
`/tmp/k3-production-mxfp8`, and `/tmp/k3-parity-bf16`. The final BF16 launcher
completed with exit zero. Its metrics confirm finite losses and gradient norms,
nonzero balancing updates, and zero overflow on all 20 steps.

The combined correctness/regression suite passed **25 tests and five subtests**.
This includes exact CPU optimizer/checkpoint continuation with the native test
oracle and output/gradient checks across residual-block and KDA chunk boundaries.

Distributed checkpoint validation saved at zero-based step 10, resumed at step
11, and reached step 19 successfully. All nine routed router/expert matrices
changed during training. Continued losses differ from the uninterrupted run by
at most 0.000248; the largest per-parameter relative RMS difference is 0.000562.
The GPU continuation is not bitwise identical: small optimizer moments differ
by up to 0.0653 relative RMS after further training. Exact checkpoint restoration
is checked separately from low-precision training continuation.

`check_restore.py` compares every addressable restored shard against the saved
checkpoint, including optimizer, RNG, expert-selection bias, and histogram state.
All **342 arrays match exactly on each of four ranks**. The continuation comparison
reports optimizer differences and gates model parameters/metrics at the BF16
2% tolerance; the separate CPU round-trip requires exact restored-step state.

Reproduce checkpoint checks with a tiny slice:

```bash
PRESET=tiny MAX_TARGET_LENGTH=64 CHECKPOINT=true STEPS=20 OUTPUT=/tmp/k3-baseline \
  tests/end_to_end/gpu/kimi/run_kimi_k3.sh
PRESET=tiny MAX_TARGET_LENGTH=64 CHECKPOINT=true STEPS=20 OUTPUT=/tmp/k3-resumed \
  EXTRA_MAXTEXT_ARGS='load_full_state_path=/tmp/k3-baseline/logdir/checkpoints/10/items' \
  tests/end_to_end/gpu/kimi/run_kimi_k3.sh
RESTORE_PARITY=/tmp/k3-baseline/logdir/checkpoints/10/items OUTPUT=/tmp/k3-restore-check \
  tests/end_to_end/gpu/kimi/run_kimi_k3.sh
JAX_PLATFORMS=cpu python tests/end_to_end/gpu/kimi/compare_resume.py /tmp/k3-baseline /tmp/k3-resumed
```

The launcher writes per-rank `metrics_rank_<rank>.jsonl` files to avoid concurrent
writers. The comparison tool also accepts the older shared metrics file used by
the initial validation runs. State comparison evidence is in
`/tmp/k3-resume-comparison-final.log`; exact restoration evidence is in
`/tmp/k3-restore-exact`.

A 37-layer tiny-width model also passed three EP=2/FSDP=2 steps with two
accumulated microbatches, repeated scanned 12-layer residual blocks, and an extra
terminal MLA layer. Loss decreased 7.453 → 7.313, gradient norms stayed finite,
balancing updates were nonzero, and overflow was zero. The cold run took 518 s,
mostly compilation; the ordinary four-layer slice remains the development default.
Evidence: `/tmp/k3-scanned-ga`, including separate per-rank metrics files.

```bash
PRESET=tiny BATCH_PER_GPU=2 MAX_TARGET_LENGTH=64 STEPS=3 \
  EXTRA_MAXTEXT_ARGS='gradient_accumulation_steps=2 base_num_decoder_layers=37 kimi_full_attn_layers=[4,8,12,16,20,24,28,32,36,37]' \
  tests/end_to_end/gpu/kimi/run_kimi_k3.sh
```
