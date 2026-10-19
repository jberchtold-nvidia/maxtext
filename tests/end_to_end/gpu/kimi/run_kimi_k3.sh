#!/usr/bin/env bash
# Single-node K3 TE surrogate. See docs/guides/kimi_k3_te_compatibility.md.
set -euo pipefail
MAXTEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
WORKSPACE_DIR="$(dirname "$MAXTEXT_DIR")"
export MAXTEXT_DIR
export PYTHONPATH="${WORKSPACE_DIR}/site_overrides:${MAXTEXT_DIR}/src:${PYTHONPATH:-}"
export DECOUPLE_GCLOUD=TRUE
export DECOUPLE_GLOUD=TRUE
export NVTE_JAX_ENFORCE_V2_GROUPED_GEMM=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-16}"
export XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE=false
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=180
export LD_LIBRARY_PATH="${WORKSPACE_DIR}/transformerengine/3rdparty/nccl-extensions/build/lib:${LD_LIBRARY_PATH:-}"
GPUS="${GPUS:-4}"
case "$GPUS" in
  4) EP=2; FSDP=2 ;;
  1) echo 'Existing TE ep_bootstrap requires world_size >= 2; use component tests for one GPU and GPUS=4 for MoE training.' >&2; exit 2 ;;
  *) echo 'GPUS must be 1 or 4' >&2; exit 2 ;;
esac
PRESET="${PRESET:-production-width}"
case "$PRESET" in
  tiny) MODEL=kimi-k3-tiny-te-surrogate ;;
  production-width) MODEL=kimi-k3-2.8t-te-surrogate ;;
  *) echo 'PRESET must be tiny or production-width' >&2; exit 2 ;;
esac
OUTPUT="${OUTPUT:-${WORKSPACE_DIR}/outputs/kimi_k3_${PRESET}_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-20}"
LENGTH="${MAX_TARGET_LENGTH:-1024}"
EXTRA="override_model_config=true base_num_decoder_layers=4 num_experts=32 kimi_full_attn_layers=[4] quantization= te_gmm_quantization=${TE_GMM_QUANTIZATION:-te_no_quant} ici_expert_parallelism=$EP ici_fsdp_parallelism=$FSDP ici_data_parallelism=1 dcn_expert_parallelism=1 dcn_fsdp_parallelism=1 dcn_data_parallelism=1 packing=false scan_layers=true max_target_length=$LENGTH float32_gate_logits=true weight_dtype=bfloat16 mu_dtype=bfloat16 shardy=true ragged_buffer_factor=0.0 te_ep_overflow_check_every_n_steps=1 logits_dot_in_fp32=true use_iota_embed=false log_config=false log_step_diagnostics=true abort_on_nan_loss=true enable_checkpointing=${CHECKPOINT:-false} checkpoint_period=10 async_checkpointing=false jax_cache_dir=$OUTPUT/jax_cache ${EXTRA_MAXTEXT_ARGS:-}"
CMD=(test-maxtext.sh --multiprocess -n 1 -b "${BATCH_PER_GPU:-1}"
  --model-name="$MODEL" --attn-type=dot_product --remat-policy=full
  --steps="$STEPS" --data-parallel=1 --fsdp=1 --tensor-parallel=1
  --tensor-sequence-parallel=1 --mem-fraction 0.85 --output "$OUTPUT" -a "$EXTRA")
if [[ "${MOE_PARITY:-0}" == 1 ]]; then
  MODEL=kimi-k3-tiny-te-surrogate
  CMD=(python3 "$MAXTEXT_DIR/tests/end_to_end/gpu/kimi/check_te_moe.py")
  EXTRA="validation_mode=moe_parity model_name=$MODEL max_target_length=32 EP=2 FSDP=2"
elif [[ -n "${RESTORE_PARITY:-}" ]]; then
  MODEL=kimi-k3-tiny-te-surrogate
  CMD=(python3 "$MAXTEXT_DIR/tests/end_to_end/gpu/kimi/check_restore.py")
  EXTRA="validation_mode=restore_parity model_name=$MODEL checkpoint=$RESTORE_PARITY EP=2 FSDP=2"
fi
printf 'Kimi-K3 TE SURROGATE: latent router; routed SwiGLU. EP=%s FSDP=%s\n' "$EP" "$FSDP"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
mkdir -p "$OUTPUT/logs"
printf '%s\n' "model_name=$MODEL kimi_te_surrogate=true $EXTRA" > "$OUTPUT/resolved-overrides.txt"
printf 'K3-TE-001: latent router\nK3-TE-002: routed SwiGLU\ndocs/guides/kimi_k3_te_compatibility.md\n' > "$OUTPUT/architecture-deviations.txt"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
START=$SECONDS
for (( rank=0; rank<GPUS; rank++ )); do
  (
    export CUDA_VISIBLE_DEVICES="$rank"
    export SLURM_STEP_GPUS="$rank" SLURM_NTASKS="$GPUS" SLURM_NPROCS="$GPUS" SLURM_STEP_NUM_TASKS="$GPUS"
    export SLURM_PROCID="$rank" SLURM_LOCALID="$rank"
    unset OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK
    export JAX_COORDINATOR_IP=127.0.0.1 JAX_COORDINATOR_PORT="${COORDINATOR_PORT:-21666}"
    export NNODES="$GPUS" NODE_RANK="$rank"
    export JAX_COORDINATOR_ADDRESS="127.0.0.1:$JAX_COORDINATOR_PORT"
    export MAXTEXT_JAX_COORDINATOR_ADDRESS="$JAX_COORDINATOR_ADDRESS"
    export JAX_PROCESS_COUNT="$GPUS" JAX_PROCESS_INDEX="$rank" JAX_LOCAL_DEVICE_IDS=0
    if [[ "${MOE_PARITY:-0}" != 1 && -z "${RESTORE_PARITY:-}" ]]; then
      CMD[-1]="${CMD[-1]} metrics_file=$OUTPUT/metrics_rank_$rank.jsonl"
    fi
    "${CMD[@]}"
  ) > "$OUTPUT/logs/proc_$rank.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do if ! wait "$pid"; then FAILED=1; fi; done
printf 'exit=%s elapsed=%ss\n' "$FAILED" "$((SECONDS-START))" | tee "$OUTPUT/summary.txt"
printf 'Results: %s\n' "$OUTPUT"
if (( FAILED )); then tail -60 "$OUTPUT/logs/proc_0.log"; fi
exit "$FAILED"
