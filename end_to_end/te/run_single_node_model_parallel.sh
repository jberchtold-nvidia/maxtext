#!/bin/bash
set -euo pipefail

# Default values
MODEL="llama2-70b-8-layers"
OUTPUT_DIR_TAG=""
STEPS=50
TRACE=false

# Parse keyword-style arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir-tag)
            OUTPUT_DIR_TAG="$2"
            shift 2
            ;;
        --trace)
            TRACE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model MODEL] [--output-dir-tag OUTPUT_DIR_TAG] [--trace true|false] [--steps STEPS]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--model MODEL] [--output-dir-tag OUTPUT_DIR_TAG] [--trace true|false] [--steps STEPS]"
            exit 1
            ;;
    esac
done

# Now your variables are set as needed
echo "MODEL=$MODEL"
echo "OUTPUT_DIR_TAG=$OUTPUT_DIR_TAG"
echo "TRACE=$TRACE"
echo "STEPS=$STEPS"

WARMUP_STEPS=10
if (( STEPS <= WARMUP_STEPS )); then
    echo "ERROR: STEPS ($STEPS) must be greater than WARMUP_STEPS ($WARMUP_STEPS)"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAXTEXT_DIR="$(realpath "$SCRIPT_DIR/../../")"
OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL}${OUTPUT_DIR_TAG:+_$OUTPUT_DIR_TAG}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

BASE_ARGS="--model $MODEL --steps $STEPS"
TE_BASE_ARGS="--te-dense true --te_mlp true --te-norm true"
TE_RECIPES=("DelayedScaling" "MXFP8BlockScaling")
TE_ENV="NVTE_JAX_CUSTOM_CALLS='NormFwdPrimitive=false,NormBwdPrimitive=false,GemmPrimitive=true'"
  
n_gpus=$(nvidia-smi -L | wc -l)
half_gpus=$((n_gpus / 2))
# List of experiments: <DP> <TP> <TPSP> <FSDP>
experiments=(
  # "1        $n_gpus     1           1"        # Single DP, full TP
  "$n_gpus  1           1           1"        # Full DP, single TP
  # "2        $half_gpus  1           1"        # DP=2, TP=half GPUs
  # "1        1           $n_gpus     1"        # Sequence parallelism maxed
  # "2        1           $half_gpus  1"        # DP=2, TPSP=half GPUs
  # "1        1           1           $n_gpus"  # FSDP across all GPUs
  # "1        1           $half_gpus  2"        # FSDP=2, TPSP=half GPUs
)

CSV="$OUTPUT_DIR/raw_results.csv"
echo -e "test\tdp\ttp\ttpsp\tfsdp\tmean\tstddev" > "$CSV"

run_and_parse() {
  local test="$1"
  local dp="$2"
  local tp="$3"
  local tpsp="$4"
  local fsdp="$5"
  local cmd="$6"
  local stdout="$OUTPUT_DIR/run_${test}_dp${dp}_tp${tp}_tpsp${tpsp}_fsdp${fsdp}.log"
  echo "===== Executing ${test}\t${dp}\t${tp}\t${tpsp}\t${fsdp} ====="
  set +e
  $cmd 2>&1 | tee "$stdout"
  set -e
  # Exclude the warning steps for warning up and last step for tracing
  ths=$(grep 'Tokens/s/device:' "$stdout" | sed '1,'"${WARMUP_STEPS}"'d;$d' | awk -F'Tokens/s/device: ' '{print $2}' | awk -F',' '{print $1}')

  if [ -z "$ths" ]; then
    mean="NA"
    stddev="NA"
  else
    mean_stddev=$(echo "$ths" | python3 -c "import sys; import numpy as np
arr = [float(l.strip()) for l in sys.stdin if l.strip()]
if arr:
  print(f'{np.mean(arr):.2f}\t{np.std(arr, ddof=1):.2f}')
else:
  print('NA\tNA')
"
    )
    mean=$(echo "$mean_stddev" | cut -f1)
    stddev=$(echo "$mean_stddev" | cut -f2)
  fi
  echo -e "${test}\t${dp}\t${tp}\t${tpsp}\t${fsdp}\t${mean}\t${stddev}" >> "$CSV"

  if [[ "$TRACE" == "true" ]]; then
    TRACE_SRC=$(grep -oE '/tmp/tmp\.[^ ]+' "$stdout" | head -n1)
    if [[ -n "$TRACE_SRC" && -e "$TRACE_SRC" ]]; then
      TRACE_DEST="${OUTPUT_DIR}/trace_${test}_dp${dp}_tp${tp}_tpsp${tpsp}_fsdp${fsdp}"
      mv "$TRACE_SRC" "$TRACE_DEST"
      echo "Trace moved: $TRACE_SRC -> $TRACE_DEST"
    else
      echo "No trace file found for $test, dp=$dp, tp=$tp, tpsp=$tpsp, fsdp=$fsdp"
    fi
  fi
}


for exp in "${experiments[@]}"; do
  read dp tp tpsp fsdp <<< "$exp"

  args="--data-parallel=$dp --tensor-parallel=$tp --tensor-sequence-parallel=$tpsp --fsdp=$fsdp"

  # MaxText FP8 baseline
  test=maxtext_fp8
  run_and_parse "$test" "$dp" "$tp" "$tpsp" "$fsdp" \
    "env PYTHONPATH=${MAXTEXT_DIR}:\$PYTHONPATH bash ${SCRIPT_DIR}/test-maxtext-te.sh $args --dtype=fp8 --trace=$TRACE $BASE_ARGS"

  # TE variants
  for recipe in "${TE_RECIPES[@]}"; do
    test="te_fp8_${recipe}"
    TE_ARGS_ALL="$TE_BASE_ARGS --te-fp8 true --te-recipe $recipe"
    run_and_parse "$test" "$dp" "$tp" "$tpsp" "$fsdp" \
      "env $TE_ENV PYTHONPATH=${MAXTEXT_DIR}:\$PYTHONPATH bash ${SCRIPT_DIR}/test-maxtext-te.sh $args $TE_ARGS_ALL --trace=$TRACE $BASE_ARGS"
  done
done

OUTPUT_FORMAT="txt" # txt or csv
echo "Experiments finished. Raw CSV at $CSV"
python3 $SCRIPT_DIR/normalize.py "$CSV" "${OUTPUT_DIR}/summary.$OUTPUT_FORMAT" "$OUTPUT_FORMAT"
cat "${OUTPUT_DIR}/summary.$OUTPUT_FORMAT"
