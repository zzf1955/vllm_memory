#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

export HF_HOME="/disk_n/zzf/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/disk_n/zzf/.cache/huggingface/hub"
export UV_CACHE_DIR="/disk_n/zzf/.cache/uv"
export PIP_CACHE_DIR="/disk_n/zzf/.pip_cache"
export TMPDIR="/disk_n/zzf/tmp"
export HF_ENDPOINT="https://hf-mirror.com"
export http_proxy="http://127.0.0.1:20171"
export https_proxy="http://127.0.0.1:20171"
export no_proxy="localhost,127.0.0.1"

PYTHON="python"
GPU="1"
MODEL="Qwen/Qwen3-0.6B"
SEQ_LEN="16"
MEM_LEN="8"
BATCH_SIZE="8"
NUM_ROUNDS="20"
GPU_MEMORY_UTILIZATION="0.2"
MAX_MODEL_LEN="512"
LAYER_IDS="1"
DTYPE="bfloat16"

RUN_ROOT="/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo"
GT_PREFIX="${RUN_ROOT}/qwen3_gt"
LOG_DIR="${RUN_ROOT}/logs"
SINGLE_WORK_DIR="${RUN_ROOT}/single"
CONCURRENT_WORK_DIR="${RUN_ROOT}/concurrent_b${BATCH_SIZE}_r${NUM_ROUNDS}"
MEM_WORK_DIR="${RUN_ROOT}/memory_b${BATCH_SIZE}_r${NUM_ROUNDS}"
LOG_FILE="${LOG_DIR}/demo_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${SINGLE_WORK_DIR}" \
  "${CONCURRENT_WORK_DIR}" "${MEM_WORK_DIR}"

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "+ $*"
  "$@"
}

main() {
  if ! "${PYTHON}" -c "import vllm" >/dev/null 2>&1; then
    echo "ERROR: Python cannot import vLLM."
    echo "Please activate the environment first:"
    echo "  conda activate /disk_n/zzf/conda_envs/vllm_memory"
    echo "Then run:"
    echo "  bash demo.sh"
    exit 1
  fi

  echo "Demo root: ${RUN_ROOT}"
  echo "Log file: ${LOG_FILE}"
  echo "Python: ${PYTHON}"
  echo "GPU: ${GPU}"
  echo "Model: ${MODEL}"
  echo "SEQ_LEN: ${SEQ_LEN}"
  echo "MEM_LEN: ${MEM_LEN}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "NUM_ROUNDS: ${NUM_ROUNDS}"
  echo "LAYER_IDS: ${LAYER_IDS}"
  echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
  echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
  echo

  run_step "GPU status before run" \
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

  run_step "Generate Hugging Face ground truth" \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py \
      --model "${MODEL}" \
      --seq-len "${SEQ_LEN}" \
      --device cuda \
      --dtype "${DTYPE}" \
      --output "${GT_PREFIX}.pth"

  run_step "Single prompt-embeds vs LLM.generate baseline" \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      examples/offline_inference/prompt_embed_extract_hidden_states.py \
      --model "${MODEL}" \
      --input-pth "${GT_PREFIX}.input.pth" \
      --output-pth "${GT_PREFIX}.output.pth" \
      --work-dir "${SINGLE_WORK_DIR}" \
      --layer-ids "${LAYER_IDS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --dtype "${DTYPE}"

  run_step "Concurrent prompt-embeds stress test" \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py \
      --model "${MODEL}" \
      --input-pth "${GT_PREFIX}.input.pth" \
      --output-pth "${GT_PREFIX}.output.pth" \
      --work-dir "${CONCURRENT_WORK_DIR}" \
      --layer-ids "${LAYER_IDS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-rounds "${NUM_ROUNDS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --dtype "${DTYPE}"

  run_step "MEM prompt-embeds stress test" \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
      --model "${MODEL}" \
      --input-pth "${GT_PREFIX}.input.pth" \
      --work-dir "${MEM_WORK_DIR}" \
      --layer-ids "${LAYER_IDS}" \
      --mem-len "${MEM_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --num-rounds "${NUM_ROUNDS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --dtype "${DTYPE}"

  echo
  echo "===== Output files ====="
  echo "Ground truth prompt: ${GT_PREFIX}.prompt.txt"
  echo "Ground truth input:  ${GT_PREFIX}.input.pth"
  echo "Ground truth output: ${GT_PREFIX}.output.pth"
  echo "Single work dir:     ${SINGLE_WORK_DIR}"
  echo "Concurrent work dir: ${CONCURRENT_WORK_DIR}"
  echo "MEM work dir:        ${MEM_WORK_DIR}"
  echo "Log file:            ${LOG_FILE}"

  echo
  echo "===== Key summary lines ====="
  grep -E \
    "Generated ids match|Stress test summary|Memory stress test summary|total_requests|failed_compares|prefix_max_abs|memory_max_abs|max_abs|mean_abs_avg|expected_shape|generated_ids|baseline_generated_ids|Prompt-embeds vs HF hidden states" \
    "${LOG_FILE}" || true
}

main 2>&1 | tee "${LOG_FILE}"
