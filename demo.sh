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
BATCH_SIZE="4"
NUM_ROUNDS="10"
GPU_MEMORY_UTILIZATION="0.7"
MAX_MODEL_LEN="512"
LAYER_IDS="all"
DTYPE="bfloat16"

RUN_ROOT="/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo"
SINGLE_ROOT="${RUN_ROOT}/single"
BATCH_ROOT="${RUN_ROOT}/batch_b${BATCH_SIZE}_r${NUM_ROUNDS}"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/demo_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${SINGLE_ROOT}" "${BATCH_ROOT}" "${LOG_DIR}"

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
  echo "LAYER_IDS: ${LAYER_IDS}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "NUM_ROUNDS: ${NUM_ROUNDS}"
  echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
  echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"

  run_step "GPU status before run" \
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

  run_step "Single sample: generate GT and run vLLM single test" \
    "${PYTHON}" examples/offline_inference/prompt_embed_single_demo.py \
      --model "${MODEL}" \
      --run-root "${SINGLE_ROOT}" \
      --gpu "${GPU}" \
      --seq-len "${SEQ_LEN}" \
      --layer-ids "${LAYER_IDS}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

  run_step "Batch sample: generate single/batch GT and run vLLM batch test" \
    "${PYTHON}" examples/offline_inference/prompt_embed_batch_demo.py \
      --model "${MODEL}" \
      --input-pth "${SINGLE_ROOT}/single_gt.input.pth" \
      --run-root "${BATCH_ROOT}" \
      --gpu "${GPU}" \
      --layer-ids "${LAYER_IDS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-rounds "${NUM_ROUNDS}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

  echo
  echo "===== Output files ====="
  echo "Single root: ${SINGLE_ROOT}"
  echo "Batch root:  ${BATCH_ROOT}"
  echo "Log file:    ${LOG_FILE}"

  echo
  echo "===== Key summary lines ====="
  grep -E \
    "SINGLE_DEMO_RESULT|BATCH_DEMO_RESULT|single_generated_ids_match|single_prompt_embeds_vs_token_hidden_states|single_vs_batch_summary|async_vs_batch_summary|async_vs_single_summary|status=FAIL|max_abs|mean_abs|Prompt-embeds vs token hidden states|Generated ids match" \
    "${LOG_FILE}" || true
}

main 2>&1 | tee "${LOG_FILE}"

 -e NVIDIA_DISABLE_REQUIRE=1

docker run -d \
  --name vllm-memory-repro \
  --gpus '"device=3"' \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 2333:2333 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /root/public/models:/root/public/models:ro \
  verlai/verl:vllm018.dev1 \
  sleep infinity

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(torch.ones(1, device="cuda") if torch.cuda.is_available() else None)
PY
