#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${DEMO_ROOT:=/home/dpsk_a2a/vllm-memory-demo}"
: "${VLLM_CONNECTOR_FILE:=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py}"
: "${PYTHON:=python}"

PATCHED_CONNECTOR="${PACKAGE_ROOT}/patches/example_hidden_states_connector.py"

if [[ ! -f "${PATCHED_CONNECTOR}" ]]; then
  echo "ERROR: missing patched connector: ${PATCHED_CONNECTOR}" >&2
  exit 1
fi

if [[ ! -f "${PACKAGE_ROOT}/docker_demo.sh" ]]; then
  echo "ERROR: missing docker demo script: ${PACKAGE_ROOT}/docker_demo.sh" >&2
  exit 1
fi

if [[ ! -d "${PACKAGE_ROOT}/examples/offline_inference" ]]; then
  echo "ERROR: missing bundled examples: ${PACKAGE_ROOT}/examples/offline_inference" >&2
  exit 1
fi

echo "Package root: ${PACKAGE_ROOT}"
echo "Demo root: ${DEMO_ROOT}"
echo "vLLM connector target: ${VLLM_CONNECTOR_FILE}"

if [[ ! -f "${VLLM_CONNECTOR_FILE}" ]]; then
  echo "ERROR: target connector does not exist: ${VLLM_CONNECTOR_FILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${VLLM_CONNECTOR_FILE}")"
backup_file="${VLLM_CONNECTOR_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
cp -a "${VLLM_CONNECTOR_FILE}" "${backup_file}"
cp -a "${PATCHED_CONNECTOR}" "${VLLM_CONNECTOR_FILE}"

if [[ "${PACKAGE_ROOT}" != "${DEMO_ROOT}" ]]; then
  mkdir -p "${DEMO_ROOT}/examples"
  cp -a "${PACKAGE_ROOT}/docker_demo.sh" "${DEMO_ROOT}/docker_demo.sh"
  rm -rf "${DEMO_ROOT}/examples/offline_inference"
  cp -a "${PACKAGE_ROOT}/examples/offline_inference" \
    "${DEMO_ROOT}/examples/offline_inference"
fi

chmod +x "${DEMO_ROOT}/docker_demo.sh"

echo
echo "Installed patched connector."
echo "Backup: ${backup_file}"
echo
echo "Installed demo files:"
echo "  ${DEMO_ROOT}/docker_demo.sh"
echo "  ${DEMO_ROOT}/examples/offline_inference"

echo
echo "===== vLLM connector check ====="
"${PYTHON}" - <<'PY'
import inspect
import vllm
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as connector,
)

print("vllm file:", inspect.getfile(vllm))
print("connector file:", inspect.getfile(connector))
print(
    "has token_ids_from_request_data:",
    hasattr(connector, "token_ids_from_request_data"),
)
if not hasattr(connector, "token_ids_from_request_data"):
    raise SystemExit("patched connector was not imported")
PY

echo
echo "Next:"
echo "  cd ${DEMO_ROOT}"
echo "  bash docker_demo.sh"
