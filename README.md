# vLLM Prompt Embeds Hidden States Demo

## start

说明：

- 这个仓库是完整的 vLLM 包。为了避免在目标容器里重新安装整个 vLLM，
  当前流程只把仓库里的必要 patch 和 demo 文件 copy 出来运行。
- `vllm-memory-demo/setup.sh` 会把 patched connector 覆盖到容器中已安装的
  vLLM package，并把 demo 脚本安装到 `/home/dpsk_a2a/vllm-memory-demo`。

```bash
docker rm -f vllm-memory-repro-patch 2>/dev/null || true # 避免重名，先删一下

docker run -dit \
--name vllm-memory-repro-patch \
--gpus '"device=7"' \
--ipc=host \
--shm-size=32g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-p 2335:2333 \
-e NVIDIA_DISABLE_REQUIRE=1 \
-e HF_HOME=/workspace/.cache/huggingface \
-e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
-e HF_ENDPOINT=https://hf-mirror.com \
-e HF_HUB_DISABLE_XET=1 \
-e HF_HUB_DISABLE_PROGRESS_BARS=0 \
-e HF_HUB_VERBOSITY=info \
-e TRANSFORMERS_VERBOSITY=info \
-e PYTHONUNBUFFERED=1 \
-v /root/public/models:/root/public/models:ro \
docker.1ms.run/verlai/verl:vllm018.dev1 \
bash
```

注：`docker.1ms.run/verlai/verl:vllm018.dev1` 是镜像源地址和当前示例 tag。
如果镜像源或版本不同，需要改这里的 registry/tag。

```bash
docker exec -it vllm-memory-repro-patch bash
```

容器内：

```bash
set -euo pipefail

git clone https://gh-proxy.com/https://github.com/zzf1955/vllm_memory.git vllm-memory-src
```

注：`https://gh-proxy.com/https://github.com/zzf1955/vllm_memory.git` 是
GitHub clone 镜像源。如果网络能直连 GitHub，可以改成原始仓库地址。

```bash
cd /home/dpsk_a2a/vllm-memory-src/vllm-memory-demo
bash setup.sh

cd /home/dpsk_a2a/vllm-memory-demo

MODEL=Qwen/Qwen3-0.6B \
GPU=0 \
HF_HOME=/workspace/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DISABLE_PROGRESS_BARS=0 \
HF_HUB_VERBOSITY=info \
TRANSFORMERS_VERBOSITY=info \
PYTHONUNBUFFERED=1 \
bash docker_demo.sh
```

注：`docker_demo.sh` 里可以修改显存使用比例 `GPU_MEMORY_UTILIZATION`。
当前默认是 `0.1`，因为 demo 只用了 `Qwen/Qwen3-0.6B` 这个 0.6B 模型。

## input/output

输入是拼接后的 prompt embedding 序列：

```text
<random tensor> <prompt emb> <random tensor>
```

默认配置下，前后两段 random tensor 各 8 个位置，中间 prompt embedding 为 16
个位置，因此实际输入长度为 32。

输出是所有选定 layer 的 hidden states。默认 `LAYER_IDS=all`，即比较所有
layer。测试会分别跑 single sequence 和 batch sequence，并把 AsyncLLM 输出
与 `LLM.generate()` 生成的 Ground Truth 做逐元素误差对比。

结果会输出两组 summary：

- `single_vs_single`：AsyncLLM single 输出 vs `LLM.generate()` single Ground Truth。
- `batch_vs_batch`：AsyncLLM batch 输出 vs `LLM.generate()` batch Ground Truth。

运行产物包括 single sequence 目录、batch sequence 目录和 `docker_demo.sh`
生成的 log 文件；具体路径会在脚本最后的 summary 中打印。

## Runtime 修改位置

核心 runtime 修改只有一处：

```text
vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
```

修改目的：

```text
让 ExampleHiddenStatesConnector 支持纯 prompt_embeds 输入时提取 hidden states。
```

原问题是：当请求只传 `prompt_embeds`、不传 `prompt_token_ids` 时，
connector 会把 token 长度当成 0，导致保存出来的 hidden states 为空。

现在的处理是：

```text
如果 prompt_token_ids 存在，继续使用真实 token ids。
如果 prompt_token_ids 为 None，则从 prompt_embeds 推出 prompt length，
并用 [0] * prompt_len 作为 connector 内部保存 hidden states 时的占位 token ids。
```

这个占位 token ids 不参与模型输入，只用于告诉 connector 应该保存多少个
prompt 位置的 hidden states。真实输入仍然是 `prompt_embeds`。
