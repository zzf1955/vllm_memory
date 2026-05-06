# Prompt Embeds Hidden States Demo Usage

## 目标

验证 vLLM 0.18.1 上的两条路径：

```text
single input: prompt_embeds -> hidden_states
batch input:  batch prompt_embeds -> hidden_states
```

当前 demo 使用：

```text
model: Qwen/Qwen3-0.6B
prompt length: 16
hidden size: 1024
layer ids: all, i.e. 1..28
max_tokens: 1
```

## 环境

先激活环境：

```bash
conda activate /disk_n/zzf/conda_envs/vllm_memory
cd /disk_n/zzf/vllm_memory
```

缓存、临时目录和代理由 `demo.sh` 内置：

```bash
HF_HOME=/disk_n/zzf/.cache/huggingface
HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
UV_CACHE_DIR=/disk_n/zzf/.cache/uv
PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
TMPDIR=/disk_n/zzf/tmp
HF_ENDPOINT=https://hf-mirror.com
http_proxy=http://127.0.0.1:20171
https_proxy=http://127.0.0.1:20171
no_proxy=localhost,127.0.0.1
```

## 一键 Demo

直接运行：

```bash
bash demo.sh
```

`demo.sh` 当前内置参数：

```text
GPU=1
MODEL=Qwen/Qwen3-0.6B
SEQ_LEN=16
LAYER_IDS=all
BATCH_SIZE=4
NUM_ROUNDS=10
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=512
DTYPE=bfloat16
RUN_ROOT=/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo
```

脚本串行跑两个入口：

```text
1. examples/offline_inference/prompt_embed_single_demo.py
2. examples/offline_inference/prompt_embed_batch_demo.py
```

输出会包含：

```text
SINGLE_DEMO_RESULT: PASS/FAIL
BATCH_DEMO_RESULT: PASS/FAIL
single_vs_batch_summary
async_vs_batch_summary
async_vs_single_summary
```

完整日志保存在：

```text
/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo/logs/
```

## 脚本 1：单样例 GT + vLLM 单样例测试

入口：

```text
examples/offline_inference/prompt_embed_single_demo.py
```

作用：

```text
1. 用 HuggingFace forward 生成单样例 GT。
2. 保存 prompt、input tensor、output tensor。
3. 用 vLLM LLM.generate token prompt 跑 baseline。
4. 用 vLLM AsyncLLM.generate prompt_embeds 跑测试。
5. 比较两条 vLLM 路径的 generated token ids 和 hidden_states。
```

命令：

```bash
python examples/offline_inference/prompt_embed_single_demo.py \
  --gpu 1 \
  --run-root /disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo/single \
  --layer-ids all
```

输出文件：

```text
single_gt.prompt.txt
single_gt.input.pth
single_gt.output.pth
vllm-single/
```

关键输出：

```text
Generated ids match: True
Prompt-embeds vs token hidden states: {'allclose': True, ...}
SINGLE_DEMO_RESULT: PASS
```

判定标准：

```text
PASS = vLLM prompt_embeds hidden_states 和 vLLM token prompt hidden_states 一致
```

注意：HF 全层 hidden states 只作为参考。最后一层可能和 vLLM `extract_hidden_states`
抽取点语义不同，不作为该 demo 的通过条件。

## 脚本 2：batch GT + vLLM batch 测试

入口：

```text
examples/offline_inference/prompt_embed_batch_demo.py
```

作用：

```text
1. 读取单样例 .input.pth 中的 prompt_embeds。
2. 构造 batch prompt embeddings。
3. 生成 single GT：每个 batch 样例逐条单独 LLM.generate。
4. 生成 batch GT：整个 batch 一次性 LLM.generate。
5. 用 AsyncLLM 连续提交 batch 请求。
6. 每个 Async 输出同时对比 single GT 和 batch GT。
```

batch 输入：

```text
b0 = prompt_embeds * 1.00
b1 = prompt_embeds * 1.01
b2 = prompt_embeds * 1.02
b3 = prompt_embeds * 1.03
```

命令：

```bash
python examples/offline_inference/prompt_embed_batch_demo.py \
  --gpu 1 \
  --input-pth /disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo/single/single_gt.input.pth \
  --run-root /disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo/batch_b4_r10 \
  --layer-ids all \
  --batch-size 4 \
  --num-rounds 10
```

输出目录：

```text
llm-single-ground-truth/
llm-batch-ground-truth/
async-prompt-embeds-batch-only/
```

每个样例都会输出两类比较：

```text
sample=b000 batch_gt_vs_single_gt status=...
round=000 sample=b000 async_vs_batch_gt status=...
round=000 sample=b000 async_vs_single_gt status=...
```

最终汇总：

```text
single_vs_batch_summary: {...}
async_vs_batch_summary: {...}
async_vs_single_summary: {...}
BATCH_DEMO_RESULT: PASS/FAIL
```

判定标准：

```text
PASS = AsyncLLM batch 输出和 batch GT 全部一致
```

`single GT` 不是 batch 测试的通过标准，它只用于展示 single sequence 和 batch sequence
的特殊数值行为。

## Runtime 修改

唯一必要的 vLLM runtime 修改在：

```text
vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
```

原问题：

```python
token_ids = new_req.prompt_token_ids or []
```

纯 `prompt_embeds` 请求里 `prompt_token_ids` 是 `None`，旧逻辑会把 prompt 长度当成 0，
导致 connector 保存出来的 hidden states 长度为 0。

当前修复：

```python
from vllm.utils import length_from_prompt_token_ids_or_embeds


def token_ids_from_request_data(request_data: NewRequestData) -> list[int]:
    if request_data.prompt_token_ids is not None:
        return request_data.prompt_token_ids

    prompt_len = length_from_prompt_token_ids_or_embeds(
        request_data.prompt_token_ids,
        request_data.prompt_embeds,
    )
    return [0] * prompt_len
```

然后在 new request 和 cached request 分支都使用：

```python
token_ids_from_request_data(...)
```

行为：

```text
token prompt:      保存真实 token ids，行为不变。
prompt_embeds:     保存 [0] * prompt_len 作为占位 token ids。
hidden_states:     按真实 prompt_embeds 长度保存。
```

这里的 `[0] * prompt_len` 不是模型输入，只是 connector 保存 safetensors 时的长度占位。
模型实际输入仍然是 `prompt_embeds`。

## Prefix Cache

这些 demo 都显式设置：

```python
enable_prefix_caching=False
```

原因：

```text
prompt 文本相同不代表 prompt_embeds 相同。
如果 prefix cache key 只按 token/text 判断，不区分 embeddings 内容，
不同 prompt_embeds 可能被错误复用。
```

生产环境如果要打开 prefix cache，需要先确认 prompt embeddings 的 cache key 语义。

## Single Seq 和 Batch Seq 的特殊行为

全层 hidden states 下，vLLM 的 single sequence 执行和 batch sequence 执行不是严格逐元素一致。

已观察到：

```text
batch=4, layer_ids=all:
  batch GT vs single GT failed = 3 / 4
  max_abs = 32.0
```

因此不能用 single GT 作为 batch 输出的严格 reference。

正确的 batch 测试应该比较：

```text
AsyncLLM batch output vs LLM.generate batch GT
```

而不是：

```text
AsyncLLM batch output vs LLM.generate single GT
```

当前 batch=4/r=10 的结果：

```text
AsyncLLM batch vs batch GT:
  failed = 0 / 40
  max_abs = 0.0
```

更高压力下曾观察到：

```text
batch=8, rounds=20:
  AsyncLLM batch vs batch GT failed = 2 / 160
  failures: round 13 b0/b1
  max_abs = 96.0
```

这说明：

```text
single-vs-batch 差异是确定存在的 reference mismatch；
batch=8 的少量 Async-vs-batch 异常才是后续需要继续定位的真实并发问题。
```

## 相关底层脚本

两个入口脚本内部复用了这些底层工具：

```text
examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py
examples/offline_inference/prompt_embed_extract_hidden_states.py
examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py
examples/offline_inference/prompt_embed_async_batch_only.py
```

MEM 拼接测试仍保留在：

```text
examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py
```
