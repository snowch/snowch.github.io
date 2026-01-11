---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# L21 - Deployment & Serving [DRAFT]

*From research code to production-ready LLM serving*

---

You've trained and optimized your model—now how do you serve it to thousands of users with low latency and high throughput? This lesson covers production serving frameworks and techniques.

By the end of this post, you'll understand:
- vLLM and continuous batching (10× throughput improvement)
- Serving frameworks (TGI, TensorRT-LLM)
- Speculative decoding (2× faster generation)
- Monitoring and observability
- Cost optimization strategies

---

## Part 1: The Inference Problem

### Naive Serving is Inefficient

```python
# Naive approach: Process one request at a time
def naive_server(model, tokenizer):
    while True:
        request = receive_request()
        input_ids = tokenizer.encode(request.prompt)

        # Generate (blocks server for 2-5 seconds!)
        output_ids = model.generate(input_ids, max_new_tokens=200)

        response = tokenizer.decode(output_ids)
        send_response(response)

# Throughput: ~0.5 requests/second (terrible!)
```

**Problems**:
1. **Sequential processing**: One request at a time
2. **No batching**: GPU sits idle during tokenization/decoding
3. **Fixed max_tokens**: Padding wastes compute

---

## Part 2: Static Batching (Baseline)

### Batch Multiple Requests

```python
def static_batching_server(model, batch_size=8, max_tokens=200):
    request_queue = Queue()

    while True:
        # Collect batch_size requests
        batch = []
        for _ in range(batch_size):
            batch.append(request_queue.get())

        # Tokenize and pad to max_tokens
        input_ids = []
        for req in batch:
            tokens = tokenizer.encode(req.prompt)
            tokens = pad_to_length(tokens, max_tokens)
            input_ids.append(tokens)

        input_ids = torch.stack(input_ids)

        # Generate for entire batch
        output_ids = model.generate(input_ids, max_new_tokens=200)

        # Send responses
        for i, req in enumerate(batch):
            response = tokenizer.decode(output_ids[i])
            send_response(req, response)

# Throughput: ~4 requests/second (8× better!)
```

**Problems**:
1. **Wait for full batch**: Adds latency
2. **Padding waste**: Short prompts padded to max length
3. **Early finishing**: Some sequences finish early, but batch continues

---

## Part 3: Continuous Batching (vLLM)

### The Key Innovation

**Idea**: Add/remove requests from batch dynamically as they finish.

```python
# Conceptual vLLM approach
class ContinuousBatchingServer:
    def __init__(self, model):
        self.model = model
        self.active_requests = []

    def step(self):
        """Generate one token for all active requests."""

        # 1. Add new requests from queue (non-blocking)
        while not request_queue.empty() and len(self.active_requests) < max_batch:
            self.active_requests.append(request_queue.get_nowait())

        # 2. Prepare batch (no padding!)
        input_ids = [req.get_next_input() for req in self.active_requests]

        # 3. Generate next token for all
        logits = self.model(input_ids)
        next_tokens = logits.argmax(dim=-1)

        # 4. Update requests
        finished = []
        for i, req in enumerate(self.active_requests):
            req.append_token(next_tokens[i])

            # Check if finished
            if req.is_finished():
                send_response(req)
                finished.append(i)

        # 5. Remove finished requests
        for i in reversed(finished):
            del self.active_requests[i]

    def run(self):
        while True:
            self.step()
```

**Advantages**:
1. **No waiting**: Requests added immediately
2. **No padding**: Each sequence has actual length
3. **Dynamic removal**: Finished sequences removed instantly

**Result**: **10-20× higher throughput** than static batching!

---

## Part 4: vLLM in Practice

### Installation and Usage

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# Initialize vLLM engine
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # Use 2 GPUs
    max_num_seqs=256,  # Max concurrent sequences
    max_model_len=4096,  # Context length
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=200
)

# Serve requests
prompts = [
    "The capital of France is",
    "Write a poem about AI",
    "Explain quantum computing"
]

# Generate (batched automatically!)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### vLLM Server (OpenAI-compatible API)

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --max-num-seqs 256
```

```python
# Client code (same as OpenAI API!)
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

response = openai.Completion.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="The capital of France is",
    max_tokens=50
)

print(response.choices[0].text)
```

---

## Part 5: Other Serving Frameworks

### HuggingFace Text Generation Inference (TGI)

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 2
```

**Features**:
- Continuous batching (like vLLM)
- Flash Attention support
- Quantization (INT8, GPTQ)
- Streaming responses

---

### TensorRT-LLM (NVIDIA)

```python
import tensorrt_llm

# Build optimized engine
builder = tensorrt_llm.Builder()
engine = builder.build_engine(
    model_path="llama-2-7b",
    max_batch_size=256,
    max_input_len=2048,
    max_output_len=512
)

# Serve with Triton Inference Server
```

**Advantages**:
- Fastest (NVIDIA-optimized kernels)
- Supports INT8, FP8 quantization
- Multi-GPU/multi-node support

**Disadvantages**:
- NVIDIA GPUs only
- More complex setup

---

## Part 6: Speculative Decoding

### 2× Faster Generation

**Problem**: Autoregressive generation is sequential (can't parallelize).

**Idea**: Use a small "draft" model to generate multiple tokens, verify with large model.

```python
def speculative_decoding(large_model, small_model, input_ids, num_tokens=100):
    """
    Generate with speculative decoding.
    """
    generated = input_ids

    while len(generated) < num_tokens:
        # 1. Draft: Small model generates K tokens quickly
        draft_tokens = small_model.generate(generated, max_new_tokens=4, do_sample=False)

        # 2. Verify: Large model computes logits for all K tokens at once
        logits = large_model(torch.cat([generated, draft_tokens], dim=-1))

        # 3. Accept/reject each draft token
        accepted = []
        for i, token in enumerate(draft_tokens):
            predicted_token = logits[:, len(generated) + i - 1].argmax()

            if predicted_token == token:
                accepted.append(token)  # Accept
            else:
                accepted.append(predicted_token)  # Reject, use large model's token
                break  # Stop at first rejection

        generated = torch.cat([generated, torch.tensor(accepted)], dim=-1)

    return generated
```

**Speed-up**:
- If draft model agrees 75% of the time: **2-3× faster**
- If 50%: **1.5× faster**
- Guaranteed to match large model's distribution!

**Used by**: Claude (Anthropic), GPT-4 Turbo (rumored)

---

## Part 7: Monitoring and Observability

### Key Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')

# Token metrics
tokens_generated = Counter('llm_tokens_generated_total', 'Tokens generated')
tokens_per_second = Gauge('llm_tokens_per_second', 'Generation throughput')

# Resource metrics
gpu_memory_used = Gauge('llm_gpu_memory_bytes', 'GPU memory usage')
batch_size = Gauge('llm_batch_size', 'Current batch size')

# Example: Tracking generation
@request_latency.time()
def generate_with_metrics(prompt):
    request_count.inc()

    output = llm.generate(prompt)

    tokens_generated.inc(len(output))
    tokens_per_second.set(len(output) / elapsed_time)

    return output
```

### Dashboard Example (Grafana)

```yaml
# Key panels:
- Request latency (p50, p95, p99)
- Throughput (requests/sec, tokens/sec)
- GPU utilization (%, memory)
- Batch size over time
- Error rate
```

---

## Part 8: Cost Optimization

### Strategy 1: Quantization

```python
# INT8: 2× memory reduction → 2× more requests per GPU
llm = LLM(model="llama-2-7b", quantization="int8")

# Savings: $100/day → $50/day (50% cost reduction)
```

### Strategy 2: Dynamic Batching

```python
# Adjust batch size based on load
if request_queue_length > 100:
    max_batch_size = 256  # High load: maximize throughput
else:
    max_batch_size = 32   # Low load: minimize latency
```

### Strategy 3: Model Swapping

```python
# Use smaller model for simple prompts
def route_request(prompt):
    complexity = estimate_complexity(prompt)

    if complexity < 0.3:
        return small_model.generate(prompt)  # 1B model
    else:
        return large_model.generate(prompt)  # 7B model

# Saves 60-80% of compute for typical workloads!
```

### Strategy 4: KV Cache Sharing

```python
# Share KV cache for common prefixes (e.g., system prompts)
system_prompt = "You are a helpful assistant."

# Cache system prompt's KV (computed once)
cached_kv = model.compute_kv(system_prompt)

# Reuse for all requests
for user_prompt in user_prompts:
    full_prompt = system_prompt + user_prompt
    output = model.generate_with_cached_kv(full_prompt, cached_kv)

# Saves 20-30% compute!
```

---

## Part 9: Production Checklist

### Before Going Live

```
✅ Load testing (target: 1000 req/sec)
✅ Failure modes tested (OOM, crash recovery)
✅ Monitoring & alerting configured
✅ Autoscaling rules defined
✅ Cost tracking enabled
✅ Model versioning strategy
✅ A/B testing infrastructure
✅ Rate limiting & DDoS protection
✅ Privacy & safety filters
✅ Backup & disaster recovery plan
```

---

## Part 10: Full Production Example

```python
# main.py - Production LLM Server
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
import logging

# Initialize
app = FastAPI()
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    max_num_seqs=256
)

# Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Generation endpoint
@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 200):
    try:
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens
        )

        outputs = llm.generate([prompt], sampling_params)
        return {"text": outputs[0].outputs[0].text}

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Summary

1. **vLLM**: 10-20× throughput via continuous batching
2. **Serving frameworks**: TGI (HuggingFace), TensorRT-LLM (NVIDIA)
3. **Speculative decoding**: 2-3× faster generation
4. **Monitoring**: Track latency, throughput, GPU usage
5. **Cost optimization**: Quantization, batching, model routing
6. **Production checklist**: Load testing, monitoring, autoscaling

**Congratulations!** You've completed the entire LLM From Scratch series, from tokenization to production deployment. You now have the knowledge to build, train, optimize, and serve LLMs at scale.

---
