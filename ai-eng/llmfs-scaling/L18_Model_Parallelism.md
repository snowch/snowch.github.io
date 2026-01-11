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

# L18 - Model Parallelism [DRAFT]

*Training models too large for a single GPU*

---

A 70B parameter model requires 280 GB in FP32, but consumer GPUs max out at 24 GB. **Model parallelism** splits the model across multiple GPUs. This lesson covers the three main strategies.

By the end of this post, you'll understand:
- Data parallelism (simplest, for small models)
- Pipeline parallelism (split layers across GPUs)
- Tensor parallelism (split individual layers)
- ZeRO optimizer (memory-efficient training)

---

## Part 1: Data Parallelism

### The Simplest Strategy

**Idea**: Replicate the entire model on each GPU, process different batches.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Create model on this GPU
rank = dist.get_rank()
model = GPT(config).to(rank)
model = DDP(model, device_ids=[rank])

# Each GPU gets different data
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, sampler=sampler)

for batch in loader:
    # Forward pass (local)
    loss = model(batch)

    # Backward pass
    loss.backward()

    # All-reduce gradients (average across GPUs)
    optimizer.step()
```

**How it works**:
1. Each GPU processes a different mini-batch
2. After backward pass, gradients are averaged across all GPUs
3. All GPUs update with the same averaged gradient

**Limitations**:
- Model must fit on ONE GPU
- Doesn't help for 70B models!

---

## Part 2: Pipeline Parallelism

### Split Layers Across GPUs

**Idea**: Put different layers on different GPUs, pass activations between them.

```
GPU 0: Layers 0-7   (Embedding + first 8 blocks)
GPU 1: Layers 8-15  (Middle 8 blocks)
GPU 2: Layers 16-23 (Last 8 blocks)
GPU 3: Final head   (LM head)
```

### Naive Pipeline (Sequential)

```python
class PipelineParallelGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Split layers across GPUs
        self.embed = nn.Embedding(vocab_size, d_model).to('cuda:0')

        self.blocks_0_7 = nn.ModuleList([
            TransformerBlock(config) for _ in range(8)
        ]).to('cuda:0')

        self.blocks_8_15 = nn.ModuleList([
            TransformerBlock(config) for _ in range(8)
        ]).to('cuda:1')

        self.blocks_16_23 = nn.ModuleList([
            TransformerBlock(config) for _ in range(8)
        ]).to('cuda:2')

        self.lm_head = nn.Linear(d_model, vocab_size).to('cuda:3')

    def forward(self, input_ids):
        # GPU 0
        x = self.embed(input_ids.to('cuda:0'))
        for block in self.blocks_0_7:
            x = block(x)

        # GPU 1
        x = x.to('cuda:1')
        for block in self.blocks_8_15:
            x = block(x)

        # GPU 2
        x = x.to('cuda:2')
        for block in self.blocks_16_23:
            x = block(x)

        # GPU 3
        x = x.to('cuda:3')
        logits = self.lm_head(x)

        return logits
```

**Problem**: GPU utilization is terrible!

```
Time:  0  1  2  3  4  5  6  7  8
GPU 0: [████]
GPU 1:      [████]
GPU 2:           [████]
GPU 3:                [████]

Utilization: 25%! (3 GPUs idle at any time)
```

---

### GPipe: Pipeline with Micro-Batches

**Solution**: Split batch into micro-batches, keep GPUs busy.

```python
# Split batch into 4 micro-batches
batch_size = 32
micro_batch_size = 8
num_micro_batches = batch_size // micro_batch_size

for micro_batch_id in range(num_micro_batches):
    start = micro_batch_id * micro_batch_size
    end = start + micro_batch_size

    micro_batch = input_ids[start:end]

    # Process through pipeline (while next micro-batch starts on GPU 0)
    logits = model(micro_batch)
```

**Timeline with micro-batches**:

```
Time:  0  1  2  3  4  5  6  7  8  9  10
GPU 0: [m0][m1][m2][m3]
GPU 1:     [m0][m1][m2][m3]
GPU 2:         [m0][m1][m2][m3]
GPU 3:             [m0][m1][m2][m3]

Utilization: ~75%! (much better)
```

**Library**: Use `torch.distributed.pipeline.sync.Pipe`

---

## Part 3: Tensor Parallelism

### Split Individual Layers

Instead of splitting layers, split the **weights within each layer**.

**Example**: Split feed-forward layer across 2 GPUs

```python
# Standard FF (4096 → 16384 → 4096)
class FeedForward(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(4096, 16384)  # 67M params
        self.fc2 = nn.Linear(16384, 4096)  # 67M params

# Tensor Parallel FF
class TensorParallelFF(nn.Module):
    def __init__(self):
        # Split fc1 along output dimension
        self.fc1_gpu0 = nn.Linear(4096, 8192).to('cuda:0')  # 33M params
        self.fc1_gpu1 = nn.Linear(4096, 8192).to('cuda:1')  # 33M params

        # Split fc2 along input dimension
        self.fc2_gpu0 = nn.Linear(8192, 4096).to('cuda:0')  # 33M params
        self.fc2_gpu1 = nn.Linear(8192, 4096).to('cuda:1')  # 33M params

    def forward(self, x):
        # Each GPU computes half the outputs
        x0 = F.gelu(self.fc1_gpu0(x.to('cuda:0')))
        x1 = F.gelu(self.fc1_gpu1(x.to('cuda:1')))

        # Each GPU computes partial fc2
        y0 = self.fc2_gpu0(x0)
        y1 = self.fc2_gpu1(x1)

        # All-reduce to combine results
        y = y0.to('cuda:0') + y1.to('cuda:0')
        return y
```

**Key insight**: Matrix multiplication is easy to split!

$$Y = XW \quad \text{where } W = [W_1 \mid W_2]$$

$$Y = X[W_1 \mid W_2] = [XW_1 \mid XW_2]$$

Each GPU computes one partition, then concatenate.

**Library**: Use `torch.distributed.tensor.parallel`

---

## Part 4: ZeRO Optimizer

### The Memory Breakdown (7B model)

| **Component** | **Memory** |
|---|---|
| Model weights | 28 GB |
| Gradients | 28 GB |
| **Optimizer states** | **56 GB** |
| Activations | 10 GB |
| **Total** | **122 GB** |

**Observation**: Optimizer states (Adam's momentum and variance) dominate!

---

### ZeRO: Zero Redundancy Optimizer

**Idea**: Shard optimizer states across GPUs.

**ZeRO Stage 1**: Partition optimizer states
```python
from deepspeed import DeepSpeedConfig

config = {
    "zero_optimization": {
        "stage": 1,  # Shard optimizer states only
    }
}

# Each GPU stores 1/N of optimizer states
# 56 GB / 8 GPUs = 7 GB per GPU
```

**ZeRO Stage 2**: Also partition gradients
```python
config = {
    "zero_optimization": {
        "stage": 2,  # Shard optimizer states + gradients
    }
}

# 56 GB optimizer + 28 GB gradients = 84 GB / 8 = 10.5 GB per GPU
```

**ZeRO Stage 3**: Also partition model weights
```python
config = {
    "zero_optimization": {
        "stage": 3,  # Shard everything!
    }
}

# Total 122 GB / 8 GPUs = 15 GB per GPU
```

**Trade-off**: More communication between GPUs, but enables training huge models.

---

## Part 5: Choosing the Right Strategy

| **Model Size** | **Strategy** | **Why** |
|---|---|---|
| < 1B | Data Parallelism | Fits on one GPU, just need speed |
| 1B - 10B | Pipeline Parallelism | Too big for one GPU, moderate communication |
| 10B - 100B | Tensor Parallelism + ZeRO | Need to split layers AND optimizer |
| 100B+ | All of the above (3D parallelism) | Maximum splitting |

**Example (GPT-3 175B)**:
- 64 GPUs total
- 8-way data parallelism (8 replicas)
- 4-way pipeline parallelism (4 stages)
- 2-way tensor parallelism (2 GPUs per layer)
- ZeRO Stage 2

---

## Part 6: Practical Example with DeepSpeed

```python
import deepspeed

# DeepSpeed config
ds_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2
        "offload_optimizer": {
            "device": "cpu",  # Offload to CPU for even more memory
            "pin_memory": True
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training loop (same as before!)
for batch in train_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**What DeepSpeed handles**:
- Sharding parameters across GPUs
- Communication between GPUs
- CPU offloading
- Mixed precision
- Gradient checkpointing

---

## Summary

1. **Data Parallelism**: Replicate model, process different batches (small models)
2. **Pipeline Parallelism**: Split layers across GPUs (medium models)
3. **Tensor Parallelism**: Split individual layer weights (large models)
4. **ZeRO**: Shard optimizer states to save memory (critical for huge models)
5. **DeepSpeed**: Production library that handles everything

**Next Up: L18 – Long Context Handling.** How models handle 100k+ token contexts!

---
