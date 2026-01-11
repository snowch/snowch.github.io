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

# L13 - Data Loading Pipelines at Scale [DRAFT]

*From toy datasets to production-ready data processing*

---

In previous lessons, we used simple datasets that fit in memory. But real LLM training requires terabytes of data streamed from disk or cloud storage. This lesson shows how to build efficient data pipelines that don't become the bottleneck.

By the end of this post, you'll understand:
- Efficient dataset loading (WebDataset, streaming)
- Data preprocessing & cleaning strategies
- Creating data shards for distributed training
- Balancing datasets (deduplication, quality filtering)
- The "data mix" problem

---

## Part 1: The Scale Problem

### Why Simple DataLoaders Break

```python
# This works for MNIST (50MB)
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# This FAILS for LLM training (500GB+)
# ❌ Can't load 500GB into RAM
# ❌ Can't even list all filenames (millions of files)
```

**The challenge**: You need to:
1. Stream data from storage (disk/S3) without loading everything
2. Shuffle data across epochs (but can't load all to shuffle)
3. Resume from checkpoints (know which data you've seen)
4. Scale to multiple GPUs (shard data across workers)

---

## Part 2: WebDataset Format

### The Tar Shard Approach

Instead of individual files, pack data into `.tar` archives:

```bash
# Dataset structure
data/
  train-0000.tar    # 10,000 samples
  train-0001.tar    # 10,000 samples
  ...
  train-0099.tar    # 10,000 samples (1M samples total)
```

Each `.tar` contains:
```
0000000.txt    # Document 0
0000001.txt    # Document 1
...
0009999.txt    # Document 9999
```

### Creating WebDataset Shards

```python
import webdataset as wds

def create_shards(documents, output_pattern, shard_size=10000):
    """
    documents: List of text strings
    output_pattern: "data/train-%04d.tar"
    """
    with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
        for idx, doc in enumerate(documents):
            sink.write({
                "__key__": f"{idx:07d}",
                "txt": doc,
            })

# Usage
documents = load_my_corpus()  # List of strings
create_shards(documents, "data/train-%04d.tar")
```

---

## Part 3: Streaming Data Loading

### Basic WebDataset Pipeline

```python
import webdataset as wds

dataset = (
    wds.WebDataset("data/train-{0000..0099}.tar")
    .decode()  # Decode tar contents
    .to_tuple("txt")  # Extract text field
    .map(tokenize_function)  # Tokenize
    .batched(32)  # Create batches
)

loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)

for batch in loader:
    # batch is already tokenized and batched
    train_step(batch)
```

### Shuffling with Shards

```python
dataset = (
    wds.WebDataset("data/train-{0000..0099}.tar")
    .shuffle(1000)  # Shuffle buffer of 1000 samples
    .decode()
    .to_tuple("txt")
    .map(tokenize_function)
    .batched(32)
)
```

**How shuffle buffer works**:
1. Load 1000 samples into memory
2. Randomly sample from this buffer
3. When buffer drops below 500, load next 500 from stream
4. Provides "local" shuffling without loading entire dataset

---

## Part 4: Data Preprocessing Pipeline

### Quality Filtering

Not all web text is training-worthy. Common filters:

```python
def quality_filter(doc):
    """Return True if document is high quality."""

    # Filter 1: Length
    if len(doc) < 100 or len(doc) > 100000:
        return False

    # Filter 2: Language (English only)
    if detect_language(doc) != 'en':
        return False

    # Filter 3: Profanity / toxicity
    if contains_profanity(doc):
        return False

    # Filter 4: Repetition (common in spam)
    if has_excessive_repetition(doc):
        return False

    # Filter 5: Code-to-text ratio (filter code dumps)
    if code_ratio(doc) > 0.5:
        return False

    return True

# Apply during shard creation
filtered_docs = [doc for doc in documents if quality_filter(doc)]
```

**Visualization: Quality Distribution**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Bar chart showing:
# - Before filtering: 100M documents
# - After length filter: 85M (-15%)
# - After language filter: 70M (-15%)
# - After toxicity filter: 68M (-2%)
# - Final dataset: 68M documents
```

---

### Deduplication

Web crawls contain massive duplication (copied articles, boilerplate).

```python
from datasketch import MinHash, MinHashLSH

def deduplicate_corpus(documents, threshold=0.8):
    """Remove near-duplicate documents using MinHash."""

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_docs = []

    for i, doc in enumerate(documents):
        # Create MinHash signature
        m = MinHash(num_perm=128)
        for word in doc.split():
            m.update(word.encode('utf8'))

        # Check if similar document exists
        result = lsh.query(m)
        if not result:
            lsh.insert(f"doc_{i}", m)
            unique_docs.append(doc)

    return unique_docs

# Reduces dataset size by 30-50%!
```

---

## Part 5: The Data Mix Problem

### What is a Data Mix?

Real models train on multiple sources with careful proportions:

| **Source** | **Proportion** | **Rationale** |
|---|---|---|
| Wikipedia | 10% | High quality, factual |
| Books | 15% | Long-form, coherent |
| Web (Common Crawl) | 50% | Diversity, but noisy |
| GitHub | 10% | Code understanding |
| ArXiv papers | 5% | Scientific reasoning |
| StackExchange | 5% | Q&A format |
| Reddit | 5% | Conversational |

**Why it matters**:
- Too much code → Bad at creative writing
- Too much Wikipedia → Encyclopedic tone
- Too much Reddit → Informal, sometimes toxic

### Creating a Balanced Dataset

```python
def create_balanced_mix(sources, proportions, target_size):
    """
    sources: Dict of {name: list_of_documents}
    proportions: Dict of {name: percentage}
    target_size: Total documents to sample
    """
    mixed_dataset = []

    for source_name, proportion in proportions.items():
        docs = sources[source_name]
        n_samples = int(target_size * proportion)

        # Sample with replacement if needed
        sampled = random.choices(docs, k=n_samples)
        mixed_dataset.extend(sampled)

    # Shuffle the final mix
    random.shuffle(mixed_dataset)
    return mixed_dataset

# Usage
sources = {
    'wikipedia': load_wikipedia(),
    'books': load_books(),
    'web': load_common_crawl(),
}

proportions = {
    'wikipedia': 0.10,
    'books': 0.15,
    'web': 0.75,
}

training_data = create_balanced_mix(sources, proportions, target_size=1_000_000)
```

**Visualization: Data Mix Composition**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Pie chart showing:
# - Wikipedia: 10%
# - Books: 15%
# - Web: 50%
# - GitHub: 10%
# - ArXiv: 5%
# - StackExchange: 5%
# - Reddit: 5%
```

---

## Part 6: Multi-GPU Data Sharding

### The Problem: Data Duplication

With multiple GPUs, naive approach causes duplication:

```python
# ❌ BAD: All GPUs see the same data
loader = DataLoader(dataset, batch_size=32)

# GPU 0 sees: [batch 0, batch 1, batch 2, ...]
# GPU 1 sees: [batch 0, batch 1, batch 2, ...]  # DUPLICATE!
```

### Solution: DistributedSampler

```python
from torch.utils.data.distributed import DistributedSampler

# Each GPU gets a different slice
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total number of GPUs
    rank=rank,  # This GPU's ID (0, 1, 2, ...)
    shuffle=True
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# GPU 0 sees: [batch 0, batch 2, batch 4, ...]
# GPU 1 sees: [batch 1, batch 3, batch 5, ...]
```

**Key insight**: Each GPU's `rank` determines which shards it reads from.

---

## Part 7: Monitoring & Checkpointing

### Track Data Progress

```python
class CheckpointableDataset:
    def __init__(self, shards):
        self.shards = shards
        self.current_shard = 0
        self.samples_seen = 0

    def state_dict(self):
        return {
            'current_shard': self.current_shard,
            'samples_seen': self.samples_seen,
        }

    def load_state_dict(self, state):
        self.current_shard = state['current_shard']
        self.samples_seen = state['samples_seen']

# Save alongside model checkpoint
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'dataset': dataset.state_dict(),
}, 'checkpoint.pt')
```

### Data Loading Speed Metrics

```python
import time

start_time = time.time()
samples = 0

for batch in loader:
    samples += len(batch)

    if samples % 10000 == 0:
        elapsed = time.time() - start_time
        throughput = samples / elapsed
        print(f"Throughput: {throughput:.0f} samples/sec")
```

**Target metrics**:
- **Single GPU**: 1000-5000 samples/sec
- **8 GPUs**: 8000-40000 samples/sec
- If slower → Data loading is the bottleneck!

---

## Part 8: Production Pipeline Example

Putting it all together:

```python
def create_production_pipeline(
    shard_pattern,
    tokenizer,
    batch_size=32,
    shuffle_buffer=10000,
    num_workers=4
):
    """Production-ready data pipeline."""

    dataset = (
        wds.WebDataset(shard_pattern)
        .shuffle(shuffle_buffer)
        .decode()
        .to_tuple("txt")
        .map(lambda x: quality_filter(x[0]))  # Filter
        .select(lambda x: x is not None)      # Remove filtered
        .map(lambda x: tokenizer.encode(x))   # Tokenize
        .batched(batch_size, collation_fn=pad_collate)
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # Already batched
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
    )

# Usage
loader = create_production_pipeline(
    "s3://my-bucket/train-{0000..0999}.tar",
    tokenizer=tokenizer,
)

for epoch in range(num_epochs):
    for batch in loader:
        train_step(batch)
```

---

## Summary

1. **WebDataset** solves streaming large-scale data with tar shards
2. **Quality filtering** removes noise (language, toxicity, repetition)
3. **Deduplication** reduces dataset size by 30-50%
4. **Data mix** balances multiple sources for diverse capabilities
5. **DistributedSampler** ensures each GPU sees different data
6. **Checkpointing** allows resuming from exact data position

**Next Up: L13 – Evaluation Frameworks.** Now that we can train efficiently, how do we know if our model is actually good?

---
