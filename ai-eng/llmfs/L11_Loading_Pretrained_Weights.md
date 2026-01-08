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

# L11 - Loading Pretrained Weights & Transfer Learning [DRAFT]

*Starting from GPT-2 instead of random initialization*

---

In [L10 - Fine-tuning](L10_Fine_tuning_and_Chat.md), we learned how to transform a base model into a chat assistant through SFT and RLHF. But we assumed you already had a trained model. In practice, you rarely train from scratch—you start with pretrained weights like GPT-2.

This lesson bridges the gap between "I built a GPT architecture" and "I loaded GPT-2 from HuggingFace and fine-tuned it."

By the end of this post, you'll understand:
- Understanding checkpoint formats (`.pt`, `.safetensors`)
- Loading GPT-2 weights from HuggingFace
- Weight conversion between architectures
- Vocabulary alignment issues
- Frozen layers vs. full fine-tuning strategies

---

## Part 1: Understanding Checkpoint Formats

### PyTorch Native (`.pt`, `.pth`)

The simplest format is PyTorch's native `torch.save()`:

```python
# Saving
torch.save(model.state_dict(), 'model.pt')

# Loading
model = GPT(config)
model.load_state_dict(torch.load('model.pt'))
```

**Structure**: A Python dictionary mapping parameter names to tensors:
```python
{
  'transformer.wte.weight': tensor([...]),  # Token embeddings
  'transformer.wpe.weight': tensor([...]),  # Position embeddings
  'transformer.h.0.attn.c_attn.weight': tensor([...]),  # Layer 0 attention
  ...
}
```

**Pros**: Simple, widely supported
**Cons**: Uses Python's `pickle` (security risk), no metadata, large file size

---

### SafeTensors (`.safetensors`)

The modern standard from HuggingFace:

```python
from safetensors.torch import load_file, save_file

# Saving
save_file(model.state_dict(), 'model.safetensors')

# Loading
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)
```

**Advantages over `.pt`**:
- **Secure**: No arbitrary code execution (unlike pickle)
- **Fast**: Zero-copy loading (mmap support)
- **Metadata**: Can include model config, training info
- **Smaller**: Better compression

**Industry adoption**: All major models now ship with SafeTensors

---

## Part 2: Loading GPT-2 from HuggingFace

### The Architecture Mismatch Problem

GPT-2 uses slightly different naming conventions than our implementation. Here's the mapping:

| **Our Implementation** | **GPT-2 (HuggingFace)** |
|---|---|
| `tok_emb.weight` | `transformer.wte.weight` |
| `pos_emb.weight` | `transformer.wpe.weight` |
| `blocks.0.attn.c_attn.weight` | `transformer.h.0.attn.c_attn.weight` |
| `blocks.0.mlp.fc1.weight` | `transformer.h.0.mlp.c_fc.weight` |
| `ln_f.weight` | `transformer.ln_f.weight` |
| `lm_head.weight` | `lm_head.weight` |

### Weight Conversion Function

```python
def convert_gpt2_weights(hf_state_dict, our_model):
    """Convert HuggingFace GPT-2 weights to our format."""

    mapping = {
        'transformer.wte.weight': 'tok_emb.weight',
        'transformer.wpe.weight': 'pos_emb.weight',
        'transformer.ln_f.weight': 'ln_f.weight',
        'transformer.ln_f.bias': 'ln_f.bias',
        'lm_head.weight': 'lm_head.weight',
    }

    # Handle transformer blocks
    for i in range(len(our_model.blocks)):
        mapping.update({
            f'transformer.h.{i}.attn.c_attn.weight': f'blocks.{i}.attn.c_attn.weight',
            f'transformer.h.{i}.attn.c_attn.bias': f'blocks.{i}.attn.c_attn.bias',
            f'transformer.h.{i}.attn.c_proj.weight': f'blocks.{i}.attn.c_proj.weight',
            f'transformer.h.{i}.attn.c_proj.bias': f'blocks.{i}.attn.c_proj.bias',
            f'transformer.h.{i}.ln_1.weight': f'blocks.{i}.ln_1.weight',
            f'transformer.h.{i}.ln_1.bias': f'blocks.{i}.ln_1.bias',
            f'transformer.h.{i}.mlp.c_fc.weight': f'blocks.{i}.mlp.fc1.weight',
            f'transformer.h.{i}.mlp.c_fc.bias': f'blocks.{i}.mlp.fc1.bias',
            f'transformer.h.{i}.mlp.c_proj.weight': f'blocks.{i}.mlp.fc2.weight',
            f'transformer.h.{i}.mlp.c_proj.bias': f'blocks.{i}.mlp.fc2.bias',
            f'transformer.h.{i}.ln_2.weight': f'blocks.{i}.ln_2.weight',
            f'transformer.h.{i}.ln_2.bias': f'blocks.{i}.ln_2.bias',
        })

    # Convert
    our_state_dict = {}
    for hf_key, hf_tensor in hf_state_dict.items():
        if hf_key in mapping:
            our_key = mapping[hf_key]
            our_state_dict[our_key] = hf_tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")

    return our_state_dict

# Usage
from transformers import GPT2LMHeadModel

# Download GPT-2
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_state_dict = hf_model.state_dict()

# Convert to our format
our_state_dict = convert_gpt2_weights(hf_state_dict, our_model)
our_model.load_state_dict(our_state_dict)
```

---

## Part 3: Vocabulary Alignment

### The Token ID Problem

When loading pretrained weights, your tokenizer must match **exactly**. If GPT-2 assigns ID `50256` to `<|endoftext|>` and your tokenizer assigns it `50257`, the embeddings will be wrong.

```python
# BAD: Different vocabularies
your_tokenizer = YourBPETokenizer(vocab_size=50000)  # ❌
gpt2_model = load_gpt2_weights()  # Expects 50257 tokens

# GOOD: Use the same tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # ✅
```

### Extending the Vocabulary

If you need to add special tokens (e.g., `<|user|>`, `<|assistant|>`):

```python
# Add new tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': ['<|user|>', '<|assistant|>']
})

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# The new tokens are randomly initialized!
# You must fine-tune to learn their representations
```

**Visualization: Before/After Vocabulary Extension**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Diagram showing:
# - Original embedding matrix (50257 x 768)
# - Extended embedding matrix (50259 x 768)
# - Highlighting the 2 new random rows
```

---

## Part 4: Frozen Layers vs. Full Fine-tuning

### Strategy 1: Full Fine-tuning

Update all parameters:

```python
# All parameters require gradients (default)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

**When to use**: Small datasets (overfitting risk is low), or when your task is very different from pretraining.

---

### Strategy 2: Freeze Early Layers

The intuition: early layers learn general features (syntax, basic semantics), while late layers learn task-specific patterns.

```python
# Freeze embeddings and first 6 layers (out of 12)
for param in model.tok_emb.parameters():
    param.requires_grad = False
for param in model.pos_emb.parameters():
    param.requires_grad = False
for i in range(6):
    for param in model.blocks[i].parameters():
        param.requires_grad = False

# Only optimize the last 6 layers + head
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # Can use higher LR since fewer params
)
```

**When to use**: Large pretrained model + small dataset (prevents overfitting).

**Visualization: Which Layers to Freeze?**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Heatmap showing:
# - X-axis: Training steps
# - Y-axis: Layer number (0-11)
# - Color: Magnitude of weight updates
# - Shows that early layers change less than late layers
```

---

### Strategy 3: Gradual Unfreezing

Start by training only the head, then progressively unfreeze layers:

```python
# Phase 1: Train only the head (1 epoch)
for param in model.blocks.parameters():
    param.requires_grad = False

# Phase 2: Unfreeze last 3 layers (1 epoch)
for i in range(9, 12):
    for param in model.blocks[i].parameters():
        param.requires_grad = True

# Phase 3: Unfreeze all (1 epoch)
for param in model.parameters():
    param.requires_grad = True
```

**When to use**: When you have moderate data and want to prevent catastrophic forgetting.

---

## Part 5: Sanity Checks

After loading weights, always verify:

### Check 1: Loss is Reasonable

```python
# Test on a known sentence
test_text = "The quick brown fox jumps over the"
input_ids = tokenizer.encode(test_text, return_tensors='pt')

with torch.no_grad():
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1)
    )

print(f"Loss: {loss.item():.4f}")
# Should be around 3-4 for GPT-2 on English text
# If it's > 10, something is wrong!
```

### Check 2: Generated Text is Coherent

```python
prompt = "Once upon a time"
generated = model.generate(prompt, max_tokens=50)
print(generated)

# Should produce grammatical English
# If it's gibberish, weights didn't load correctly
```

### Check 3: Parameter Counts Match

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Our model: {count_parameters(our_model):,}")
print(f"GPT-2:     {count_parameters(hf_model):,}")
# Should be identical!
```

---

## Part 6: Common Pitfalls

### Pitfall 1: Transposed Weight Matrices

PyTorch Linear layers use `(out_features, in_features)`, but some implementations use `(in_features, out_features)`.

```python
# If shapes don't match, transpose:
if our_weight.shape != hf_weight.shape:
    our_weight = hf_weight.T
```

### Pitfall 2: Missing Bias Terms

Some models have bias, others don't:

```python
# Check if bias exists before loading
if 'bias' in hf_key and hasattr(our_layer, 'bias'):
    our_layer.bias.data = hf_tensor
```

### Pitfall 3: Device Mismatches

```python
# Make sure model and weights are on same device
model = model.to('cuda')
state_dict = {k: v.to('cuda') for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

---

## Summary

1. **SafeTensors** is the modern standard for model checkpoints
2. **Weight conversion** requires careful mapping between naming conventions
3. **Vocabulary must match** exactly, or embeddings will be wrong
4. **Freezing strategies**: Full fine-tuning vs. partial freezing vs. gradual unfreezing
5. **Always sanity check**: Test loss, generation, and parameter counts

**Next Up: L12 – Data Loading Pipelines at Scale.** Now that we can load models, we need to feed them data efficiently!

---
