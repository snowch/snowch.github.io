---
title: "TR04 — Train a Tiny Decoder Model"
description: "Train a small decoder-only transformer end-to-end in PyTorch: dataset prep, teacher forcing, cross-entropy loss, and a baseline generate loop."
keywords:
  - training loop
  - teacher forcing
  - next-token prediction
  - gpt from scratch
  - text generation
  - pytorch
---

# TR04 — Train a Tiny Decoder Model

This post connects everything into a complete loop:

1. build a tokenizer  
2. build a next-token dataset  
3. train the decoder-only model  
4. generate text from a prompt  

The result is a small, end-to-end baseline that is easy to understand and extend.

---

## What you will build

By the end, you will have:

- a runnable training loop (teacher forcing + cross-entropy)
- a baseline `generate()` function that produces text autoregressively
- the mental model for why inference later benefits from KV caching

---

## Part 1: Tokenization (toy, but perfect for learning)

Real LLMs use BPE-style tokenizers. For learning, a **character-level tokenizer** is ideal:

- completely deterministic
- easy to debug
- no external dependencies

```{code-cell} ipython3
:tags: [remove-input]

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
```

---

## Part 2: Dataset for next-token prediction

Given a long stream of token IDs, sample windows of length `T`:

- input:  `x = ids[i : i+T]`
- target: `y = ids[i+1 : i+T+1]`

```{code-cell} ipython3
:tags: [remove-input]

import torch

class NextTokenDataset(torch.utils.data.Dataset):
    def __init__(self, ids, block_size: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size - 1

    def __getitem__(self, i):
        x = self.ids[i : i + self.block_size]
        y = self.ids[i + 1 : i + 1 + self.block_size]
        return x, y
```

---

## Part 3: Training loop (teacher forcing + cross-entropy)

Training steps:

1. forward pass: `logits = model(x)`  
2. compute cross-entropy loss against `y`  
3. backprop + optimizer step  

```{code-cell} ipython3
:tags: [remove-input]

import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)  # (B, T)

        logits = model(x)                  # (B, T, vocab)
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, V),
            y.view(B * T),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))
```

---

## Part 4: A baseline generate loop (autoregressive decoding)

Autoregressive decoding:

1. start with a prompt token sequence  
2. repeatedly:
   - run the model  
   - take the last position logits  
   - sample the next token  
   - append it to the sequence  

```{code-cell} ipython3
:tags: [remove-input]

import torch

@torch.no_grad()
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0):
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(idx)                   # (B, T, vocab)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

        idx = torch.cat([idx, next_id], dim=1)

    return idx
```

This baseline is correct and easy to understand.

The performance limitation is also clear: each decode step recomputes work for all earlier tokens.  
Caching K/V is the standard optimization.

---

## Part 5: Put it together (minimal runnable script)

The script below shows how the components fit together in practice.  
It is written as a plain code block so it does not require a dataset to execute during a docs build.

```python
import torch
from torch.utils.data import DataLoader

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = open("tiny_corpus.txt", "r", encoding="utf-8").read()
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    block_size = 128
    ds = NextTokenDataset(ids, block_size)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    model = TinyGPT(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_len=block_size,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(5):
        loss = train_one_epoch(model, loader, opt, device)
        print(f"epoch {epoch} loss {loss:.4f}")

        prompt = "I love "
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(device)
        out = generate(model, idx, max_new_tokens=200, temperature=1.0)
        print(tok.decode(out[0].tolist()))
        print("-" * 60)

if __name__ == "__main__":
    main()
```

---

## Summary

You now have:

- tokenization
- a next-token dataset
- a training loop
- a generation loop

From here, the next step is inference performance: caching K/V and improving batching and memory behavior.
