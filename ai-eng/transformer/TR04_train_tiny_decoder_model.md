---
title: "TR04 — Train a Tiny Decoder Model (Next-Token Dataset + Generation)"
description: "Train a small decoder-only transformer end-to-end in PyTorch: dataset prep, teacher forcing, cross-entropy loss, and a clean generate() loop."
keywords:
  - training loop
  - teacher forcing
  - next-token prediction
  - gpt from scratch
  - text generation
  - pytorch
---

# TR04 — Train a Tiny Decoder Model (Next-Token Dataset + Generation)

Now we train the model from TR03 on a tiny text corpus.

This is **Scope A**: we rely on PyTorch for autograd, but the model is implemented from scratch (attention, blocks, stack, generation).

---

## 1) A tiny tokenizer (toy, but enough to learn)

For learning, start with a **character-level** tokenizer. It’s simple, debuggable, and shows the mechanics clearly.

```python
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

For production, you’d switch to a BPE tokenizer — but training mechanics are identical.

---

## 2) Dataset: fixed-length chunks for next-token prediction

Given a long stream of token IDs, we sample windows of length `block_size`:

- input: `x = ids[i : i+T]`
- target: `y = ids[i+1 : i+T+1]`

```python
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

## 3) Training loop (teacher forcing + cross-entropy)

We feed `x` (true previous tokens) and compute logits for each position.
Then we compute cross-entropy against `y`.

```python
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)  # (B, T)

        logits = model(x)                  # (B, T, vocab)
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B*T, V),
            y.view(B*T)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))
```

---

## 4) A clean generate() loop (prefill + decode)

Generation is autoregressive:

1. Start with a prompt (token IDs)
2. Repeatedly:
   - run model
   - take the last position logits
   - sample the next token
   - append it

```python
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

This works, but it’s not optimal.

**Why?** Each step recomputes work for all previous tokens.  
That’s exactly why KV cache exists (see your IN01 post).

---

## 5) Put it together (minimal runnable script)

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

## 6) Where to go next (systems)

At this point you have the full learning loop:

- tokenization → embeddings
- transformer blocks
- next-token loss
- autoregressive generation

Now you’re ready for *systems* posts:

- KV cache (your IN01)
- continuous batching
- chunked prefill
- paged KV memory management

A great follow-up is: **“Speed up generate() with KV cache”** — which turns TR04 directly into IN01.
