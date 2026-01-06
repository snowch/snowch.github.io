---
title: "NN04 — PyTorch Basics: Rebuilding the Flexible MLP"
description: "Rebuild the NN03 flexible network in PyTorch, introduce nn.Module, nn.Sequential, loss functions, and the standard training loop."
keywords:
  - neural networks from scratch
  - pytorch
  - mlp
  - training loop
---

# NN04 — PyTorch Basics: Rebuilding the Flexible MLP

In NN03 we built a flexible MLP from scratch with NumPy. Now we’ll rebuild the
**same network** in PyTorch so future chapters (CNNs, RNNs, LSTMs) can reuse a
standard training loop and familiar building blocks.

---

## Learning goals

By the end of this chapter you will be able to:

1. Define a flexible MLP using `nn.Module` and `nn.Sequential`
2. Use `CrossEntropyLoss` for multi-class classification
3. Train with a simple PyTorch training loop
4. Recognize how this maps directly to CNNs/RNNs/LSTMs later

---

## 1) The same architecture, now in PyTorch

The pattern is identical to our NumPy version: build a list of layers and wrap
them in a module.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

---

## 2) Data: reuse the edge detector dataset

We can reuse the same NumPy dataset from NN03, then convert to tensors.

```python
import numpy as np

X_train, y_train = make_data(200)
X_test, y_test = make_data(50)

xb = torch.tensor(X_train, dtype=torch.float32)
yb = torch.tensor(y_train.argmax(axis=1), dtype=torch.long)
```

---

## 3) Training loop (standard PyTorch)

```python
model = MLP([25, 16, 8, 2])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    logits = model(xb)
    loss = criterion(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 4) Why this unlocks future chapters

Once we have this PyTorch baseline:

- **CNNs** are just `nn.Conv2d` + pooling
- **RNNs/LSTMs** are built-in modules (`nn.RNN`, `nn.LSTM`)
- **Training loops stay the same**, so we can focus on architecture intuition

In NN05 we’ll shift to architecture patterns that bridge into transformers.
