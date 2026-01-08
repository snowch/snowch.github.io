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

# Neural Networks From Scratch: Understanding Deep Learning From First Principles

*From single neurons to flexible architectures—build intuition before abstractions*

---

Welcome to **Neural Networks From Scratch**, a hands-on series that demystifies deep learning by building everything from the ground up.

This series takes a different approach than most tutorials: we start with **visual intuition** (how does a neuron detect an edge?), then gradually introduce the math and abstractions. By the time we reach frameworks like PyTorch, you'll understand what's happening under the hood.

## What You'll Build

Starting from basic pattern matching with a single neuron, we'll build up to flexible multi-layer networks:

- A **single neuron edge detector** that shows how weights encode patterns
- A complete **training pipeline** with backpropagation from scratch
- A **flexible neural network class** that handles arbitrary architectures
- The **PyTorch equivalent** of everything we built, ready for production
- **Architecture patterns** (embeddings, residuals, LayerNorm) that bridge to Transformers

## The Series

### Core Foundations: From Neurons to Networks

```{list-table}
:header-rows: 1
:widths: 5 40 55

* - Lesson
  - Title
  - What You'll Learn
* - **NN01**
  - [Edge Detection Intuition: A Single Neuron as Pattern Matching](nn_edge_detector_blog.md)
  - *Pattern matching fundamentals* — Understand how weights, ReLU, and bias work together to detect patterns
* - **NN02**
  - [Training an Edge-Detection Neural Network from Scratch](nn_tutorial_blog.md)
  - *How networks learn* — Implement forward pass, loss functions, backpropagation, and gradient descent
* - **NN03**
  - [Building a Flexible Neural Network from Scratch](nn_flexible_network_blog.md)
  - *Generalize your architecture* — Create a Layer class and stack them for arbitrary network depths
* - **NN04**
  - [PyTorch Basics: Rebuilding the Flexible MLP](nn_pytorch_basics.md)
  - *Transition to industry tools* — Rebuild your network using nn.Module, nn.Sequential, and standard training loops
```

### Bridge to Modern Architectures

```{list-table}
:header-rows: 1
:widths: 5 40 55

* - Lesson
  - Title
  - What You'll Learn
* - **NA01**
  - [Architecture Patterns: Bridge to Transformers](nn_architecture_patterns.md)
  - *Modern patterns explained* — Master sequence tensors (B,T,C), embeddings, residual connections, and LayerNorm
```

## Learning Path

The series follows a carefully designed progression:

1. **Start Visual** (NN01): See how a single neuron detects edges before worrying about derivatives
2. **Learn Training** (NN02): Understand the full training loop with a concrete example
3. **Generalize** (NN03): Abstract away hardcoded layers into a flexible system
4. **Go Professional** (NN04): Learn the PyTorch equivalents of everything you built
5. **Bridge Forward** (NA01): Connect to modern architectures (Transformers, ResNets)

## Prerequisites

This series assumes:
- **Basic Python** (functions, classes, simple loops)
- **High school math** (multiplication, addition, basic algebra)
- **Curiosity** about how AI actually works

We'll explain everything else, including:
- What gradients are and why they matter
- How backpropagation flows through layers
- Why certain activation functions work better than others

## Philosophy

**Intuition First:** Every concept starts with a visual explanation or concrete example.

**Build, Then Name:** We implement ideas before introducing formal terminology.

**No Black Boxes:** We write every line of the forward and backward pass by hand before using frameworks.

**Bridge to Production:** After understanding the fundamentals, we show the PyTorch equivalents you'll use in practice.

**Question Everything:** We explain not just "how" but "why"—why ReLU over sigmoid? Why cross-entropy loss? Why LayerNorm in Transformers?

## What Makes This Series Different

Most tutorials either:
- Show you TensorFlow/PyTorch code without explaining the math, OR
- Dive into calculus and linear algebra without building intuition

We do both: start with intuition (edge detection!), build it from scratch (pure NumPy), understand the math (backpropagation), then transition to frameworks (PyTorch).

## Real-World Examples

- **NN01-02:** Edge detection (the fundamental operation in CNNs)
- **NN03:** MNIST-ready architecture (extendable to real datasets)
- **NN04:** Production-style training loops
- **NA01:** Transformer building blocks (preparing for LLMs)

## How to Use This Series

1. **Start from NN01:** Each lesson builds on the previous one
2. **Run the Code:** These are executable Jupyter notebooks—experiment!
3. **Pause and Modify:** Change weights, learning rates, architectures
4. **Connect Forward:** After NN04, you're ready for CNNs, RNNs, or the [LLM From Scratch](../llmfs/index.md) series

## Ready to Begin?

Let's start by seeing how a single neuron can detect edges in images.

**Next: [NN01 - Edge Detection Intuition →](nn_edge_detector_blog.md)**

---

## Where to Go Next

After completing this series:

- **For Computer Vision:** Learn CNNs (filters, pooling, ResNet)
- **For Sequences:** Learn RNNs and LSTMs (time-series, text)
- **For Language Models:** Check out our [LLM From Scratch](../llmfs/index.md) series (Transformers, attention, GPT)

## Additional Resources

- **Research Papers:** Key papers referenced in context (BatchNorm, LayerNorm, ResNet)
- **PyTorch Docs:** Official documentation for production implementations
- **Visual Explanations:** Diagrams and animations throughout

*This series is designed for anyone who wants to understand neural networks deeply—whether you're a beginner trying to break into AI, or an experienced practitioner who wants to understand what's really happening under the hood.*
