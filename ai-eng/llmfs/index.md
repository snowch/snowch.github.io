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

# LLM From Scratch: Building GPT From First Principles

*A complete journey from text to tokens to transformers to chat assistants*

---

Welcome to **LLM From Scratch**, a hands-on series that demystifies Large Language Models by building a GPT-style transformer from the ground up.

This isn't just theory—by the end of this series, you'll understand every component that powers systems like ChatGPT, Claude, and Llama. We'll write the code, visualize the math, and connect the dots between research papers and real implementations.

## What You'll Build

Over 10 lessons, we'll construct a complete GPT architecture:

- A **Byte Pair Encoding tokenizer** that handles any text
- **Embedding layers** that give words geometric meaning
- The **self-attention mechanism** that lets words talk to each other
- **Multi-head attention** for parallel relationship processing
- The **causal mask** that prevents cheating during training
- A complete **decoder-only Transformer** (the GPT architecture)
- A **training pipeline** with modern optimizers and learning rate schedules
- **Inference techniques** (temperature, top-p, beam search)
- **Fine-tuning methods** (SFT and RLHF) to create chat assistants

## The Series

### Foundation: Text to Tensors

```{list-table}
:header-rows: 1
:widths: 5 35 60

* - Lesson
  - Title
  - What You'll Learn
* - **L01**
  - [Tokenization From Scratch](L01_Tokenization_From_Scratch.md)
  - *Teaching computers to read* — Build a BPE tokenizer that converts text into token IDs
* - **L02**
  - [Embeddings & Positional Encoding](L02_Embeddings_and_Positional_Encoding.md)
  - *Giving numbers meaning* — Transform tokens into vectors and add positional information
```

### Core Architecture: The Attention Mechanism

```{list-table}
:header-rows: 1
:widths: 5 35 60

* - Lesson
  - Title
  - What You'll Learn
* - **L03**
  - [Self-Attention: The Search Engine of Language](L03_The_Attention_Mechanism.md)
  - *How words talk to each other* — Implement Query-Key-Value attention and understand parallel processing
* - **L04**
  - [Multi-Head Attention](L04_Multi_Head_Attention.md)
  - *Why eight brains are better than one* — Split attention into multiple heads for richer representations
* - **L05**
  - [Normalization & Residuals](L05_Normalization_and_Residuals.md)
  - *The plumbing of deep networks* — Stabilize training with LayerNorm and residual connections
* - **L06**
  - [The Causal Mask](L06_The_Causal_Mask.md)
  - *How to stop cheating* — Prevent the model from seeing future tokens during training
```

### Assembly and Training

```{list-table}
:header-rows: 1
:widths: 5 35 60

* - Lesson
  - Title
  - What You'll Learn
* - **L07**
  - [Assembling the GPT](L07_Assembling_the_GPT.md)
  - *The grand finale* — Stack all components into a complete decoder-only Transformer
* - **L08**
  - [Training the LLM](L08_Training_the_LLM.md)
  - *Learning to speak* — Implement the training loop with AdamW, learning rate schedules, and gradient accumulation
```

### Inference and Fine-tuning

```{list-table}
:header-rows: 1
:widths: 5 35 60

* - Lesson
  - Title
  - What You'll Learn
* - **L09**
  - [Inference & Sampling](L09_Inference_and_Sampling.md)
  - *Controlling the creativity* — Generate text with temperature, top-p sampling, and beam search
* - **L10**
  - [Fine-tuning: From Completion to Conversation](L10_Fine_tuning_and_Chat.md)
  - *Transforming into a chat assistant* — Apply SFT and RLHF to create helpful, harmless AI assistants
```

## Prerequisites

This series assumes:
- **Python fundamentals** (functions, classes, basic numpy)
- **High school math** (algebra, basic calculus concepts)
- **Neural network basics** (what a layer is, forward/backward pass)

We'll explain everything else from scratch, including:
- Matrix operations and why they matter
- Backpropagation and gradient flow
- PyTorch fundamentals as we go

## Philosophy

**Code First, Math Second:** Every concept is implemented in PyTorch before diving into equations.

**Visual Intuition:** We use diagrams, plots, and animations to build intuition before formalism.

**No Magic:** We demystify research papers by showing the gap between "attention is all you need" and "here's how it actually works."

**Production-Aware:** We explain not just what works, but why certain choices (Pre-Norm vs Post-Norm, AdamW vs SGD) became industry standard.

## How to Use This Series

1. **Sequential Reading:** Lessons build on each other—start from L01
2. **Run the Code:** Each lesson is an executable Jupyter notebook
3. **Pause and Experiment:** Modify parameters, break things, rebuild understanding
4. **Skip What You Know:** Familiar with embeddings? Jump ahead (but skim for our specific approach)

## Ready to Begin?

Let's start with the first step in any LLM pipeline: teaching computers to read.

**Next: [L01 - Tokenization From Scratch →](L01_Tokenization_From_Scratch.md)**

---

## Additional Resources

- **Research Papers:** Each lesson links to relevant papers (Attention is All You Need, GPT-2, etc.)
- **PyTorch Docs:** We reference official documentation for implementation details
- **Further Reading:** Suggested deep dives for those who want more mathematical rigor

*This series is designed for engineers and researchers who want to understand LLMs deeply enough to modify, debug, and innovate on them.*
