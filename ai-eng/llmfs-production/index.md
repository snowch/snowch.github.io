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

# LLM From Scratch: Production Techniques

*Bridge the gap between research code and production deployment*

---

Welcome to the **Production Techniques** series, where we take the GPT architecture you built in the [core series](../llmfs/index.md) and make it production-ready.

You've learned how to build and train a Transformer from scratch. Now it's time to tackle the challenges that arise when moving from toy datasets to real-world deployment: loading massive pretrained models, processing terabytes of data, evaluating model quality, and fine-tuning efficiently on consumer hardware.

## What You'll Learn

This series covers the essential techniques used in production LLM workflows:

- **Loading pretrained weights** from HuggingFace and other sources
- **Building data pipelines** that handle terabytes of training data
- **Evaluating models** with industry-standard benchmarks
- **Fine-tuning efficiently** with LoRA on consumer GPUs
- **Training faster** with mixed precision (FP16/BF16)

## Prerequisites

You should have completed (or be familiar with):
- The core [LLM From Scratch](../llmfs/index.md) series (L01-L11)
- Basic PyTorch training loops
- Understanding of transformer architecture fundamentals

## The Series

```{list-table}
:header-rows: 1
:widths: 5 40 55

* - Lesson
  - Title
  - What You'll Learn
* - **L12**
  - [Loading Pretrained Weights & Transfer Learning](L12_Loading_Pretrained_Weights.md)
  - *Starting from GPT-2 instead of random* — Load HuggingFace weights, handle vocabulary mismatches, choose freezing strategies
* - **L13**
  - [Data Loading Pipelines at Scale](L13_Data_Loading_Pipelines.md)
  - *From toy datasets to production* — Stream terabytes with WebDataset, quality filtering, deduplication, data mixing
* - **L14**
  - [Evaluation Frameworks](L14_Evaluation_Frameworks.md)
  - *How do I know if my model is good?* — Perplexity, MMLU, HellaSwag, TruthfulQA, and custom benchmarks
* - **L15**
  - [Parameter-Efficient Fine-Tuning (LoRA)](L15_LoRA_PEFT.md)
  - *Fine-tune 7B models on a single GPU* — Low-rank adaptation mathematics, QLoRA, adapter swapping
* - **L16**
  - [Mixed Precision Training](L16_Mixed_Precision_Training.md)
  - *Train 2-3× faster with half the memory* — FP16, BF16, gradient scaling, PyTorch AMP
```

## Why These Topics Matter

### The Reality Gap

There's a massive gap between:
- Training a 124M parameter model on a toy dataset → **Academic exercise**
- Fine-tuning a 7B model on real data for production → **Real-world challenge**

This series bridges that gap with practical techniques used at companies like OpenAI, Anthropic, and HuggingFace.

### Real-World Constraints

Production LLM work faces constraints that research doesn't:

| **Constraint** | **Solution in This Series** |
|---|---|
| Can't train from scratch (too expensive) | L12: Load pretrained weights |
| Can't fit model in GPU memory | L15: LoRA (14 GB → 4 GB) |
| Can't wait weeks for training | L16: Mixed precision (2-3× speedup) |
| Can't trust arbitrary benchmarks | L14: Comprehensive evaluation |
| Can't load all data into RAM | L13: Streaming data pipelines |

## Learning Path

**Sequential approach (recommended)**:
1. **L12**: Start here if you want to fine-tune existing models
2. **L13**: Learn data engineering for real training runs
3. **L14**: Understand how to measure success
4. **L15**: Make fine-tuning practical on limited hardware
5. **L16**: Speed up everything with mixed precision

**Jump-in approach**:
- Need to fine-tune **now**? → Start with L15 (LoRA)
- Building a data pipeline? → Jump to L13
- Choosing between models? → Go to L14 (Evaluation)

## What's Next

After completing this series:
- **[Scaling & Optimization](../llmfs-scaling/index.md)**: Attention optimizations, model parallelism, long contexts, quantization, deployment
- **Real projects**: Fine-tune models for your domain, build production pipelines

## Philosophy

**Production-First**: Every technique is chosen because it's used in real production systems, not just because it's theoretically interesting.

**Resource-Aware**: We optimize for consumer hardware (24GB GPUs) and cloud budgets, not unlimited research clusters.

**Measured Impact**: We quantify improvements (2× faster, 4× less memory) instead of vague claims.

**End-to-End**: From loading weights to evaluation, we cover the full workflow.

## Ready to Begin?

Let's start by learning how to load pretrained weights like GPT-2 and fine-tune them for your tasks.

**Next: [L12 - Loading Pretrained Weights & Transfer Learning →](L12_Loading_Pretrained_Weights.md)**

---

*This series assumes you've completed the [core LLM From Scratch series](../llmfs/index.md). If you're new, start there to understand the fundamentals of transformers, attention, and training.*
