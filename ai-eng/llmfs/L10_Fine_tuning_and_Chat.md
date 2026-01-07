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

# L10 - Fine-tuning: From Completion to Conversation [DRAFT]

*Transforming a base LLM into a helpful Chat Assistant*

---

We have reached the finish line. In [L09 Inference and Sampling](L09_Inference_and_Sampling.md), we saw that our model can complete text. However, if you ask a "Base" model: *"What is the capital of France?"*, it might respond with: *"...and what is the capital of Germany?"*

This final post concludes the journey by explaining how we transition from a model that simply "predicts the next word" to a model that can actually follow instructions and act as a helpful assistant.

Why? Because it thinks it's looking at a geography quiz, not a conversation. It is a completion engine, not an assistant. To fix this, we need **Fine-tuning**.

By the end of this post, you'll understand:
- The difference between **Pre-training** and **Fine-tuning**.
- **SFT (Supervised Fine-Tuning):** Teaching the model to follow instructions.
- **RLHF (Reinforcement Learning from Human Feedback):** Aligning the model with human values.

```{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Two Stages of Training

Building a Chatbot is a two-step process:

1. **Pre-training (The Library):** The model reads the whole internet. It learns grammar, facts, and reasoning. This creates the **Base Model**.
2. **Fine-tuning (The Training):** We show the model specific examples of "Question -> Answer" pairs. This creates the **Chat/Instruct Model**.

---

## Part 2: Supervised Fine-Tuning (SFT)

During SFT, we use a smaller, high-quality dataset of dialogues. We format them using special tokens so the model knows who is talking.

**Example Template:**

```text
<|user|>: What is 2+2?
<|assistant|>: 2+2 is 4.

```

We train the model using the exact same "Next Token Prediction" loss we used in L08, but only on the **assistant's** parts. We tell the model: "When you see the `<|user|>` block, you should predict the `<|assistant|>` block."

---

## Part 3: RLHF - Learning from Human Taste

Sometimes, "correct" isn't enough. We want the model to be polite, safe, and helpful.

In **Reinforcement Learning from Human Feedback (RLHF)**:

1. Humans rank different model responses (Response A is better than Response B).
2. We train a **Reward Model** to predict what humans like.
3. We use that Reward Model to nudge the LLM toward "high-reward" answers.

---

## Part 4: Visualizing the Shift in Behavior

Let's see how the probability distribution shifts after fine-tuning.

```{code-cell} ipython3
:tags: [remove-input]

labels = ["Continuation", "Answer", "Clarification", "Gibberish"]
base_probs = [0.6, 0.2, 0.1, 0.1]
chat_probs = [0.05, 0.85, 0.05, 0.05]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, base_probs, width, label='Base Model', color='lightgray')
ax.bar(x + width/2, chat_probs, width, label='Chat Model', color='skyblue')

ax.set_ylabel('Probability')
ax.set_title('Probability Distribution: "What is the capital of France?"')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

```

---

## Summary: We Built an LLM!

Over these 10 blogs, we have traveled from raw text to a functional, instruction-following AI.

* **The Foundation:** Math, Vectors, and PyTorch.
* **The Core:** Multi-Head Attention and Transformer Blocks.
* **The Polish:** Masking, Sampling, and Fine-tuning.

You now have the blueprints to build, train, and talk to your own neural networks from scratch.

---

**What's next for you?** You could take the code from [L07 Assembling the GPT](L07_Assembling_the_GPT.md) and train it on a small dataset like **TinyShakespeare** or **OpenWebText**. Or, you could explore **Quantization** to make these models run faster on your laptop.

Thank you for following this "From Scratch" series!

---
