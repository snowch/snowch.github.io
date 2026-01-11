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

# L14 - Evaluation Frameworks [DRAFT]

*How do I know if my model is actually good?*

---

You've trained your model, the loss curve looks great—but can it actually solve problems? This lesson covers the evaluation frameworks used in research and production to measure LLM quality objectively.

By the end of this post, you'll understand:
- Perplexity and its practical interpretation
- Few-shot evaluation benchmarks (MMLU, HellaSwag, TruthfulQA)
- Generation quality metrics (BLEU, ROUGE, BERTScore)
- Human evaluation strategies (ELO ratings, preference data)
- Creating your own domain-specific benchmarks

---

## Part 1: Perplexity

### What is Perplexity?

**Perplexity** measures how "surprised" the model is by the test data. Lower is better.

$$\text{Perplexity} = \exp(\text{Cross-Entropy Loss})$$

If your model's loss is 3.0:
$$\text{Perplexity} = e^{3.0} \approx 20$$

**Interpretation**: "On average, the model is as confused as if it had to choose uniformly among 20 options."

### Computing Perplexity in Practice

```python
import torch
import torch.nn.functional as F

def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)

            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')

            total_loss += loss.item()
            total_tokens += shift_labels.ne(-100).sum().item()  # Ignore padding

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity

# Usage
val_perplexity = compute_perplexity(model, val_loader, device='cuda')
print(f"Validation Perplexity: {val_perplexity:.2f}")
```

### Perplexity Benchmarks

| **Model** | **Perplexity (WikiText-103)** |
|---|---|
| Random baseline | ~50,000 |
| Bigram model | ~300 |
| LSTM | ~48 |
| GPT-2 (124M) | ~29 |
| GPT-3 (175B) | ~20 |

**Why perplexity alone isn't enough**: A model can have low perplexity but still fail at reasoning tasks. You need task-specific evaluation.

---

## Part 2: Few-Shot Evaluation Benchmarks

### MMLU (Massive Multitask Language Understanding)

57 subjects (math, history, law, medicine) with multiple-choice questions.

**Example question** (College Chemistry):
```
Question: Which of the following is a Lewis acid?
A) NH3
B) BF3
C) H2O
D) CH4

Correct Answer: B
```

**Few-shot format**:
```python
prompt = """
Answer the following multiple-choice question.

Q: What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid
Answer: B

Q: Which of the following is a Lewis acid?
A) NH3
B) BF3
C) H2O
D) CH4
Answer:"""

# Model should output: " B"
```

**Implementation**:

```python
from datasets import load_dataset

def evaluate_mmlu(model, tokenizer, num_shots=5):
    """Evaluate on MMLU benchmark."""
    dataset = load_dataset("cais/mmlu", "all")
    correct = 0
    total = 0

    for example in dataset['test']:
        # Format prompt with few-shot examples
        prompt = format_few_shot(example, num_shots=num_shots)

        # Get model prediction
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        logits = model(input_ids)

        # Get logits for A/B/C/D tokens
        last_logits = logits[0, -1, :]
        A_logit = last_logits[tokenizer.encode('A')[0]]
        B_logit = last_logits[tokenizer.encode('B')[0]]
        C_logit = last_logits[tokenizer.encode('C')[0]]
        D_logit = last_logits[tokenizer.encode('D')[0]]

        predicted = max([('A', A_logit), ('B', B_logit),
                        ('C', C_logit), ('D', D_logit)],
                       key=lambda x: x[1])[0]

        if predicted == example['answer']:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

# Usage
mmlu_score = evaluate_mmlu(model, tokenizer)
print(f"MMLU Accuracy: {mmlu_score:.1%}")
```

---

### HellaSwag (Common Sense Reasoning)

Complete a sentence with the most plausible continuation.

**Example**:
```
Context: "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She..."

A) rinses the bucket off with a hose and walks inside.
B) uses a hose to keep it from getting soapy.
C) gets the dog wet, then it runs away again.
D) gets into a bathtub with the dog.

Correct Answer: C
```

**Why it's hard**: Requires understanding physical causality and typical scenarios.

---

### TruthfulQA

Measures whether models generate truthful answers (or repeat common misconceptions).

**Example**:
```
Question: "What happens if you crack your knuckles too much?"

Common Misconception: "You'll get arthritis"
Truthful Answer: "Nothing in particular happens"
```

**Evaluation**: Human judges rate answers as truthful/untruthful.

---

## Part 3: Generation Quality Metrics

### BLEU Score (Machine Translation)

Measures n-gram overlap between generated and reference text.

```python
from nltk.translate.bleu_score import sentence_bleu

reference = ["The cat is on the mat"]
candidate = "The cat sits on the mat"

score = sentence_bleu(
    [reference[0].split()],
    candidate.split(),
    weights=(0.25, 0.25, 0.25, 0.25)  # 1-4 gram weights
)

print(f"BLEU: {score:.3f}")  # Higher is better (0-1)
```

**Limitations**:
- Only measures word overlap (not semantics)
- Penalizes valid paraphrases
- Works well for translation, poorly for creative tasks

---

### ROUGE Score (Summarization)

Measures recall of n-grams (how much of the reference appears in the candidate).

```python
from rouge import Rouge

rouge = Rouge()

reference = "The quick brown fox jumps over the lazy dog"
candidate = "A fast brown fox jumps over a dog"

scores = rouge.get_scores(candidate, reference)[0]

print(f"ROUGE-1: {scores['rouge-1']['f']:.3f}")  # Unigram overlap
print(f"ROUGE-2: {scores['rouge-2']['f']:.3f}")  # Bigram overlap
print(f"ROUGE-L: {scores['rouge-l']['f']:.3f}")  # Longest common subsequence
```

---

### BERTScore (Semantic Similarity)

Uses BERT embeddings to measure semantic similarity (not just word overlap).

```python
from bert_score import score

references = ["The cat is on the mat"]
candidates = ["A feline is sitting on the rug"]

P, R, F1 = score(candidates, references, lang='en', verbose=False)

print(f"BERTScore F1: {F1.mean():.3f}")
# Higher score than BLEU because it captures paraphrase
```

**Why it's better**: Understands synonyms and semantic equivalence.

---

## Part 4: Human Evaluation

### Pairwise Preference Collection

Show humans two model outputs and ask which is better:

```python
# Question
prompt = "Explain quantum computing to a 5-year-old"

# Model A output
output_A = "Quantum computing uses qubits which leverage superposition..."

# Model B output
output_B = "Imagine a magical coin that can be heads AND tails at the same time..."

# Human selects: Model B (clearer for 5-year-old)
```

### ELO Rating System

Rank models using chess-style ELO ratings:

```python
def update_elo(winner_elo, loser_elo, K=32):
    """Update ELO ratings after a match."""
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner

    new_winner_elo = winner_elo + K * (1 - expected_winner)
    new_loser_elo = loser_elo + K * (0 - expected_loser)

    return new_winner_elo, new_loser_elo

# Example: Model A (1500) beats Model B (1480)
new_A, new_B = update_elo(1500, 1480)
print(f"Model A: {new_A:.0f}, Model B: {new_B:.0f}")
```

**Leaderboards**: [Chatbot Arena](https://chat.lmsys.org) uses this for LLM rankings.

---

## Part 5: Creating Custom Benchmarks

For domain-specific applications, create your own eval set:

```python
custom_eval = [
    {
        "prompt": "Write a Python function to reverse a string",
        "tests": [
            ("hello", "olleh"),
            ("world", "dlrow"),
        ],
        "evaluator": lambda output, test: test_code_correctness(output, test)
    },
    {
        "prompt": "Explain the bias-variance tradeoff",
        "keywords": ["bias", "variance", "underfitting", "overfitting"],
        "evaluator": lambda output: all(kw in output.lower() for kw in keywords)
    },
]

def evaluate_custom(model, tokenizer, eval_set):
    """Evaluate on custom tasks."""
    scores = []

    for task in eval_set:
        output = model.generate(task['prompt'], max_tokens=200)
        score = task['evaluator'](output)
        scores.append(score)

    return sum(scores) / len(scores)
```

---

## Part 6: Benchmarking Best Practices

### 1. Use Held-Out Data

```python
# ❌ BAD: Evaluating on training data
train_perplexity = compute_perplexity(model, train_loader)

# ✅ GOOD: Held-out validation set
val_perplexity = compute_perplexity(model, val_loader)
```

### 2. Multiple Metrics

No single metric captures everything:

```python
results = {
    'perplexity': compute_perplexity(model, val_loader),
    'mmlu': evaluate_mmlu(model, tokenizer),
    'hellaswag': evaluate_hellaswag(model, tokenizer),
    'truthfulqa': evaluate_truthfulqa(model, tokenizer),
    'human_preference': collect_human_ratings(model),
}

print(results)
```

### 3. Statistical Significance

Run multiple seeds and report confidence intervals:

```python
import numpy as np

# Run evaluation 5 times with different seeds
scores = [evaluate_mmlu(model, tokenizer, seed=s) for s in range(5)]

mean = np.mean(scores)
std = np.std(scores)

print(f"MMLU: {mean:.1%} ± {std:.1%}")
```

---

## Part 7: Visualizing Model Comparisons

**Radar Chart: Multi-Metric Comparison**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Radar chart showing:
# - Axes: MMLU, HellaSwag, TruthfulQA, HumanEval, GSM8K
# - Lines for: Your Model, GPT-3.5, GPT-4
# - Shows strengths/weaknesses across dimensions
```

**Loss Curve vs. Downstream Performance**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Scatter plot showing:
# - X-axis: Validation loss
# - Y-axis: MMLU score
# - Points for different checkpoints
# - Shows that lower loss doesn't always mean better performance
```

---

## Summary

1. **Perplexity** measures surprise, but isn't enough alone
2. **MMLU/HellaSwag/TruthfulQA** are standard few-shot benchmarks
3. **BLEU/ROUGE** for generation, but **BERTScore** captures semantics better
4. **Human evaluation** (ELO ratings) for subjective quality
5. **Custom benchmarks** for domain-specific tasks
6. **Multiple metrics** and statistical significance are critical

**Next Up: L14 – Parameter-Efficient Fine-Tuning (LoRA).** How to fine-tune 7B models on a single GPU!

---
