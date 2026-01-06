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

# The Birthday Paradox

*Why you only need 23 people for a 50% chance of a shared birthday*

---

Here's a counterintuitive fact: in a room of just 23 people, there's a better than 50% chance that two of them share a birthday. With 70 people, it's virtually certain (99.9%). Let's build up to understanding why, starting from the simplest case.

+++

## The Clever Trick: Complement Counting

Directly counting all the ways people could share birthdays is messy — you'd have to consider pairs sharing, triplets sharing, different combinations... Instead, we flip the problem:

$$P(\text{at least 2 share a birthday}) = 1 - P(\text{no one shares a birthday})$$

Calculating "no shared birthdays" is surprisingly elegant. We just need everyone to pick *different* days. Let's build this up person by person.

+++

---

## Case 1: Just One Person

With only one person, there's no one to share a birthday with!

$$P(\text{no shared birthday}) = 1$$

$$P(\text{at least 2 share}) = 0$$

They can pick any of the 365 days — all options are "safe."

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
ax.set_title('1 Person: Any birthday works', fontsize=14, fontweight='bold', pad=10)

blue = '#3b82f6'
green = '#10b981'

# Start node
ax.add_patch(plt.Circle((50, 80), 4, color=blue, zorder=3))
ax.text(50, 80, 'Start', ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=4)

# Person 1 choices
positions = [(15, 'Jan 1'), (30, 'Jan 2'), (50, '...'), (70, 'Dec 30'), (85, 'Dec 31')]
for x, label in positions:
    ax.plot([50, x], [76, 45], color=blue, linewidth=1.5)
    if label == '...':
        ax.text(x, 40, '• • •', ha='center', va='center', fontsize=14, color='#6b7280')
    else:
        ax.add_patch(plt.Circle((x, 40), 3.5, color=green, zorder=3))
        ax.text(x, 40, label, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)

ax.text(50, 18, 'All 365 days are safe → P(no conflict) = 365/365 = 1', 
        ha='center', va='center', fontsize=12, style='italic', color='#374151')

plt.tight_layout()
plt.show()
```

---

## Case 2: Two People

Now it gets interesting. Person 1 picks any day. Person 2 must pick a *different* day to avoid a collision.

- **Person 1:** 365 choices out of 365 → probability = 365/365
- **Person 2:** 364 safe choices out of 365 → probability = 364/365

$$P(\text{no shared birthday}) = \frac{365}{365} \times \frac{364}{365} = \frac{364}{365} \approx 0.9973$$

$$P(\text{at least 2 share}) = 1 - 0.9973 = 0.27\%$$

```{code-cell}
:tags: [remove-input]

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
ax.set_title('2 People: Person 2 must avoid Person 1\'s birthday', fontsize=14, fontweight='bold', pad=10)

blue = '#3b82f6'
green = '#10b981'
purple = '#8b5cf6'
red = '#ef4444'
light_red = '#fecaca'

# Labels
ax.text(3, 82, 'Person 1', fontsize=11, fontweight='bold', color='#374151')
ax.text(3, 42, 'Person 2', fontsize=11, fontweight='bold', color='#374151')

# Start
ax.add_patch(plt.Circle((50, 92), 3, color=blue, zorder=3))
ax.text(50, 92, 'Start', ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=4)

# Person 1: picks Jan 1 (as example)
ax.plot([50, 25], [89, 72], color=blue, linewidth=2)
ax.add_patch(plt.Circle((25, 68), 3.5, color=green, zorder=3))
ax.text(25, 68, 'Jan 1', ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=4)
ax.text(35, 82, '1/365', color=blue, fontsize=10, fontweight='bold')

# Show other P1 options faded
for x in [50, 75]:
    ax.plot([50, x], [89, 72], color='#d1d5db', linewidth=1, linestyle='--')
ax.text(50, 68, '...', ha='center', va='center', fontsize=12, color='#9ca3af')
ax.text(75, 68, '...', ha='center', va='center', fontsize=12, color='#9ca3af')

# Person 2 branches from Jan 1
p2_data = [(12, 'Jan 2', False), (22, 'Jan 3', False), (32, '...', None), 
           (42, 'Dec 31', False), (55, 'Jan 1', True)]

for x, label, is_bad in p2_data:
    color = red if is_bad else green
    style = '--' if is_bad else '-'
    ax.plot([25, x], [64.5, 45], color=color, linewidth=1.5, linestyle=style)
    
    if is_bad is None:
        ax.text(x, 40, '•••', ha='center', va='center', fontsize=11, color='#6b7280')
    elif is_bad:
        ax.add_patch(plt.Circle((x, 40), 3, color=light_red, ec=red, linewidth=2, zorder=3))
        ax.text(x, 40, label, ha='center', va='center', color=red, fontsize=8, fontweight='bold', zorder=4)
        ax.plot([x-2, x+2], [38, 42], color=red, linewidth=2, zorder=5)
        ax.plot([x-2, x+2], [42, 38], color=red, linewidth=2, zorder=5)
    else:
        ax.add_patch(plt.Circle((x, 40), 3, color=purple, zorder=3))
        ax.text(x, 40, label, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)

ax.text(20, 55, '364/365', color=green, fontsize=10, fontweight='bold')
ax.text(52, 55, '1/365 ✗', color=red, fontsize=9)

# Result box
bbox = mpatches.FancyBboxPatch((62, 30), 35, 18, boxstyle='round,pad=0.02', 
                                facecolor='#f0fdf4', edgecolor='#22c55e', linewidth=2)
ax.add_patch(bbox)
ax.text(79.5, 43, 'P(no shared) = 364/365', ha='center', fontsize=10, fontweight='bold', color='#166534')
ax.text(79.5, 36, '= 99.73%', ha='center', fontsize=11, fontweight='bold', color='#166534')

plt.tight_layout()
plt.show()
```

---

## Case 3: Three People

Person 3 joins. Now they must avoid *two* birthdays.

- **Person 1:** 365/365
- **Person 2:** 364/365 (avoid 1 day)
- **Person 3:** 363/365 (avoid 2 days)

$$P(\text{no shared birthday}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \approx 0.9918$$

$$P(\text{at least 2 share}) = 1 - 0.9918 = 0.82\%$$

```{code-cell}
:tags: [remove-input]

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
ax.set_title('3 People: Each person has fewer safe options', fontsize=14, fontweight='bold', pad=10)

blue = '#3b82f6'
green = '#10b981'
purple = '#8b5cf6'
orange = '#f59e0b'
red = '#ef4444'
light_red = '#fecaca'

# Labels
ax.text(2, 85, 'Person 1', fontsize=11, fontweight='bold', color='#374151')
ax.text(2, 55, 'Person 2', fontsize=11, fontweight='bold', color='#374151')
ax.text(2, 22, 'Person 3', fontsize=11, fontweight='bold', color='#374151')

# Start
ax.add_patch(plt.Circle((50, 95), 2.5, color=blue, zorder=3))
ax.text(50, 95, 'Start', ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=4)

# Person 1
level1_y = 82
for x, label, sub in [(15, 'Jan 1', '(Day 1)'), (30, 'Jan 2', '(Day 2)'), 
                       (70, 'Dec 30', '(Day 364)'), (85, 'Dec 31', '(Day 365)')]:
    ax.plot([50, x], [92.5, level1_y + 3], color=blue, linewidth=1.5)
    ax.add_patch(plt.Circle((x, level1_y), 3, color=green, zorder=3))
    ax.text(x, level1_y + 0.3, label, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)
    ax.text(x, level1_y - 1, sub, ha='center', va='center', color='white', fontsize=6, zorder=4)
ax.plot([50, 50], [92.5, level1_y + 3], color=blue, linewidth=1.5)
ax.text(50, level1_y, '• • •', ha='center', va='center', fontsize=14, color='#6b7280')

ax.text(32, 89, '1/365', color=blue, fontsize=9, fontweight='bold')

bbox1 = mpatches.FancyBboxPatch((88, 79), 10, 6, boxstyle='round,pad=0.02', facecolor='#dbeafe', edgecolor=blue)
ax.add_patch(bbox1)
ax.text(93, 82, '365', ha='center', fontsize=9, fontweight='bold', color='#1e40af')
ax.text(93, 80, 'options', ha='center', fontsize=7, color='#1e40af')

# Person 2 (from Jan 1)
level2_y = 52
p2_data = [(8, 'Jan 2', False), (16, 'Jan 3', False), (24, '...', None), (32, 'Dec 31', False), (42, 'Jan 1', True)]
for x, label, is_bad in p2_data:
    color = red if is_bad else green
    style = '--' if is_bad else '-'
    ax.plot([15, x], [level1_y - 3, level2_y + 2.5], color=color, linewidth=1.5, linestyle=style)
    if is_bad is None:
        ax.text(x, level2_y, '•••', ha='center', va='center', fontsize=10, color='#6b7280')
    elif is_bad:
        ax.add_patch(plt.Circle((x, level2_y), 2.5, color=light_red, ec=red, linewidth=2, zorder=3))
        ax.text(x, level2_y, label, ha='center', va='center', color=red, fontsize=7, fontweight='bold', zorder=4)
        ax.plot([x-1.5, x+1.5], [level2_y-1.5, level2_y+1.5], color=red, linewidth=2, zorder=5)
        ax.plot([x-1.5, x+1.5], [level2_y+1.5, level2_y-1.5], color=red, linewidth=2, zorder=5)
    else:
        ax.add_patch(plt.Circle((x, level2_y), 2.5, color=purple, zorder=3))
        ax.text(x, level2_y, label, ha='center', va='center', color='white', fontsize=7, fontweight='bold', zorder=4)

ax.text(10, 67, '364/365', color=green, fontsize=9, fontweight='bold')
ax.text(40, 67, '1/365 ✗', color=red, fontsize=8)

bbox2 = mpatches.FancyBboxPatch((46, 49), 12, 6, boxstyle='round,pad=0.02', facecolor='#d1fae5', edgecolor=green)
ax.add_patch(bbox2)
ax.text(52, 52, '364 safe', ha='center', fontsize=8, fontweight='bold', color='#065f46')
ax.text(52, 50, 'options', ha='center', fontsize=7, color='#065f46')

# Person 3 (from Jan 2)
level3_y = 19
p3_data = [(3, 'Jan 3', False), (9, 'Jan 4', False), (15, '...', None), 
           (21, 'Dec 31', False), (28, 'Jan 1', True), (35, 'Jan 2', True)]
for x, label, is_bad in p3_data:
    color = red if is_bad else purple
    style = '--' if is_bad else '-'
    ax.plot([8, x], [level2_y - 2.5, level3_y + 2], color=color, linewidth=1.5, linestyle=style)
    if is_bad is None:
        ax.text(x, level3_y, '•••', ha='center', va='center', fontsize=9, color='#6b7280')
    elif is_bad:
        ax.add_patch(plt.Circle((x, level3_y), 2, color=light_red, ec=red, linewidth=2, zorder=3))
        ax.text(x, level3_y, label, ha='center', va='center', color=red, fontsize=6, fontweight='bold', zorder=4)
        ax.plot([x-1.2, x+1.2], [level3_y-1.2, level3_y+1.2], color=red, linewidth=2, zorder=5)
        ax.plot([x-1.2, x+1.2], [level3_y+1.2, level3_y-1.2], color=red, linewidth=2, zorder=5)
    else:
        ax.add_patch(plt.Circle((x, level3_y), 2, color=orange, zorder=3))
        ax.text(x, level3_y, label, ha='center', va='center', color='white', fontsize=6, fontweight='bold', zorder=4)

ax.text(6, 35, '363/365', color=purple, fontsize=9, fontweight='bold')
ax.text(32, 35, '2/365 ✗', color=red, fontsize=8)

bbox3 = mpatches.FancyBboxPatch((39, 16), 12, 6, boxstyle='round,pad=0.02', facecolor='#fef3c7', edgecolor=orange)
ax.add_patch(bbox3)
ax.text(45, 19, '363 safe', ha='center', fontsize=8, fontweight='bold', color='#92400e')
ax.text(45, 17, 'options', ha='center', fontsize=7, color='#92400e')

# Formula box
formula_box = mpatches.FancyBboxPatch((55, 8), 42, 24, boxstyle='round,pad=0.02', 
                                       facecolor='#f0fdf4', edgecolor='#22c55e', linewidth=2)
ax.add_patch(formula_box)
ax.text(76, 28, 'P(no shared birthday)', ha='center', fontsize=11, fontweight='bold', color='#166534')
ax.text(76, 22, '= 365/365 × 364/365 × 363/365', ha='center', fontsize=10, fontweight='bold', color='#166534')
ax.text(76, 16, '= 0.9918 (99.18%)', ha='center', fontsize=10, color='#166534')
ax.text(76, 11, 'P(≥2 share) = 0.82%', ha='center', fontsize=10, color='#166534')

# Legend
legend_box = mpatches.FancyBboxPatch((65, 50), 24, 18, boxstyle='round,pad=0.02', facecolor='#f9fafb', edgecolor='#e5e7eb')
ax.add_patch(legend_box)
ax.text(77, 65, 'Legend', ha='center', fontsize=9, fontweight='bold', color='#374151')
ax.add_patch(plt.Circle((69, 60), 1.2, color=green, zorder=3))
ax.text(72, 60, 'Safe path', ha='left', fontsize=8, color='#374151')
ax.add_patch(plt.Circle((69, 55), 1.2, color=light_red, ec=red, linewidth=1, zorder=3))
ax.text(72, 55, 'Collision', ha='left', fontsize=8, color='#374151')

plt.tight_layout()
plt.show()
```

### The Pattern Emerges

Notice what's happening:
- Each new person has **one fewer safe day** to choose from
- We **multiply** all the individual probabilities together
- The tree has **365 × 364 × 363 valid paths** out of **365³ total paths**

+++

---

## The General Case: N People

For *n* people, the pattern continues:

$$P(\text{no shared birthday}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \cdots \times \frac{365-n+1}{365}$$

Or more compactly using **product notation**:

$$P(\text{no shared birthday}) = \prod_{k=0}^{n-1} \frac{365-k}{365} = \frac{365!}{(365-n)! \cdot 365^n}$$

> **What does ∏ mean?** The symbol $\prod$ (capital Greek letter "pi") means "product" — multiply all the terms together. It's like $\sum$ (summation), but for multiplication instead of addition. Here, $\prod_{k=0}^{n-1}$ means "multiply the expression for each value of k from 0 to n-1."

### From Product to Factorial

Here's how to get from product notation to factorial form:

**Step 1:** Separate the numerators and denominators (n fractions, so n terms each):

$$\prod_{k=0}^{n-1} \frac{365-k}{365} = \frac{365 \times 364 \times 363 \times \cdots \times (365-n+1)}{365 \times 365 \times 365 \times \cdots \times 365} = \frac{365 \times 364 \times \cdots \times (365-n+1)}{365^n}$$

**Step 2:** Recognize the numerator as a "falling factorial" — it's $365!$ with the smaller terms cut off:

$$365 \times 364 \times \cdots \times (365-n+1) = \frac{365!}{(365-n)!}$$

This works because $365! = 365 \times 364 \times \cdots \times (365-n+1) \times (365-n) \times \cdots \times 1$, and dividing by $(365-n)!$ cancels the tail.

**Step 3:** Combine:

$$P(\text{no shared birthday}) = \frac{365!}{(365-n)! \cdot 365^n}$$

Let's see how this grows:

+++ {"tags": ["hide-input"]}

| People | P(no shared) | P(≥2 share) |
|--------|--------------|-------------|
| 2      | 99.73%       | 0.27%       |
| 3      | 99.18%       | 0.82%       |
| 5      | 97.29%       | 2.71%       |
| 10     | 88.31%       | 11.69%      |
| 15     | 74.71%       | 25.29%      |
| 20     | 58.86%       | 41.14%      |
| **23** | **49.27%**   | **50.73%** ← 50%! |
| 30     | 29.37%       | 70.63%      |
| 40     | 10.88%       | 89.12%      |
| 50     | 2.96%        | 97.04%      |
| 70     | 0.08%        | 99.92%      |

+++

The probability rises *fast*. Let's visualize this:

```{code-cell}
:tags: [remove-input]

import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))

people = list(range(1, 71))
probs = []
for n in people:
    p_no_shared = 1.0
    for i in range(n):
        p_no_shared *= (365 - i) / 365
    probs.append(1 - p_no_shared)

ax.plot(people, [p * 100 for p in probs], color='#3b82f6', linewidth=2.5)
ax.fill_between(people, [p * 100 for p in probs], alpha=0.2, color='#3b82f6')

# 50% threshold
ax.axhline(y=50, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=23, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.7)
ax.scatter([23], [50.73], color='#ef4444', s=100, zorder=5)
ax.annotate('23 people → 50.7%!', xy=(23, 50.73), xytext=(32, 38),
            fontsize=12, fontweight='bold', color='#ef4444',
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5))

# Mark our buildup points
ax.scatter([1], [0], color='#10b981', s=80, zorder=5)
ax.scatter([2], [probs[1] * 100], color='#10b981', s=80, zorder=5)
ax.scatter([3], [probs[2] * 100], color='#10b981', s=80, zorder=5)
ax.annotate('3 people\n0.82%', xy=(3, probs[2]*100), xytext=(8, 12),
            fontsize=10, color='#10b981', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#10b981'))

ax.set_xlabel('Number of People', fontsize=12)
ax.set_ylabel('P(at least 2 share a birthday) %', fontsize=12)
ax.set_title('The Birthday Paradox — How Probability Grows', fontsize=14, fontweight='bold')
ax.set_xlim(1, 70)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Why Is It So Counterintuitive?

Our intuition fails because we think about the wrong question. We instinctively imagine:

> "What's the chance someone shares *my* birthday?"

That probability *is* low — about 6% with 23 people. Here's why: each of the other 22 people has a 364/365 chance of *not* matching your birthday, so:

$$P(\text{someone matches mine}) = 1 - \left(\frac{364}{365}\right)^{22} \approx 0.059 = 5.9\%$$

But the birthday paradox asks a fundamentally different question:

> "What's the chance *any two people* share a birthday?"

The key difference: **you're no longer special**. We're not anchoring on one person's birthday — we're checking *every possible pair*.

### The Power of Pairs

With 23 people, there are:

$$\binom{23}{2} = \frac{23 \times 22}{2} = 253 \text{ possible pairs}$$

Each pair is an independent opportunity for a match. Think of it like buying lottery tickets:
- A single ticket (one pair) has low odds: roughly 1/365
- But 253 tickets (253 pairs) give you 253 chances to win!

While the pairs aren't truly independent (they overlap in people), the intuition holds: **more pairs = more chances for a collision**.

### The Exponential Trap

Here's another way to see it. The probability of *no* match drops with each person:

| People | Pairs | P(no shared) |
|--------|-------|--------------|
| 2      | 1     | 99.7%        |
| 10     | 45    | 88.3%        |
| 23     | 253   | 49.3%        |
| 30     | 435   | 29.4%        |

The pairs grow **quadratically** (n² growth), while our intuition thinks **linearly** (n growth). That's the heart of the paradox — we underestimate how fast opportunities for coincidence multiply.

+++

---

## Common Mistake to Avoid

When computing the probability, keep the **denominator as 365** throughout. A common error is writing:

$$\frac{1}{365} \times \frac{1}{364} \times \frac{1}{363} \quad \text{✗ WRONG}$$

But each person always chooses from 365 possible days. The shrinking **numerator** (365, 364, 363...) counts how many of those choices are "safe."

$$\frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \quad \text{✓ CORRECT}$$

+++

---

## Summary

The birthday paradox isn't really a paradox — just a demonstration of how quickly combinatorial possibilities grow.

**Key insights:**

1. **Use complement counting** — Calculate P(none share) and subtract from 1

2. **Build up person by person** — Each new person narrows the safe choices by one

3. **Multiply the probabilities** — P(all different) = product of individual "safe" probabilities

4. **Pairs grow quadratically** — n people create n(n-1)/2 pairs, which is why probability rises so fast

---

*Now you understand why, in a typical classroom of 30 students, there's a 70% chance two people share a birthday!*
