# Two Readings: Drills

```{note}
Part 3 of [From Gates to a Compiled If](from-gates-to-a-compiled-if.md). It runs in the browser with nothing to install:
**[Open Two readings: drills →](https://snowch.github.io/programmer-calculator/docs/readings-drills.html)**
```

Five short randomised exercises on the idea every later page assumes: one bit pattern is two numbers, and which one you get depends only on how you read the top bit. They start at 4 bits, where there are only sixteen patterns, and offer 8 bits once that feels mechanical. Each asks for your answer before it shows the working, drawn as bits.

1. **Read a pattern both ways.** Give the unsigned and signed readings; the reveal shows the place values, with the top bit worth minus its power of two under the signed reading.
2. **Write a value as a pattern.** The reveal shows both routes: count back from zero round the wheel, or invert and add 1.
3. **Step round the wheel.** Predict where a subtraction lands and what it reads as; at 4 bits the reveal draws the wheel with the arc.
4. **Negate by hand.** Invert, add 1, check the reading. The one pattern that cannot be negated turns up.
5. **Does it fit?** Judge whether the unsigned and signed readings of a sum are the true sum: the C and V flags before they have names.

## Next

- [Bit Lab](bit-lab.md): the ALU that does this arithmetic, with the flags drawn from gates.
