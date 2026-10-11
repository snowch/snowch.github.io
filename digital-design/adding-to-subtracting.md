# From Adding to Subtracting

```{note}
An interactive page hosted alongside my Programmer Calculator. It runs in the browser with nothing to install:
**[Open From adding to subtracting →](https://snowch.github.io/programmer-calculator/docs/adding-to-subtracting.html)**
```

This sits between [From Gates to an ALU](gates-to-alu.md) and the [Bit Lab](bit-lab.md). It answers the question the lab's `CMP` raises the first time you see it: how does 3 − 5 come out as 0xFE, why is that the right answer, and where is the subtractor? Every section shows the same A − B, which you can change at the top or by tapping bits.

## What it covers

1. **A byte wraps around.** A 16-position wheel at 4 bits with unsigned labels outside and signed labels inside. Subtracting steps back around the wheel, and when the steps pass zero the register lands near the top of the range. The same pattern is 14 to one reading and −2 to the other.
2. **Negating: NOT, then add 1.** Stepping back B places is stepping forward 16 − B. Inverting every bit of B gives 15 − B, so NOT B + 1 is 16 − B, which is −B. Shown as bit rows for the current B.
3. **The adder does it.** Feed NOT B into the adder with the carry-in set to 1 and it computes A − B. Tap bits in A or B and watch the carries; the carry out of the top bit is the borrow signal inverted, which is why ARM's C = 1 means no borrow.

## Next

- The [Bit Lab](bit-lab.md): SUB and CMP are this circuit at 8 bits, and its `CMP` snippet now has a "show the working" button that draws these rows.
- [From Gates to a Register](gates-to-register.md): the other half of a CPU.
- The [flag drills](https://snowch.github.io/programmer-calculator/docs/flag-drills.html) begin with predicting the flags of a `CMP`.
