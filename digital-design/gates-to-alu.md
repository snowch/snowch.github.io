# From Gates to an ALU

```{note}
An interactive page hosted alongside my Programmer Calculator. It runs in the browser with nothing to install:
**[Open From gates to an ALU →](https://snowch.github.io/programmer-calculator/docs/gates-to-alu.html)**
```

This is the first of two short prerequisites for the [Bit Lab](bit-lab.md). It assumes only the truth tables of XOR, AND and OR, and it builds the circuit at the heart of every CPU's arithmetic logic unit from those gates up.

## What it covers

1. **One full adder, five gates.** Two XOR, two AND and one OR add two bits and a carry-in, producing a sum and a carry-out. Tap A, B and carry-in and watch every wire and gate follow, with the matching row of the truth table highlighted. The notation used in gate equations (⊕, ·, +, ¬) is defined on the page.
2. **Four of them add 4-bit numbers.** The adders chained by their carry wires, which is exactly adding on paper: each column has its own adder, and the only connection between columns is the carry. The same sum is shown as long addition with the carry row in red. Tap any FA box to open it and see its five gates working on that column, and press *ripple* to watch the carries settle one column at a time, which is why the top bit is always the last to be ready.
3. **Where this leads.** Bitwise operations need one gate per bit and no carry; an ALU is these circuits side by side with control wires choosing the result.

## Next

- [From Adding to Subtracting](adding-to-subtracting.md): how this same adder subtracts, with the borrow read off the carry.
- [From Gates to a Register](gates-to-register.md): the other half of a CPU, how the same kinds of gates are made to remember.
- The [Bit Lab](bit-lab.md): this adder inside a working 8-bit ALU with subtraction, the N Z C V flags and guided exercises.
