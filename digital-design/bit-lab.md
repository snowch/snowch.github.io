# Bit Lab: Gates, Adders, Registers and Flags

```{note}
The Bit Lab is an interactive page hosted alongside my Programmer Calculator. It runs in the browser with nothing to install:
**[Open the Bit Lab →](https://snowch.github.io/programmer-calculator/docs/bit-lab.html)**
```

The Bit Lab builds a miniature CPU datapath from gates you can tap, in the order the ideas depend on each other. It is the on-ramp to the sequential-logic notes that follow it here, and to the ALU and flag material that Computer Architecture will assume.

## What it covers

1. **Notation and prerequisites.** What ⊕, ·, + and ¬ mean inside a gate equation, and a three-question self-check.
2. **One full adder from five gates.** Two XOR, two AND and one OR. Tap A, B and carry-in and watch the sum and carry-out follow, with the truth table row highlighted.
3. **Four of them add 4-bit numbers.** The adders chained by their carry wires, the same sum written out as long addition, any column opened to show its gates, and a *ripple* button that settles the columns one at a time.
4. **Gates that remember.** Why the ALU needs no memory, how two cross-coupled NOR gates hold a bit, a D latch built from five gates, and a 4-bit register of flip-flops with a clock button and a load line.
5. **A miniature ALU.** Thirteen operations at 4 or 8 bits, the gate network drawn for the chosen operation, inputs you can flip bit by bit, and the result in binary, hex, unsigned and signed.
6. **The N Z C V flags.** Each flag's gates drawn live, why carry (unsigned overflow) and V (signed overflow) are independent, and where an instruction's operation comes from, with a table of example instructions.
7. **Exercises, a quiz, and a breadboard build.** Eight guided exercises, six check-yourself questions, and a schematic plus parts list for building the whole thing from 74HC-series chips.

Jump straight to a part:
[Background](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#background) ·
[The lab](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#lab) ·
[Guided exercises](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#exercises) ·
[Build it for real](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#build)

## Companions

- **[Two's complement: a short tutorial (PDF)](https://snowch.github.io/programmer-calculator/docs/twos-complement.pdf)** — seven pages on reading and writing negative numbers, NOT versus negation, arithmetic modulo 256, overflow, sign extension and reading hex directly, with practice problems and a cheat sheet.
- **[Programmer Calculator](https://snowch.github.io/programmer-calculator/)** — the installable calculator the Bit Lab lives with: hex, decimal, octal and binary, bitwise operations, shifts and rotates, CPU-style C V Z N flags, and 110 digital-design exercises whose primers link back to the tutorial and the lab.
- **[Build a 4-bit binary calculator from logic gates](https://github.com/snowch/binary-adder)** — a breadboard tutorial that builds the half adder, full adder and ripple-carry adder from 74HC86, 74HC08 and 74HC32 chips.

## Where to go next

Once the register section makes sense, continue with the [D Flip-Flops tutorial](d_flipflop_tutorial.md), which opens up the edge-triggered cell: the master and slave latches inside it, setup and hold time, and clock-to-Q delay.
