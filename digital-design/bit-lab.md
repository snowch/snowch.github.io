# Bit Lab: Gates, Adders, Registers and Flags

```{note}
The Bit Lab is an interactive page hosted alongside my Programmer Calculator. It runs in the browser with nothing to install:
**[Open the Bit Lab →](https://snowch.github.io/programmer-calculator/docs/bit-lab.html)**
```

The Bit Lab is a working miniature ALU made of gates you can tap, with the N Z C V flags built from gates beside it. Its two prerequisites are the previous pages here, [From Gates to an ALU](gates-to-alu.md) and [From Gates to a Register](gates-to-register.md); the lab summarises both in a paragraph each and links to them, so you can also start here and step back when needed. It leads on to the ALU and flag material that Computer Architecture will assume.

## What it covers

1. **Notation and prerequisites.** What ⊕, ·, + and ¬ mean inside a gate equation, a three-question self-check, and a summary of the two companion pages.
2. **Background.** What an ALU is and where an instruction's operation comes from, with a table of example instructions; the two readings of the same bits; negation; and the N Z C V flags as four separate stored bits.
3. **A miniature ALU.** Thirteen operations at 4 or 8 bits, the gate network drawn for the chosen operation, inputs you can flip bit by bit, and the result in binary, hex, unsigned and signed.
4. **The flags, live.** Each flag's gates drawn from the current result, and why carry (unsigned overflow) and V (signed overflow) are independent.
5. **Exercises, a quiz, and a breadboard build.** Eight guided exercises, six check-yourself questions, and a schematic plus parts list for building the whole thing from 74HC-series chips.

Jump straight to a part:
[Background](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#background) ·
[The lab](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#lab) ·
[Guided exercises](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#exercises) ·
[Build it for real](https://snowch.github.io/programmer-calculator/docs/bit-lab.html#build)

## Companions

- **[Two's complement: a short tutorial (PDF)](https://snowch.github.io/programmer-calculator/docs/twos-complement.pdf)** — seven pages on reading and writing negative numbers, NOT versus negation, arithmetic modulo 256, overflow, sign extension and reading hex directly, with practice problems and a cheat sheet.
- **[Programmer Calculator](https://snowch.github.io/programmer-calculator/)** — the installable calculator the Bit Lab lives with: hex, decimal, octal and binary, bitwise operations, shifts and rotates, CPU-style C V Z N flags, and 110 digital-design exercises whose primers link back to the tutorial and the lab.

## Where to go next

Once the register section makes sense, continue with the [D Flip-Flops tutorial](d_flipflop_tutorial.md), which opens up the edge-triggered cell: the master and slave latches inside it, setup and hold time, and clock-to-Q delay.

For the whole journey from signals to a working CPU, with lessons on gates, the ALU, registers, flags and instructions, take the [Digital Design course](https://snowch.github.io/digital-design/).
