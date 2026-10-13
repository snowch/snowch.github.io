# Signed or Unsigned: Who Decides?

```{note}
Part 4 of [From Gates to a Compiled If](from-gates-to-a-compiled-if.md). It runs in the browser with nothing to install:
**[Open Signed or unsigned: who decides? →](https://snowch.github.io/programmer-calculator/docs/who-decides.html)**
```

The hardware stores no sign. One compare sets four flags, and the next instruction chooses which of them to read. This page makes that choice visible for any pair of bytes you pick or tap.

## What it covers

1. **Who decides.** Not the register and not the adder: the ALU serves both readings at once, and the instruction that follows a `CMP` picks one. Instruction sets come in pairs for exactly this reason.
2. **One compare, every branch.** `CMP A, B` for your pair, ending in `BEQ`, then every ARM conditional branch in two tables, unsigned and signed: what it means, which flags it tests, whether it is taken, and whether that agrees with the arithmetic. A "show the working" button draws the subtraction in bits.
3. **The same thing as a program.** One `if (a < b)` in C beside the ARM a compiler turns it into, with a `uint8_t`/`int8_t` toggle. The declaration changes the load and the branch; the lines that run are highlighted and a four-step trace follows the flags.
4. **What the loads do to the bits.** A summary, handing on to [From Bytes to Registers](bytes-to-registers.md).

## Next

- [From Bytes to Registers](bytes-to-registers.md), then the [flag drills](https://snowch.github.io/programmer-calculator/docs/flag-drills.html).
