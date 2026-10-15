# From Bytes to Registers

```{note}
An interactive page hosted alongside my Programmer Calculator. It runs in the browser with nothing to install:
**[Open From bytes to registers →](https://snowch.github.io/programmer-calculator/docs/bytes-to-registers.html)**
```

This page answers a question that the [Bit Lab](bit-lab.md)'s signed-or-unsigned section raises: why does a compiled `if` use one load instruction for a `uint8_t` and a different one for an `int8_t`, when both are a single byte in memory? It is part 8 of [From Gates to a Compiled If](from-gates-to-a-compiled-if.md), after the compare-and-branch page.

## What it covers

1. **Why a byte has to widen.** The processor computes in registers, ARM's are 32 bits wide, and there is no 8-bit compare, so a byte must be loaded into a full register and 24 bits that were never in memory have to be filled with something.
2. **One byte, both loads.** 0x80 through `LDRB` (zeros, giving 128) and `LDRSB` (copies of bit 7, giving −128), then 0x05, on which the two loads agree. The top bit decides.
3. **Your pair, live.** Two tappable bytes and a `uint8_t`/`int8_t` toggle. The panel widens both bytes, runs the 32-bit compare as A + NOT B + 1 across 32 columns with the carries shown, reads N Z C V off the result, and says when and why the 32-bit flags differ from the lab's 8-bit ones while the branch lands the same way.

## Next

- The [Bit Lab](bit-lab.md): the loads inside a compiled `if`, beside the C that produced it.
- The [flag drills](https://snowch.github.io/programmer-calculator/docs/flag-drills.html): widen a byte by hand and check it.
