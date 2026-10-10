# From Gates to a Register

```{note}
An interactive page hosted alongside my Programmer Calculator. It runs in the browser with nothing to install:
**[Open From gates to a register →](https://snowch.github.io/programmer-calculator/docs/gates-to-register.html)**
```

This is the second prerequisite for the [Bit Lab](bit-lab.md), and the gentlest route into the sequential-logic notes that follow it here. It starts from a question the adder raises: if a gate's output only ever follows its inputs, how does a computer remember anything?

## What it covers

1. **Gates that compute, gates that remember.** Why the ALU needs no memory, why the registers on either side of it do, and the trick that makes memory from ordinary gates: *feedback*, an output led back round to become an input.
2. **The smallest memory: two NOR gates.** Each one's output is an input of the other, giving two stable states held for as long as the power is on. An inverter and two AND gates in front turn the pair into a D latch: while enable is 1, Q copies D; drop enable to 0 and Q stays put while you change D.
3. **A clock, and four cells in a row.** What a clock is and what a clock edge is, why real registers use an edge-triggered flip-flop rather than a transparent latch, and a 4-bit register of four flip-flops with a clock button and a load line. Between edges the D inputs can change freely; a clock edge copies them into Q only when load is on.

## Next

- The [D Flip-Flops tutorial](d_flipflop_tutorial.md) opens up the edge-triggered cell: the master and slave latches inside it, setup and hold time, and clock-to-Q delay, with an [interactive visualisation](d-flip-flop-interactive.ipynb).
- The [Bit Lab](bit-lab.md): what the registers feed, an 8-bit ALU with its flags, each flag one flip-flop loaded at the same clock edge as the result.
