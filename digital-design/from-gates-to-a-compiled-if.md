# From Gates to a Compiled If

```{note}
Seven short interactive pages and a side trip, hosted alongside my Programmer Calculator. They run in the browser with nothing to install:
**[Open the contents page →](https://snowch.github.io/programmer-calculator/docs/notes.html)**
```

These pages grew out of my own notes while learning how a processor handles signed and unsigned numbers. They start with logic gates you can tap and end with an `if` statement compiled to ARM, traced flag by flag. Each page answers the question the one before it raises, every claim is drawn as bits or gates you can change, and the pages compute their values in the open rather than simulating anything behind the scenes.

They are about real machines: the assembly is ARM, the types are C's, and where x86 differs they say so. The [Digital Design course](https://snowch.github.io/digital-design/) is the other track, where you build a machine of your own from gates to a running program. These notes read the machines you will actually meet.

## Reading order

1. [From Gates to an ALU](gates-to-alu.md): one full adder from five gates, four of them chained, and why the carry ripples.
2. [From Adding to Subtracting](adding-to-subtracting.md): why 3 − 5 comes out as 0xFE, negation as NOT then add 1, the borrow as a carry.
3. [Two Readings: Drills](readings-drills.md): one pattern is two numbers; read, write, step, negate and judge, at 4 bits first.
4. [Bit Lab](bit-lab.md): a miniature ALU drawn as gates, thirteen operations, the N Z C V flags, exercises, a quiz and a breadboard build.
5. [Signed or Unsigned: Who Decides?](who-decides.md): every ARM branch on one compare, then a real `if` in C beside the assembly a compiler emits.
6. [From Bytes to Registers](bytes-to-registers.md): why a byte is widened to fill a register, and what the two loads write into the spare bits.
7. [Flag drills](https://snowch.github.io/programmer-calculator/docs/flag-drills.html): randomised predict-then-check exercises on all of the above.

Side trip: [From Gates to a Register](gates-to-register.md), how the same kinds of gates are made to remember, any time after part 4.

You need binary and hex, and the truth tables of NOT, AND, OR and XOR. Nothing else is assumed.
