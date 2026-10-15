# Where C Meets the Wheel

```{note}
A side trip from [From Gates to a Compiled If](from-gates-to-a-compiled-if.md). It runs in the browser with nothing to install:
**[Open Where C meets the wheel →](https://snowch.github.io/programmer-calculator/docs/c-meets-the-wheel.html)**
```

Ten small C programs that go wrong at the edge of the integer range, each stepped through one ARM instruction at a time with the registers drawn as bits before and after and the N Z C V flags shown. Everything is at 8 bits with `int8_t` and `uint8_t` so every bit is on screen; the 32-bit equivalents are named where they differ.

1. **The loop that never ends.** An unsigned countdown: 0 − 1 is 255, and 255 ≥ 0.
2. **127 + 1.** The largest signed value plus one becomes −128, with V set and C clear.
3. **200 into an `int8_t`.** No arithmetic, only a narrowing store: the bits stay 0xC8 and now read as −56.
4. **−(−128).** Negating the most negative value gives it back, with V set. The `abs()` trap.
5. **Parsing −128.** A parser that accumulates positively and negates at the end, and the one input whose positive form never fitted.
6. **Detecting a wrap in C.** C cannot read the carry flag, so the idiom `d > a` asks a question only a wrap can answer yes.
7. **`%d` and `%u`.** One register printed under both readings; nothing is computed, the format chooses what the top bit is worth.
8. **`INT_MIN / −1`.** The one division that overflows: ARM's SDIV hands back the same pattern, x86's IDIV raises an exception.
9. **The check that vanished.** `if (x + 1 > x)` compiled with optimisation contains no add and no compare; the compiler assumed the undefined case away. `-fwrapv` brings it back.
10. **255 + 1, and detecting it.** The carry out of the top bit, and the `s < a` idiom that reveals it from C.

Read it any time after [From Adding to Subtracting](adding-to-subtracting.md), whose collapsed "In C" notes these scenarios expand.
