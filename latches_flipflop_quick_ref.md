# Sequential Logic Quick Reference

```{note}
This reference covers the behavior, truth tables, and timing requirements for common memory elements in digital design.

```

---

## Latches vs. Flip-Flops

| Feature | **Latches** | **Flip-Flops** |
| --- | --- | --- |
| **Trigger** | Level-Triggered (High/Low) | Edge-Triggered (Rising/Falling) |
| **Transparency** | Transparent during active level | Only captures data on the transition |
| **Primary Use** | Simple storage, high-speed buffers | Synchronous logic, registers, FSMs |

---

## Latches (Level-Triggered)

### SR Latch (Set-Reset)The SR latch is the most basic form of memory, but it includes an invalid state.

| EN | S | R | Q(next) | State |
| --- | --- | --- | --- | --- |
| 0 | X | X | Q | **Hold** |
| 1 | 0 | 0 | Q | **Hold** |
| 1 | 0 | 1 | 0 | **Reset** |
| 1 | 1 | 0 | 1 | **Set** |
| 1 | 1 | 1 | ? | {red}`**FORBIDDEN**` |

### D Latch (Transparent)Eliminates the forbidden state by ensuring S and R are never both high.

| EN | D | Q(next) | State |
| --- | --- | --- | --- |
| 0 | X | Q | **Hold** |
| 1 | 0 | 0 | **Reset** |
| 1 | 1 | 1 | **Set** |

---

## Flip-Flops (Edge-Triggered)

### D Flip-Flop (Data)The industry standard for registers and synchronous design.

| CLK | D | Q(next) |
| --- | --- | --- |
| ↑ | 0 | 0 |
| ↑ | 1 | 1 |
| — | X | Q |

### JK Flip-Flop (Universal)

```{tip}
The JK Flip-Flop is "universal" because it can be configured to act like a D or T flip-flop.

```

| CLK | J | K | Q(next) |
| --- | --- | --- | --- |
| ↑ | 0 | 0 | Q (Hold) |
| ↑ | 0 | 1 | 0 (Reset) |
| ↑ | 1 | 0 | 1 (Set) |
| ↑ | 1 | 1 | **Q̄ (Toggle)** |

---

## Timing ConstraintsFor reliable operation, data must be stable around the active clock edge.

| Parameter | Symbol | Description |
| --- | --- | --- |
| **Setup Time** | t_{su} | Minimum time input must be stable **before** the clock edge. |
| **Hold Time** | t_h | Minimum time input must remain stable **after** the clock edge. |
| **Prop Delay** | t_{pd} | Time taken for the output to reflect the input change. |

---

## Selection Guide

```{important}
**Rule of Thumb:** Use **D Flip-Flops** for almost all synchronous logic (pipelines, registers) unless you specifically require a toggle behavior (counters), in which case **T Flip-Flops** are more efficient.
```
