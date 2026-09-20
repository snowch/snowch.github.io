# Sizing and TCO

Capacity planning, sizing and total cost of ownership, modelled as code. A self-study book and toolkit built around one question — how big, how much, and how wrong could I be? — and one rule for answering it: every number says where it came from.

## Topics Covered

- **What You're Sizing**: point estimates, what a workload is, where the numbers come from, peak, mean and growth
- **Ceilings**: Little's Law, queueing and the knee, when adding servers stops helping, regime changes
- **Sizing**: capacity, bandwidth and the binding constraint, headroom and failure domains
- **Uncertainty**: Monte Carlo from first principles, correlation and convergence
- **Cost**: capex, opex and lifecycle, power first, unit economics, the five-year model
- **Sensitivity**: which input is really the answer, and the missing node
- **Presenting It**: a TCO for finance, and comparing two TCOs
- **Appendices**: DSL reference, the Monte Carlo module, distributions, units, two worked models, glossary

## What Makes It Different

- **Models are code** — a model is a graph of named quantities in a YAML file, evaluated and version controlled like anything else
- **Units fail the build** — every node declares a unit, so a model that multiplies the wrong two things is rejected rather than published
- **Provenance is mandatory** — every input declares whether it is a measured fact, a vendor's claim, or an assumption
- **Gaps stay visible** — a constant nobody has measured leaves the nodes below it empty instead of quietly filled in
- **Monte Carlo from scratch** — inverse-transform sampling, rank correlation and convergence written to be read, not called

## Access the Full Book

**[Sizing and TCO →](https://snowch.github.io/sizing-and-tco/)**

The complete book is available as an interactive resource, with worked models you can adjust and re-run in the browser. Source and toolkit on [GitHub](https://github.com/snowch/sizing-and-tco).
