# Books, Labs & Courses

Explore books, courses and hands-on labs with worked examples, exercises and experiments.

## Computer Systems

| Lab or course | What you'll do |
|---------------|----------------|
| [Digital Design](https://snowch.github.io/digital-design/) | Build from bits and logic gates towards a working computer |
| [Bit Lab](https://snowch.github.io/bit-lab/) | Explore binary and hexadecimal, fixed-width arithmetic, CPU flags, masks, shifts and byte order through interactive experiments and checked questions. **Draft course.** |
| [Assembly Lab](https://snowch.github.io/assembly-lab/) | Explore processor instructions one at a time, with the corresponding C alongside them |
| [RISC-V Assembly Lab](https://snowch.github.io/riscv-assembly-lab/) | Learn RISC-V through focused lessons: edit, compile, debug and run real assembly and C in an xv6 guest, all in your browser |
| [Systems Lab](https://snowch.github.io/computer-systems/) — *Systems From Scratch* | Explore machine code, memory, operating systems and CPU performance through RISC-V/xv6 experiments and measurements on real hardware |
| [xv6 Lab](https://snowch.github.io/xv6-lab/) | Learn how an operating system works through 25 browser experiments with a real xv6 kernel, including processes, virtual memory, scheduling and filesystems, with live kernel inspectors |
| [C Network Programming Lab](https://snowch.github.io/c-network-programming-lab/) | Start with Linux socket APIs in C, then work through TCP, UDP, DNS, TLS, Unix-domain sockets and loopback diagnostics |
| [C Linux Systems Lab](https://snowch.github.io/c-linux-systems-lab/) | Start with Linux API return values, errno and file I/O, then explore processes, concurrency, event loops, tracing and supervision |
| [Introduction to Linux Device Drivers in C](https://snowch.github.io/c-linux-device-driver-lab/) | Start with module load, unload and parameters before testing character devices, wait queues, memory mapping and synthetic platform binding |
| [GPU Lab](https://snowch.github.io/gpu-lab/) | Investigate tensor layouts, memory access and GPU execution through browser simulations and downloadable CUDA/PyTorch examples |

The three C labs run Linux in your browser using WebAssembly. Choose **Download for offline use** in each lab before disconnecting.

## Programming

| Lab or course | What you'll do |
|---------------|----------------|
| [C Foundry](https://snowch.github.io/c-foundry/#/course) | Learn C, memory management, algorithms and database internals through editable browser examples and native investigations |
| [C++ Foundry](https://snowch.github.io/cpp-foundry/#/course) | Explore modern C++20, ownership, resource management, algorithms and database components with runnable examples |
| [Rust Foundry](https://snowch.github.io/rust-foundry/) | Learn Rust, ownership and borrowing through browser-compiled examples and exercises, then build a database through optional TinyDB projects |

## Data Engineering & AI

Parquet Lab builds the reader used by Query Engine Lab, so they work well in that order.

| Book, lab or course | What you'll learn |
|---------------------|-------------------|
| [Parquet Lab](https://snowch.github.io/parquet-book/) — *Parquet, byte by byte* | Build a Parquet reader in Python and Rust using real file bytes, interactive examples and browser exercises |
| [Query Engine Lab](https://snowch.github.io/query-engine-book/) — *Queries, operator by operator* | Build a Python query engine on Apache Arrow and compare operator costs with DuckDB. **In progress.** |
| [Iceberg Lab](https://snowch.github.io/iceberg-lab/) | Explore table metadata, snapshots, schema evolution, pruning, maintenance and governance through 42 focused browser lessons with real Iceberg experiments running in WebAssembly |
| [PostgreSQL Lab](https://snowch.github.io/postgresql-lab/) | Investigate query plans, indexes, transactions, migrations and recovery with editable SQL running in the browser and optional PostgreSQL server extensions |
| [Transactions Lab](https://snowch.github.io/transactions-lab/) | Design, break and repair transactions across databases, file systems and distributed workflows through editable SQL, controlled failure experiments and real-system extensions |
| [Event Streaming Lab](https://snowch.github.io/event-streaming-lab/) | Explore events, state, windows, replay and recovery through real Bytewax and Kafka protocol experiments running inside browser Linux |
| [Messaging Patterns Lab](https://snowch.github.io/messaging-patterns-lab/) | Explore messaging patterns, routing, queues, retries and acknowledgements through 20 real ZeroMQ investigations and 10 deeper readings in your browser |
| [Sizing & TCO Lab](sizing.md) | Adjust capacity and cost models, explore uncertainty and sensitivity, and trace inputs to their sources |
| [Benchmarking Lab](https://snowch.github.io/benchmarking-lab/chapters/start-here.html) | Design fair technology comparisons, run browser experiments and downloadable service harnesses, analyse evidence and build a recommendation |
| [Observability & Anomaly Detection](anomaly-detection.md) | Work through notebooks and examples for embeddings, anomaly detection and deployment with observability data |
| [ML-101](https://github.com/snowch/ml-101/blob/main/00-README-FIRST.ipynb) | Practise machine learning fundamentals with scikit-learn |
| [Embeddings at Scale](embeddings.md) | Vector representations, similarity search and deploying embedding systems at scale |
| [Metadata Systems](https://snowch.github.io/metadata-book/) | Understand metadata collection, lineage, quality, governance and controlled reuse through worked examples and a visual metadata map |

## Research & Evidence

| Lab or course | What you'll do |
|---------------|----------------|
| [Scientific Method Lab](https://snowch.github.io/scientific-method-lab/) | Design fair experiments, analyse uncertainty and test hypotheses through 30 short Python investigations running in your browser |
| [Causal Inference Lab](https://snowch.github.io/causal-inference-lab/) | Infer causal effects from observational data through 18 browser Python investigations covering causal diagrams, fair comparisons and the assumptions behind each estimate |
| [Sensitivity Analysis Lab](https://snowch.github.io/sensitivity-analysis-lab/) | Use simulations and logs to find what drives an outcome, then apply the methods to a household electricity budget |

## Mathematics

| Book, lab or course | What you'll learn |
|---------------------|-------------------|
| [Linear Algebra V2](https://snowch.github.io/learn_linear_algebra/interactive/) | Explore vectors, transformations, linear systems, decompositions and PCA through visual experiments, guided practice and optional Python running in the browser |
| [Monte Carlo Lab](https://snowch.github.io/monte-carlo-lab/) | Model uncertainty, compare scenarios and make decisions through 60 core and 16 advanced Python investigations running in your browser |
| [SymPy Lab](https://snowch.github.io/sympy-lab/) | Learn symbolic algebra, equations, calculus and matrices through 48 Python investigations running in your browser, with optional maths refresher panels |
| [Probability](probability/probability.md) | Probability rules, conditional probability, distributions and Python examples |
| [Linear Algebra](linear-algebra.md) | Vectors, matrices, eigenvalues and singular value decomposition, with SageMath exercises |
| [Matrix Calculus](matrix-calculus.md) | Matrix derivatives, gradients, Jacobians and Hessians for machine learning |

## Media

| Lab | What you'll do |
|-----|----------------|
| [Video Lab](https://snowch.github.io/video-lab/) | Run editable FFmpeg and ffprobe commands in the browser to practise cutting, joining, framing, sound, subtitles and export |

Assembly Lab, GPU Lab and Video Lab also support offline learning. Follow each lab's instructions to prepare its offline content.

## Wellbeing & self-reflection

| Lab | What you’ll explore |
|-----|---------------------|
| [Cravings Lab](https://snowch.github.io/cravings-lab/) | Learn about cravings, hunger and habits through six short lessons, then explore optional check-ins, guided practices and small experiments. Tab-only by default, with optional device saving. Self-reflection for adults, not diagnosis or treatment. |

## Psychology

The three Common Ground courses share fictional repair-café stories, narration, decisions, written practice, optional research, glossaries and browser offline access.

| Book or course | What you'll learn |
|----------------|-------------------|
| [Social Psychology](https://snowch.github.io/common-ground-social-psychology/) | Explore social psychology through ten illustrated stories, narrated chapters, decisions and recall practice |
| [Cognitive Psychology](https://snowch.github.io/common-ground-cognitive-psychology/) | Explore attention, perception, memory, language and reasoning through ten narrated café stories, evidence checks and practice |
| [Personality Psychology](https://snowch.github.io/common-ground-personality-psychology/) | Understand traits, assessment, development, motives, identity and culture through ten narrated stories and evidence-based interpretation |
| [Everyday CBT](https://snowch.github.io/everyday-cbt/) | Learn introductory CBT skills through twelve illustrated stories, recorded narration, decisions and practical labs. **Draft course; practitioner review pending.** |
| [Understanding People](https://snowch.github.io/understanding-people/) | Understand what drives people through psychology, sociology and decision science, with practical habits and an expanded reference library |
