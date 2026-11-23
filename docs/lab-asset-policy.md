# Self-hosted assets for every lab

All required course assets belong in that lab's own published project scope.
Fonts, diagrams, recordings, Python packages, workers, compilers, models,
WebAssembly, terminals, firmware, kernels and guest filesystems follow this rule.
There must be no runtime CDN fallback or essential third-party iframe. Ordinary
optional documentation links can remain and require internet access.

Build-time downloads are allowed with pinned versions and verified checksums.
Publish notices and corresponding source wherever redistribution licences require
it. Never remove an essential exercise or call it optional merely to pass a scan.
The full course download includes the complete transitive runtime dependency graph.

The [course design guide](https://github.com/snowch/cuda-python-lab/blob/main/docs/course-design-guide.md#self-hosted-course-assets)
and [offline/update design](https://github.com/snowch/cuda-cpp-lab/blob/main/docs/offline-update-design.md#self-hosted-dependency-boundary)
are the canonical adoption contracts.

## Verification and audit scope

`tools/audit-lab-assets.cjs` inventories first visits in fresh browser contexts,
blocks other origins, and records actual requests plus static URL candidates.
Install Playwright 1.62.1 in the test environment and provide Chromium. Run it with
`node tools/audit-lab-assets.cjs [sites.json] [report.json]`. Static URL candidates
need inspection: an ordinary reference is not an automatically loaded asset.

The initial 23 November 2026 inventory covered 38 catalogue entry pages. It found
external fonts/maths assets on Computer Systems and the Linear Algebra notebook
book, Google Fonts on Probability Distributions, and a GitHub API status request
on Digital Design. Follow-up source checks found CDN-hosted Python runtimes in
Probability Distributions and Linear Algebra's interactive course, and external
Linux emulator/kernel dependencies in Go Foundry and C Foundry. The report does
not establish that every feature of the other courses is independent.

Release acceptance must also deny requests outside the individual project scope
and exercise a cold runner, first compilation, an edited run, package-dependent
examples, embedded tools and lazy media. Then save a complete release, disconnect,
open an unvisited lesson and execute it. Inspect workers, iframes and service
workers as well as page requests. Keep saved learner work intact. Passing an
entry-page scan alone is insufficient evidence of offline readiness.
