# Manager Notes

- GPU pipeline is one Protocol (not three) — Compiler/Benchmarker/Profiler are internal to the pipeline impl
- max_rounds is a safety net, not a first-class exit condition (user removed budget from architecture)
- Stubs should use deterministic seeds in tests to avoid flaky results
- compile fail short-circuit back to Coding Agent is V2 — V1 discards and lets next round react
