# Benchmarking conventions

## BENCH-001 — Benchmark named representative scenarios

- Define a named workload, measured unit, optimization direction, sampling method, and environment fingerprint.
- Use deterministic inputs and keep setup outside the measured region unless setup itself is the subject of the benchmark.
- Record throughput when it makes differently sized workloads easier to compare.

## BENCH-002 — Compare candidates against versioned baselines

- Compare equivalent harness runs on equivalent infrastructure.
- Fail only regressions beyond committed relative and absolute-noise thresholds.
- Treat incompatible environment fingerprints as incomparable rather than silently accepting or rejecting the candidate.

## BENCH-003 — Separate blocking regression signals from noisy wall-clock timing

- Do not make raw wall-clock deltas from ordinary shared CI runners a blocking gate.
- Use a deterministic or sufficiently low-noise proxy for blocking regression thresholds when that proxy represents the intended workload.
- Keep the metric identity explicit: instruction counts, allocations, cache events, operation counts, and wall-clock latency are different evidence and must not be presented as interchangeable.
- Use controlled hardware or otherwise controlled execution environments whenever wall-clock latency itself is a blocking contract.
- Persistent wall-clock history used for regression decisions must come from dedicated, pinned, or otherwise controlled execution environments with workload and environment fingerprints.
- Bind controlled baseline and candidate observations to exact source or artifact identities and preserve the raw observations needed to explain the comparison.
- Ordinary shared-runner wall-clock results may remain informational but must not silently join controlled history as equivalent samples.
- Missing history, unavailable runners, or incompatible fingerprints are unavailable or incomparable evidence rather than a green regression result.

## BENCH-004 — Benchmark enough workload shapes to expose scaling regressions

- Performance-sensitive algorithms should cover multiple representative input sizes or shapes when one fixed case could hide an asymptotic, cache, allocation, or branch-behavior regression.
- Include common cases and a bounded stress or adversarial case when they exercise materially different behavior.
- Prefer a small stable matrix over an exhaustive benchmark suite that is too expensive to run or review routinely.

## BENCH-005 — Keep benchmark references outside production boundaries

- Reference implementations and comparison libraries should remain development-only or explicitly feature-gated unless production behavior deliberately depends on them.
- Benchmark-only dependencies must not silently become part of public APIs or runtime selection paths.
- Competitive benchmarks are evidence about implementation quality, not a requirement to copy another library's architecture.

## BENCH-006 — Treat profiling as explanatory evidence

- Capture CPU/hotspot profiles against a named representative scenario and record the exact source revision, profiler/tool version, target, features, and environment fingerprint.
- Preserve whether evidence is sampled, instrumented, or deterministic; sampled hotspots are evidence about where time was observed, not a correctness proof or an exact operation count.
- Compare profiles only when the workload and relevant environment inputs are equivalent. Otherwise report them as incomparable.
- Use profiles to explain or prioritize an observed regression; do not invent a performance regression solely from a changed sample percentage on an uncontrolled run.

## BENCH-007 — Keep memory metrics semantically distinct

- Record which memory signal is measured: retained/live heap, allocation count or rate, RSS/working set, GC pause, sampled peak, or another explicitly named metric.
- Use deterministic or bounded representative scenarios and retain tool, runtime, target, feature, and workload fingerprints with the evidence.
- Do not substitute one memory metric for another. A lower allocation rate does not prove lower retained memory, and a lower sampled peak does not prove lower steady-state RSS.
- Treat unavailable collectors or incompatible runtime configurations as unavailable/incomparable evidence rather than success.

## BENCH-008 — Bound service and load scenarios explicitly

- Service/load checks must declare the target, fixture/state setup, request count or duration, concurrency bound, timeout, and measured throughput/latency/error metrics.
- Default only to isolated local/test targets. Never run load scenarios against production endpoints unless a separate explicit operational policy authorizes that target.
- Keep load/performance evidence separate from behavioral correctness. A fast response with the wrong status or state transition is still incorrect.
- Prefer a small reproducible smoke workload for routine validation and reserve broad stress/capacity experiments for an explicit wider tier.

## BENCH-009 — Measure browser and mobile performance through representative journeys

- Browser traces should identify the immutable build artifact and representative interaction journey being measured; developer-server timing is not automatically deployment timing.
- Keep long tasks/main-thread work, render counts or render budgets, network/loading evidence, and audit scores as distinct metrics rather than one synthetic truth.
- Mobile evidence should identify device/runtime conditions and distinguish startup, frame stalls, CPU, memory, and interaction latency.
- Performance traces supplement behavioral and accessibility tests; loading a page or completing a trace does not prove the interaction is correct.

## BENCH-010 — Compare size only across equivalent artifacts

- Binary/bundle size evidence must identify the exact artifact, entry point, target, profile/mode, feature set, and relevant toolchain inputs.
- Compare equivalent artifacts against a versioned baseline; incompatible targets, feature sets, minification modes, or packaging boundaries are incomparable.
- Keep size evidence separate from runtime latency, CPU, and memory evidence. Smaller is not automatically faster or better.
- When a size budget is blocking, commit the budget and the artifact-selection rule so the measured boundary cannot drift silently.

## Performance architecture

## BENCH-011 — Declare the hot-path cost model

- Performance-sensitive repositories and components must name the workload dimensions that drive execution cost, such as entities, candidates, samples, contacts, rows, pixels, queries, materializations, or network messages.
- Describe which stages are expected to be constant, linear, logarithmic, quadratic, or otherwise bounded in those dimensions.
- Review architectural changes against the cost model, not only against local function complexity.
- When a production path changes the unit of repeated work, update the representative scenario and its counters in the same slice.

## BENCH-012 — Make deterministic work counters part of observability

- Expose cheap structured counters at natural operation boundaries for expensive reusable work such as candidate visits, exact evaluations, cache rebuilds/reuses, allocations, copied or materialized objects/bytes, solver passes, queries, or serialization bytes.
- When copying or materialization can dominate a hot path, count both occurrences and affected volume, such as elements or bytes, so architecture-level amplification is visible.
- Prefer deterministic operation-count, allocation-count, or copy/materialization-count sentinels for blocking CI when they faithfully represent the workload.
- Treat counters as evidence about work performed, not as substitutes for behavioral correctness or real latency measurements.
- Consumers should aggregate counters from authoritative foundations instead of reimplementing equivalent instrumentation locally.

## BENCH-013 — Reuse work deliberately and make invalidation explicit

- Do not repeatedly clone, parse, integrate, prepare, index, materialize, or query identical state inside one logical operation when reusable state can be retained safely.
- Apply `PRINCIPLE-008` for ownership-preserving sharing decisions across module, layer, and API boundaries; this rule adds performance evidence and reuse requirements rather than a second sharing policy.
- Give prepared or cached state an explicit owner, lifetime, identity key, and invalidation rule.
- When a hot path legitimately performs full copies or materializations under `PRINCIPLE-008`, expose representative occurrence and affected-volume counters so that work remains visible in the cost model and performance evidence.
- Quiescent systems should avoid repeated simulation/render/query work when authoritative state has not changed.

## BENCH-014 — Separate reference semantics from production mechanics

- Apply `PRINCIPLE-010` and `TEST-022` when a performance-sensitive production path differs from a simple semantic reference.
- Production implementations may use retained indexes, shared sampling, reordered traversal, bounded common-case arithmetic, incremental updates, or other optimizations once differential evidence establishes semantic equivalence.
- Keep the reference outside production boundaries under `BENCH-005`; do not preserve an expensive reference execution strategy merely because it was the easiest version to prove initially.
- Pair equivalence evidence with `BENCH-011` cost models and `BENCH-012` deterministic work counters so a semantically correct optimization cannot silently regress into whole-input recomputation, copying, or materialization.

## BENCH-015 — Make exact common cases cheap

- Exactness and determinism do not require every operation to use the widest supported representation.
- Prefer the smallest exact bounded representation that can prove the common operation, with a deterministic exact fallback for genuinely wider inputs.
- Preserve the fallback as the correctness authority and add direct fast-path-versus-fallback equivalence coverage.
- Do not replace exact semantics with approximation solely to recover performance unless the product contract explicitly allows that approximation.

## BENCH-016 — Require representative performance evidence before feature depth

- Once a performance-sensitive subsystem drives a realistic vertical slice, add at least one representative end-to-end workload before substantially expanding feature depth.
- Cover a common active case, an idle/quiescent case when meaningful, and a bounded scaling or stress case when their execution shapes differ.
- A locally faster microbenchmark does not justify a production change that regresses the representative workload.
- If a representative workload materially regresses and the cause is not already demonstrated by deterministic counters, profile that workload before starting a sequence of speculative micro-optimizations.

## BENCH-017 — Budget product composition separately from foundation kernels

- Product and game repositories should budget the cost of their composition: simulation ticks, AI/pathfinding, foundation calls, rendering submissions, persistence/snapshot work, and other domain orchestration.
- Do not duplicate every foundation microbenchmark in each consumer; inherit foundation correctness/performance evidence and add consumer journeys that expose pathological composition.
- Attribute work to the authoritative subsystem where practical so a product-level regression identifies whether the problem is domain orchestration, physics, rendering, networking, storage, or another foundation.
- Keep repository-specific metric names and thresholds domain-owned while using shared evidence formats and comparison policy.
