# TypeScript conventions

## TS-001 — Prefer TypeScript over JavaScript for authored code

- Use TypeScript for application code, libraries, tests, build tooling, scripts, and configuration code whenever the selected runtime or tool supports TypeScript directly or through the repository's normal build pipeline.
- Treat plain JavaScript as an exception, not a peer default. Use it only when an external tool, runtime contract, generated artifact, or interoperability fixture requires JavaScript, and keep that exception narrowly scoped and documented when it is not self-evident.
- When substantively changing an owned JavaScript module, migrate it to TypeScript in the same change when the migration is bounded and does not force unrelated architectural work.
- Do not keep parallel hand-maintained JavaScript and TypeScript implementations. If JavaScript output is required for distribution, generate it from the TypeScript source.
- TypeScript adoption must include the repository's normal validation path; merely renaming files without type-aware validation does not satisfy this rule.

## TS-002 — Model invalid states out of the type system

- Prefer types, especially discriminated unions, that make invalid combinations unrepresentable.

## TS-003 — Prefer type over interface

- Use type aliases instead of interfaces for TypeScript models and declarations.
- If interface-specific semantics such as declaration merging are genuinely required, use an explicit, narrowly scoped lint suppression rather than weakening the shared rule.
- Deterministic lint configuration: [`./TS-003.oxlint.json`](./TS-003.oxlint.json).

## TS-004 — Use strict compiler options that expose missing-state mistakes

- Enable `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `noImplicitOverride`, and `noFallthroughCasesInSwitch` by default.
- Enable unused-local and unused-parameter diagnostics for implementation code; intentional callback/adapter parameters should use the language/tooling's explicit convention rather than weakening the project globally.
- Repositories may document a narrow compatibility exception for generated code or an incompatible external tool, but the normal state is strict.

## TS-005 — Enforce exhaustive owned variants in TypeScript

- TypeScript code applies `REP-010` to application-owned enums and discriminated unions.
- Deterministic TypeScript enforcement belongs to this technology scope rather than duplicating the broader policy.

## TS-006 — Enforce owned asynchronous work in TypeScript

- TypeScript code applies `REL-006`: promises are observed unless an explicit abstraction owns detached work.
- Deterministic TypeScript enforcement belongs to this technology scope rather than duplicating the broader policy.

## TS-007 — Use structurally explicit control flow and side effects

- Require braces around control-flow bodies.
- Do not use nested ternaries, sequence expressions, chained assignments, or assignments in conditions.
- Do not place lexical declarations directly under `switch` case labels; introduce a block.
- Unmarked `switch` fallthrough is invalid; adjacent empty case labels may intentionally share a body.
- Deterministic lint configuration: [`./TS-007.oxlint.json`](./TS-007.oxlint.json).

## TS-008 — Keep imports mechanically canonical

- Combine duplicate compatible imports from the same resolved module; namespace imports may remain separate when syntax requires it.
- Prefer inline type specifiers so compatible type and value imports can share one declaration.
- Let the formatter deterministically sort ordinary imports; preserve side-effect-only import order by default.
- Deterministic configuration: [`./TS-008.oxlint.json`](./TS-008.oxlint.json) and [`./TS-008.oxfmt.json`](./TS-008.oxfmt.json).
