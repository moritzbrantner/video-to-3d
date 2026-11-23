# Tooling conventions

## BUN-001 — Use Bun as the default JavaScript toolchain

- Use Bun for packages, scripts, and JavaScript/TypeScript where required tooling supports it.
- A JavaScript/TypeScript repository uses exactly one package manager and one corresponding lockfile family; do not keep npm, pnpm, Yarn, and Bun lockfiles side by side.
- Use the latest stable Bun everywhere (owner decision 2026-11-23: exact pins create churn). Do not declare `packageManager` in `package.json`, and do not pin a Bun version in `.bun-version`, `.tool-versions`, `mise.toml`, devcontainers, `engines`, or `setup-bun` inputs; CI installs `latest` (REP-008).
- Keep the committed `bun.lock` and install with `bun install --frozen-lockfile`; the lockfile, not a Bun version, makes a commit's dependencies reproducible.
- A repository may deliberately choose another package manager for a concrete compatibility/tooling reason, but that is a repository-level exception rather than a second package-manager path.

## TAILWIND-001 — Prefer Tailwind CSS when practical

- Prefer Tailwind for application styling when utility classes preserve ownership near the markup.

## TAILWIND-002 — Use semantic tokens and named variants

- Use semantic tokens and named variants for visual decisions.
- Do not default to arbitrary radii, shadows, gradients, blur, or raw palette colors.

## Child scopes

- [`vite/`](vite/)
- [`storybook/`](storybook/)
- [`playwright/`](playwright/)
- [`lighthouse/`](lighthouse/)
- [`vitest/`](vitest/)
