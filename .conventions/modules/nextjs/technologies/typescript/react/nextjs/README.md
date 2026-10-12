# Next.js conventions

## NEXT-001 — Prefer the donut pattern

- Keep the tree server-rendered by default; add focused client boundaries only for interactive behavior.

## NEXT-002 — Keep page-specific code local

- Keep code used by one route in its local scope; move it only when real reuse broadens ownership.
