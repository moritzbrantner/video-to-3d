/** Summarise the Rust `colmap-surface` stage funnel for the studio status line. */
export interface DenseSurfaceDiagnostics {
  verification: {
    reference_samples: number;
    missing_reference_depth: number;
    accepted: number;
  };
  meshing: { exported_triangles: number };
  topology: { islands: number; small_islands: number; largest_island_triangles: number };
  dominant_rejection: string | null;
}

const count = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const percent = (part: number, whole: number) => `${Math.round((100 * part) / whole)}%`;

export function denseDiagnosticSummary(report: unknown): string | null {
  const diagnostics = (report as { diagnostics?: DenseSurfaceDiagnostics } | null)?.diagnostics;
  const v = diagnostics?.verification;
  const m = diagnostics?.meshing;
  const t = diagnostics?.topology;
  if (!v || !m || !t || ![v.reference_samples, v.missing_reference_depth, m.exported_triangles,
    t.islands, t.small_islands, t.largest_island_triangles].every(count) || v.reference_samples === 0
    || m.exported_triangles === 0) {
    return null;
  }
  const parts = [
    `Surface is ${t.islands.toLocaleString("en-US")} disconnected pieces` +
      ` (largest ${percent(t.largest_island_triangles, m.exported_triangles)} of triangles,` +
      ` ${t.small_islands.toLocaleString("en-US")} tiny).`,
    `Stereo gave no depth for ${percent(v.missing_reference_depth, v.reference_samples)} of reference samples.`,
  ];
  if (diagnostics.dominant_rejection) {
    parts.push(`Most rejected samples: ${diagnostics.dominant_rejection.replaceAll("_", " ")}.`);
  }
  return parts.join(" ");
}
