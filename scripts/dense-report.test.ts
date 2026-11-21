import { expect, test } from "bun:test";
import { denseDiagnosticSummary } from "./dense-report";

// The production TUM export (108 keyframes, 32 PatchMatch reference views) as reported by Rust.
const tum = {
  triangles: 108445,
  diagnostics: {
    dominant_rejection: "missing_reference_depth",
    meshing: { exported_triangles: 108445 },
    topology: { islands: 3586, small_islands: 3436, largest_island_triangles: 4160 },
    verification: { reference_samples: 590400, missing_reference_depth: 278912, accepted: 195320 },
  },
};

test("summarises fragmentation and missing stereo depth", () => {
  expect(denseDiagnosticSummary(tum)).toBe(
    "Surface is 3,586 disconnected pieces (largest 4% of triangles, 3,436 tiny). " +
      "Stereo gave no depth for 47% of reference samples. Most rejected samples: missing reference depth.",
  );
});

test("omits the summary for reports without a valid funnel", () => {
  expect(denseDiagnosticSummary({ triangles: 5 })).toBeNull();
  expect(denseDiagnosticSummary(null)).toBeNull();
  const empty = structuredClone(tum);
  empty.diagnostics.verification.reference_samples = 0;
  expect(denseDiagnosticSummary(empty)).toBeNull();
  const negative = structuredClone(tum);
  negative.diagnostics.topology.islands = -1;
  expect(denseDiagnosticSummary(negative)).toBeNull();
});

test("reports how often views disagreed before their depths were averaged", () => {
  // The fused TUM export with multi-view depth consensus (2026-11-21).
  const fused = {
    diagnostics: {
      dominant_rejection: "missing_reference_depth",
      meshing: { exported_triangles: 77061, verified_samples: 73875, consensus_moved_over_one_percent: 1595 },
      topology: { islands: 611, small_islands: 593, largest_island_triangles: 71823 },
      verification: { reference_samples: 590400, missing_reference_depth: 278912, accepted: 195320 },
    },
  };
  expect(denseDiagnosticSummary(fused)).toBe(
    "Surface is 611 disconnected pieces (largest 93% of triangles, 593 tiny). " +
      "Stereo gave no depth for 47% of reference samples. " +
      "Views disagreed by over 1% in depth on 2% of verified samples (averaged). " +
      "Most rejected samples: missing reference depth.",
  );
});
