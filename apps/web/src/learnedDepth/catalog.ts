export type LearnedProviderId = "depth-anything-v2-small" | "moge-2-vits";
export type LearnedReconstructionMode = "classic" | LearnedProviderId | "benchmark";

export const DEFAULT_LEARNED_RECONSTRUCTION_MODE: LearnedReconstructionMode = "classic";

export type LearnedProviderDescriptor = {
  id: LearnedProviderId;
  label: string;
  description: string;
  runtime: "transformers-js" | "litert-js";
  evidence: "relative_depth" | "affine_point_map";
  preferredBackend: "webgpu";
  approximateDownloadBytes: number;
  modelReference: string;
};

export const LEARNED_PROVIDER_CATALOG: readonly LearnedProviderDescriptor[] = [
  {
    id: "depth-anything-v2-small",
    label: "Depth Anything V2 Small",
    description:
      "Small relative-depth model. Depth remains provider-local until Rust aligns and revalidates it against accepted camera geometry.",
    runtime: "transformers-js",
    evidence: "relative_depth",
    preferredBackend: "webgpu",
    approximateDownloadBytes: 19_100_000,
    modelReference: "onnx-community/depth-anything-v2-small",
  },
  {
    id: "moge-2-vits",
    label: "MoGe-2 ViT-S",
    description:
      "Small monocular geometry model. The browser receives the raw affine point map; it is not promoted to calibrated geometry without focal/shift recovery and Rust validation.",
    runtime: "litert-js",
    evidence: "affine_point_map",
    preferredBackend: "webgpu",
    approximateDownloadBytes: 71_000_000,
    modelReference: "litert-community/MoGe-2-LiteRT",
  },
] as const;

export function isLearnedProviderId(value: string): value is LearnedProviderId {
  return LEARNED_PROVIDER_CATALOG.some(({ id }) => id === value);
}

export function learnedModeFromQueryParam(value: string | null): LearnedReconstructionMode {
  if (value === "classic" || value === "benchmark") return value;
  if (value !== null && isLearnedProviderId(value)) return value;
  return DEFAULT_LEARNED_RECONSTRUCTION_MODE;
}

export function learnedModeQueryParam(mode: LearnedReconstructionMode): string | null {
  return mode === DEFAULT_LEARNED_RECONSTRUCTION_MODE ? null : mode;
}

export function learnedProviderDescriptor(id: LearnedProviderId): LearnedProviderDescriptor {
  const descriptor = LEARNED_PROVIDER_CATALOG.find((candidate) => candidate.id === id);
  if (!descriptor) {
    throw new Error(`unknown learned reconstruction provider: ${id}`);
  }
  return descriptor;
}

export function learnedProvidersForMode(mode: LearnedReconstructionMode): LearnedProviderId[] {
  if (mode === "classic") return [];
  if (mode === "benchmark") return LEARNED_PROVIDER_CATALOG.map(({ id }) => id);
  return [mode];
}

export function learnedModeLabel(mode: LearnedReconstructionMode): string {
  switch (mode) {
    case "classic":
      return "Classic Rust/WASM only";
    case "benchmark":
      return "Benchmark both learned providers";
    default:
      return learnedProviderDescriptor(mode).label;
  }
}
