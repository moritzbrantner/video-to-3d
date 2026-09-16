export type FeatureAlgorithm = "baseline_harris_patch" | "orb_style";

export type FeaturePoint = {
  x: number;
  y: number;
  score: number;
  scale: number;
  angle_radians: number;
};

export type FeatureMatch = {
  source_index: number;
  target_index: number;
  distance: number;
  dx: number;
  dy: number;
};

export type FeatureAnalysis = {
  algorithm: FeatureAlgorithm;
  source_features: FeaturePoint[];
  target_features: FeaturePoint[];
  matches: FeatureMatch[];
};

export type FeatureOptions = {
  max_features: number;
  min_feature_distance: number;
  descriptor_radius: number;
  match_radius: number;
  baseline_max_distance: number;
  orb_max_hamming: number;
  ratio_threshold: number;
  fast_threshold: number;
  pyramid_levels: number;
};

export const defaultFeatureOptions: FeatureOptions = {
  max_features: 400,
  min_feature_distance: 6,
  descriptor_radius: 3,
  match_radius: 48,
  baseline_max_distance: 36,
  orb_max_hamming: 82,
  ratio_threshold: 0.8,
  fast_threshold: 18,
  pyramid_levels: 4,
};

type FeatureWasmModule = {
  default: (wasmUrl: string) => Promise<unknown>;
  analyze_feature_pair: (request: unknown) => FeatureAnalysis;
};

let wasmPromise: Promise<FeatureWasmModule> | undefined;

async function loadFeatureWasm(): Promise<FeatureWasmModule> {
  if (!wasmPromise) {
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    const moduleUrl = `${basePath}/feature-wasm/video_to_3d_feature_wasm.js`;
    const importModule = new Function("url", "return import(url)") as (
      url: string,
    ) => Promise<FeatureWasmModule>;
    wasmPromise = importModule(moduleUrl).then(async (module) => {
      await module.default(
        `${basePath}/feature-wasm/video_to_3d_feature_wasm_bg.wasm`,
      );
      return module;
    });
  }
  return wasmPromise;
}

export async function analyzeFeaturePair(
  source: ImageData,
  target: ImageData,
  algorithm: FeatureAlgorithm,
  options: FeatureOptions,
): Promise<FeatureAnalysis> {
  if (source.width !== target.width || source.height !== target.height) {
    throw new Error("Source and target frames must have the same dimensions.");
  }

  const wasm = await loadFeatureWasm();
  return wasm.analyze_feature_pair({
    width: source.width,
    height: source.height,
    source_rgba: new Uint8Array(
      source.data.buffer,
      source.data.byteOffset,
      source.data.byteLength,
    ),
    target_rgba: new Uint8Array(
      target.data.buffer,
      target.data.byteOffset,
      target.data.byteLength,
    ),
    algorithm,
    options,
  });
}
