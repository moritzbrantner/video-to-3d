import {
  loadReconstructionWasm,
  type DenseReferencePatchStats,
  type ReconstructionResult,
} from "./reconstruction";

/** RGBA pixels of one sampled frame retained as reference-appearance evidence. */
export type ReferenceImage = {
  frame_index: number;
  width: number;
  height: number;
  rgba: Uint8Array;
};

export type RecordedAppearance = {
  reference_frame: number;
  appearance_key: string;
};

export type SurfaceMaterialFallbackReasons = {
  mixed_reference: number;
  unobserved_reference: number;
  missing_reference_image: number;
  missing_grid_site: number;
  degenerate_footprint: number;
};

export type BakedReferenceMaterial = {
  reference_frame: number;
  camera_authority: string;
  source_frames: number[];
  provenance: string[];
  triangles: number[];
  appearance_key: string;
};

export type AppearanceInvalidation = {
  reused: number[];
  invalidated: number[];
  added: number[];
  removed: number[];
};

export type SurfaceMaterialBakeResult = {
  glb: Uint8Array;
  bake: {
    schema_version: number;
    point_count: number;
    triangle_count: number;
    materials: BakedReferenceMaterial[];
    fallback: { triangles: number[]; reasons: SurfaceMaterialFallbackReasons };
  };
  invalidation: AppearanceInvalidation;
  appearance: RecordedAppearance[];
  diagnostic: string;
};

export type SurfaceMaterialSource = Pick<
  ReconstructionResult,
  "dense_points" | "dense_grid_sites" | "mesh_triangles" | "accepted_camera_evidence"
> & {
  dense: { reference_patches: DenseReferencePatchStats[] };
};

const NATIVE_LITTLE_ENDIAN =
  new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

function littleEndianBytes(values: Float32Array | Uint32Array): Uint8Array {
  if (NATIVE_LITTLE_ENDIAN) {
    return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  }
  const bytes = new Uint8Array(values.byteLength);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    if (values instanceof Float32Array) view.setFloat32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return bytes;
}

/** Frames referenced by accepted dense patches; only these are retained for baking. */
export function referenceFrameIndices(source: Pick<SurfaceMaterialSource, "dense">): number[] {
  return [...new Set(source.dense.reference_patches.map((patch) => patch.reference_frame))].sort(
    (left, right) => left - right,
  );
}

export function retainReferenceImages(
  source: Pick<SurfaceMaterialSource, "dense">,
  frames: { width: number; height: number; rgba: Uint8Array }[],
): ReferenceImage[] {
  return referenceFrameIndices(source).flatMap((frame_index) => {
    const frame = frames[frame_index];
    return frame
      ? [{ frame_index, width: frame.width, height: frame.height, rgba: frame.rgba }]
      : [];
  });
}

export function buildSurfaceMaterialRequest(
  source: SurfaceMaterialSource,
  referenceImages: ReferenceImage[],
  previousAppearance: RecordedAppearance[] = [],
) {
  return {
    dense_point_count: source.dense_points.length,
    dense_points_f32_le: littleEndianBytes(source.dense_points.values),
    dense_rgb: source.dense_points.rgb,
    dense_grid_sites_u32_le: littleEndianBytes(source.dense_grid_sites.xy),
    triangle_indices_u32_le: littleEndianBytes(source.mesh_triangles.indices),
    triangle_confidence_f32_le: littleEndianBytes(source.mesh_triangles.confidence),
    reference_patches: source.dense.reference_patches.map((patch) => ({
      reference_frame: patch.reference_frame,
      source_frames: patch.source_frames,
      primary_start: patch.primary_start,
      primary_points: patch.primary_points,
      completion_start: patch.completion_start,
      completed_points: patch.completed_points,
    })),
    cameras: source.accepted_camera_evidence,
    reference_images: referenceImages,
    previous_appearance: previousAppearance,
  };
}

function plain(value: unknown): unknown {
  if (value instanceof Map) {
    return Object.fromEntries([...value].map(([key, entry]) => [key, plain(entry)]));
  }
  if (Array.isArray(value)) return value.map(plain);
  return value;
}

export function normalizeSurfaceMaterialResult(value: unknown): SurfaceMaterialBakeResult {
  const result = plain(value) as SurfaceMaterialBakeResult;
  if (
    !result ||
    !(result.glb instanceof Uint8Array) ||
    typeof result.diagnostic !== "string" ||
    !Array.isArray(result.appearance) ||
    !Array.isArray(result.bake?.materials) ||
    !Array.isArray(result.bake?.fallback?.triangles)
  ) {
    throw new Error("surface material contract mismatch: WASM returned an invalid bake");
  }
  const covered =
    result.bake.materials.reduce((sum, material) => sum + material.triangles.length, 0) +
    result.bake.fallback.triangles.length;
  if (covered !== result.bake.triangle_count) {
    throw new Error(
      "surface material contract mismatch: every accepted triangle must be textured or fall back exactly once",
    );
  }
  return result;
}

export async function bakeTexturedSurface(
  source: SurfaceMaterialSource,
  referenceImages: ReferenceImage[],
  previousAppearance: RecordedAppearance[] = [],
): Promise<SurfaceMaterialBakeResult> {
  const wasm = await loadReconstructionWasm();
  return normalizeSurfaceMaterialResult(
    wasm.bake_textured_surface(
      buildSurfaceMaterialRequest(source, referenceImages, previousAppearance),
    ),
  );
}
