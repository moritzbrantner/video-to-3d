import type { ReconstructionResult } from "../reconstruction";

const DEFAULT_LEARNED_FRAME_CAP = 8;

function evenlyBounded(values: number[], cap: number): number[] {
  if (values.length <= cap) return values;
  if (cap <= 1) return [values[0]];

  const selected: number[] = [];
  for (let index = 0; index < cap; index += 1) {
    const position = Math.round((index * (values.length - 1)) / (cap - 1));
    const value = values[position];
    if (selected[selected.length - 1] !== value) selected.push(value);
  }
  return selected;
}

export function selectLearnedFrameIndices(
  reconstruction: ReconstructionResult,
  cap = DEFAULT_LEARNED_FRAME_CAP,
): number[] {
  if (!Number.isSafeInteger(cap) || cap <= 0) return [];

  const accepted = [
    ...reconstruction.camera_state.calibrated_seed_cameras,
    ...reconstruction.camera_state.registered_cameras,
  ];
  const unique = [...new Set(accepted.map(({ frame_index }) => frame_index))].sort(
    (left, right) => left - right,
  );
  return evenlyBounded(unique, cap);
}
