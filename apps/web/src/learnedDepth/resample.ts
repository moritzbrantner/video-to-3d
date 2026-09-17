export function resampleRelativeDepth(
  values: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  for (const [label, value] of [
    ["source width", sourceWidth],
    ["source height", sourceHeight],
    ["target width", targetWidth],
    ["target height", targetHeight],
  ] as const) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`relative-depth ${label} must be a positive integer`);
    }
  }
  if (values.length !== sourceWidth * sourceHeight) {
    throw new Error(
      `relative-depth source has ${values.length} values for ${sourceWidth}×${sourceHeight}`,
    );
  }

  if (sourceWidth === targetWidth && sourceHeight === targetHeight) {
    return values instanceof Float32Array ? values.slice() : Float32Array.from(values);
  }

  const output = new Float32Array(targetWidth * targetHeight);
  const xScale = sourceWidth / targetWidth;
  const yScale = sourceHeight / targetHeight;

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = Math.min(
      sourceHeight - 1,
      Math.max(0, (targetY + 0.5) * yScale - 0.5),
    );
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    const yWeight = sourceY - y0;

    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const sourceX = Math.min(
        sourceWidth - 1,
        Math.max(0, (targetX + 0.5) * xScale - 0.5),
      );
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const xWeight = sourceX - x0;

      const topLeft = values[y0 * sourceWidth + x0];
      const topRight = values[y0 * sourceWidth + x1];
      const bottomLeft = values[y1 * sourceWidth + x0];
      const bottomRight = values[y1 * sourceWidth + x1];
      const top = topLeft + (topRight - topLeft) * xWeight;
      const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
      output[targetY * targetWidth + targetX] = top + (bottom - top) * yWeight;
    }
  }

  return output;
}
