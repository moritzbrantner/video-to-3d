import type { LearnedProviderId } from "./catalog";
import type { LearnedProviderSession } from "./types";

export async function createLearnedProviderSession(
  providerId: LearnedProviderId,
): Promise<LearnedProviderSession> {
  switch (providerId) {
    case "depth-anything-v2-small": {
      const { createDepthAnythingSession } = await import("./depthAnything");
      return createDepthAnythingSession();
    }
    case "moge-2-vits": {
      const { createMogeSession } = await import("./moge");
      return createMogeSession();
    }
  }
}
