/**
 * Shared state between the vision + paste extensions (SPEC-3 §6.2).
 *
 * `paste.ts` needs access to the same `VisionConfig` + `VisionCache` instances
 * that `vision.ts` manages — so auto-delegation (text-only + "auto" mode)
 * reuses the exact same cache as explicit `describe_image` calls (cross-path
 * cache hits: auto-describe now → describe_image later = cache hit).
 *
 * **IMPORTANT:** pi creates a separate jiti module instance for each extension
 * (each `loadExtensionModule` call constructs a fresh `createJiti`). So
 * module-level `let _config` variables are NOT shared between vision.ts and
 * paste.ts — each gets its own copy. We use `globalThis` instead, which is
 * shared across the entire Node.js process regardless of module caching.
 *
 * `vision.ts` calls `setSharedState` on `session_start` + every mutation
 * (`applyAndSave`, `rebuildCache`). `paste.ts` reads via `getSharedConfig` /
 * `getSharedCache`. Falls back gracefully (returns undefined) if read before
 * `session_start` fires — the paste hook returns `continue` in that case.
 */
import type { VisionConfig } from "./config.ts";
import type { VisionCache } from "./cache.ts";

const GLOBAL_KEY = "__getpipher_vision_shared_state__";

interface SharedState {
  config: VisionConfig | undefined;
  cache: VisionCache | undefined;
}

/** Get (or initialize) the shared state on globalThis. */
function getState(): SharedState {
  const g = globalThis as Record<string, unknown>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = { config: undefined, cache: undefined } as SharedState;
  }
  return g[GLOBAL_KEY] as SharedState;
}

/** Set the shared config + cache refs. Called by vision.ts on session_start
 *  + after every config/cache mutation. */
export function setSharedState(config: VisionConfig, cache: VisionCache): void {
  const state = getState();
  state.config = config;
  state.cache = cache;
}

/** Get the shared config, or undefined if session_start hasn't fired yet. */
export function getSharedConfig(): VisionConfig | undefined {
  return getState().config;
}

/** Get the shared cache, or undefined if session_start hasn't fired yet. */
export function getSharedCache(): VisionCache | undefined {
  return getState().cache;
}

/** Clear the shared state (testing utility). */
export function clearSharedState(): void {
  const g = globalThis as Record<string, unknown>;
  delete g[GLOBAL_KEY];
}