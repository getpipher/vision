/**
 * Shared state between the vision + paste extensions (SPEC-3 §6.2).
 *
 * `paste.ts` needs access to the same `VisionConfig` + `VisionCache` instances
 * that `vision.ts` manages — so auto-delegation (text-only + "auto" mode)
 * reuses the exact same cache as explicit `describe_image` calls (cross-path
 * cache hits: auto-describe now → describe_image later = cache hit).
 *
 * `vision.ts` calls `setSharedState` on `session_start` + every mutation
 * (`applyAndSave`, `rebuildCache`). `paste.ts` reads via `getSharedConfig` /
 * `getSharedCache`. Falls back gracefully (returns undefined) if read before
 * `session_start` fires — the paste hook returns `continue` in that case.
 *
 * Leaf module: imports only types, no extension imports → no circular
 * dependency risk.
 */
import type { VisionConfig } from "./config.ts";
import type { VisionCache } from "./cache.ts";

let _config: VisionConfig | undefined;
let _cache: VisionCache | undefined;

/** Set the shared config + cache refs. Called by vision.ts on session_start
 *  + after every config/cache mutation. */
export function setSharedState(config: VisionConfig, cache: VisionCache): void {
  _config = config;
  _cache = cache;
}

/** Get the shared config, or undefined if session_start hasn't fired yet. */
export function getSharedConfig(): VisionConfig | undefined {
  return _config;
}

/** Get the shared cache, or undefined if session_start hasn't fired yet. */
export function getSharedCache(): VisionCache | undefined {
  return _cache;
}

/** Clear the shared state (testing utility). */
export function clearSharedState(): void {
  _config = undefined;
  _cache = undefined;
}