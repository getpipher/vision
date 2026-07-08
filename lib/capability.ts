/**
 * Capability-aware delegation core — the shared module that decides whether
 * `describe_image` should be visible to the LLM.
 *
 * The thesis (PRD §2): a multimodal primary model can process images
 * natively, so delegating to a second vision model is pure waste. We hide
 * the tool from multimodal models via `pi.setActiveTools()` so the wasted
 * call is structurally impossible, not just discouraged.
 *
 * This module is pure (no I/O, no side effects beyond the tool-visibility
 * toggle) so it can be unit-tested without a real pi runtime, and reused by
 * `paste.ts` in SPEC-3 without refactor.
 */
import type { Api, Model } from "@earendil-works/pi-ai";

/** Tool name registered with pi. Stable across versions. */
export const TOOL_NAME = "describe_image";

/** The minimal slice of `ExtensionAPI` that capability sync needs.
 *  Narrowing the input to just these two methods makes the function
 *  unit-testable without constructing a full pi runtime, and documents
 *  exactly what we rely on. `ExtensionAPI` satisfies this structurally. */
export interface ToolAvailabilityController {
  getActiveTools(): string[];
  setActiveTools(toolNames: string[]): void;
}

/**
 * Whether the given model can process images natively.
 *
 * Treats `undefined` (no model selected yet) and any model missing `"image"`
 * in its `input` array as text-only. This is the safe default: a text-only
 * model gets the delegation tool; a multimodal model gets native pass-through.
 */
export function isMultimodal(model: Model<Api> | undefined): boolean {
  return !!model?.input?.includes("image");
}

export interface CapabilitySyncOptions {
  /** Whether the vision tool is enabled in config. When false, the tool is
   *  always hidden regardless of model capability. */
  enabled: boolean;
}

/**
 * Synchronise `describe_image` visibility with the active model's capability.
 *
 * Read-merge-write against `pi.getActiveTools()` so other extensions' tools
 * are preserved (same pattern as pi's `plan-mode` example). Idempotent: a
 * no-op when the tool is already in the desired state, so it is safe to call
 * on every `session_start` and `model_select`.
 *
 * Rules:
 * - multimodal primary  → tool hidden (PASS-THROUGH; native image reasoning)
 * - text-only primary    → tool shown (when enabled) so the LLM can delegate
 * - disabled in config   → tool hidden regardless of capability
 */
export function syncToolAvailability(
  pi: ToolAvailabilityController,
  model: Model<Api> | undefined,
  options: CapabilitySyncOptions,
): void {
  const active = pi.getActiveTools();
  const hasTool = active.includes(TOOL_NAME);
  const shouldHaveTool = !isMultimodal(model) && options.enabled;

  if (shouldHaveTool && !hasTool) {
    pi.setActiveTools([...new Set([...active, TOOL_NAME])]);
  } else if (!shouldHaveTool && hasTool) {
    pi.setActiveTools(active.filter((name) => name !== TOOL_NAME));
  }
  // else: already in the desired state — leave the active set untouched so we
  // don't trigger unnecessary system-prompt rebuilds for other extensions.
}