/**
 * Auto-detect workflow-fit vision defaults (SPEC-5 §3.3, PLAN-5 §1.4/step 2).
 *
 * On a fresh install (or `/vision clear`), the extension auto-detects the
 * vision model at `session_start` from `models.json` — preferring the
 * `Ollama` provider's vision models (Ollama Cloud, per AGENTS.md
 * "flat-rate, private, open-weight primary") for the primary, and the first
 * vision-capable model under a *different* provider for the **fallback**
 * (the frontier-escalation path per AGENTS.md "escalate to the proper
 * frontier model for that job").
 *
 * Pure over `Model<Api>[]` — no I/O, no pi runtime. Deterministic (sorted by
 * `(provider, id)` so the registry's iteration order doesn't matter).
 *
 * **The `:cloud` preference is implicit** (PLAN-5 §1.4): it falls out of the
 * vision-capable filter + the sort by id, NOT a separate enforced filter. On
 * RECTOR's machine the only Ollama vision models are `:cloud` ones, so the
 * sort yields an Ollama Cloud model. If a local Ollama vision model is added
 * that the user doesn't want as primary, they set the primary explicitly
 * (auto-detect only fires when both provider + model are unset).
 */
import type { Api, Model } from "@earendil-works/pi-ai";

export interface DetectedDefaults {
  provider: string | undefined;
  model: string | undefined;
  fallbackProvider: string | undefined;
  fallbackModel: string | undefined;
}

/** The provider id we prefer for the primary vision model (Ollama Cloud per
 *  AGENTS.md LLM-backend policy). */
export const PREFERRED_PRIMARY_PROVIDER = "Ollama";

/**
 * Scan vision-capable models + pick a workflow-fit primary + frontier fallback.
 *
 * Algorithm:
 * 1. Filter to vision-capable models (`input` includes "image").
 * 2. If none → all undefined (no-op; the existing not-configured error guides
 *    the user).
 * 3. Primary: prefer the `Ollama` provider; among those, first by sorted
 *    `(provider, id)`. If no Ollama vision model, pick the first vision
 *    model of any provider by sorted id.
 * 4. Fallback: first vision model NOT under the primary's provider (frontier
 *    escalation — a different provider's vision model). If only one provider
 *    has vision models, fallback is undefined.
 *
 * Pure + deterministic (same input in any order → same output).
 */
export function autoDetectDefaults(models: Model<Api>[]): DetectedDefaults {
  const visionModels = models.filter((m) => m.input?.includes("image"));
  if (visionModels.length === 0) {
    return {
      provider: undefined,
      model: undefined,
      fallbackProvider: undefined,
      fallbackModel: undefined,
    };
  }

  // Deterministic order: sort by provider then id.
  const sorted = [...visionModels].sort((a, b) => {
    const pa = a.provider ?? "";
    const pb = b.provider ?? "";
    return pa < pb ? -1 : pa > pb ? 1 : a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });

  // Primary: prefer the PREFERRED provider, then first by sort.
  const preferred = sorted.find((m) => m.provider === PREFERRED_PRIMARY_PROVIDER);
  const primary = preferred ?? sorted[0]!;

  // Fallback: first vision model NOT under the primary's provider.
  const fallback = sorted.find((m) => m.provider !== primary.provider);

  return {
    provider: primary.provider,
    model: primary.id,
    fallbackProvider: fallback?.provider,
    fallbackModel: fallback?.id,
  };
}