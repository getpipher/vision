/**
 * DELEGATE mode — call a configured vision model to analyze an image and
 * return a text description. Used only when the active primary model is
 * text-only (capability-aware gating in `lib/capability.ts` hides
 * `describe_image` from multimodal models, so this path never fires for
 * them under mechanism A).
 *
 * v0.2.0 (SPEC-2) adds four resilience layers:
 * 1. Caching — a content-addressed cache (`lib/cache.ts`) returns a stored
 *    description on a hit, with ZERO vision-model API calls.
 * 2. Custom system prompt — `config.systemPrompt` is prepended to the request.
 * 3. Retry + fallback — `lib/resilience.ts` retries retryable errors
 *    (5xx/429/network) with backoff, then falls back to a configured
 *    secondary vision model on failure.
 * 4. Abort-aware — `ctx.signal` stops retry + skips fallback.
 *
 * Clean-room: the OpenAI-compatible `/chat/completions` request shape with a
 * base64 data-URL image is standard API usage, not copied from pi-vision-tool.
 */
import type { Model, Api } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { isConfiguredForDelegation, loadConfig, type ReasoningLevel, type VisionConfig } from "./config.ts";
import { loadImage, type LoadedImage } from "./image.ts";
import { cacheKey, type VisionCache } from "./cache.ts";
import { AbortError, classifyError, withRetry } from "./resilience.ts";
import { appendAuditEntry, resolveAuditPath, truncateImagePathForLog, type AuditEntry } from "./audit.ts";

export interface DelegateParams {
  image_path: string;
  prompt: string;
  compress: boolean;
  reasoning: ReasoningLevel;
}

export interface DelegateSuccess {
  ok: true;
  text: string;
  details: {
    model: string;
    image_path: string;
    prompt: string;
    compressed: boolean;
    reasoning: ReasoningLevel;
    /** true if the result came from the cache (0 vision-model calls). */
    cached: boolean;
    /** true if the result came from the fallback vision model. */
    fallback: boolean;
  };
}

export interface DelegateFailure {
  ok: false;
  error: { code: string; message: string };
  /** Traceability for fallback failures: the primary error + which fallback
   *  model was attempted. */
  details?: { primaryError?: string; fallbackModel?: string };
}

export type DelegateResult = DelegateSuccess | DelegateFailure;

/** Build provider-specific reasoning params for the request body, if any. */
function buildReasoningParams(
  visionModel: Model<Api>,
  level: ReasoningLevel,
): Record<string, unknown> | undefined {
  if (!visionModel.reasoning || level === "off") return undefined;
  return { reasoning_effort: level };
}

/**
 * Call the vision model's OpenAI-compat chat/completions endpoint with the
 * image as a data URL + the user's prompt (and an optional system prompt).
 * Returns the model's text response. Exported + fetch-based so tests can mock
 * `globalThis.fetch`.
 */
export async function callVisionModel(
  visionModel: Model<Api>,
  apiKey: string | undefined,
  providerHeaders: Record<string, string> | undefined,
  image: LoadedImage,
  prompt: string,
  signal: AbortSignal | undefined,
  reasoning: ReasoningLevel,
  systemPrompt?: string,
): Promise<string> {
  const baseUrl = visionModel.baseUrl.replace(/\/+$/, "");
  const messages: unknown[] = [];
  if (systemPrompt && systemPrompt.length > 0) {
    messages.push({ role: "system", content: systemPrompt });
  }
  messages.push({
    role: "user",
    content: [
      {
        type: "image_url",
        image_url: { url: `data:${image.mimeType};base64,${image.data}` },
      },
      { type: "text", text: prompt },
    ],
  });
  const body: Record<string, unknown> = {
    model: visionModel.id,
    messages,
    max_tokens: 4096,
    temperature: 0,
  };
  const reasoningParams = buildReasoningParams(visionModel, reasoning);
  if (reasoningParams) Object.assign(body, reasoningParams);

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  if (providerHeaders) Object.assign(headers, providerHeaders);

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errBody = await response.text().catch(() => "");
    throw new Error(
      `Vision model returned ${response.status}: ${errBody.slice(0, 500)}`,
    );
  }

  const json = (await response.json()) as {
    choices?: Array<{ message?: { content?: string; reasoning_content?: string } }>;
  };
  const msg = json.choices?.[0]?.message;
  const text = msg?.content || msg?.reasoning_content;
  if (!text) {
    throw new Error("Vision model returned no content in the response");
  }
  return text;
}

function formatImageError(error: { code: string; path?: string; message?: string }, inputPath: string): string {
  switch (error.code) {
    case "not_found":
      return `Vision tool error: image not found at "${inputPath}".`;
    case "not_a_file":
      return `Vision tool error: "${inputPath}" is not a file.`;
    case "too_large":
      return `Vision tool error: image "${inputPath}" exceeds the ${Math.round(64)}MB source cap. Compress it first or pass a smaller file.`;
    case "unsupported_format":
      return `Vision tool error: could not determine the image format of "${inputPath}". Supported: PNG, JPEG, GIF, WebP.`;
    default:
      return `Vision tool error: could not read image "${inputPath}"${error.message ? `: ${error.message}` : "."}`;
  }
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Short traceability string for a primary error (used when the fallback
 *  also fails, so the caller can see why the primary was abandoned). */
function primaryErrorTag(err: unknown): string {
  const cls = classifyError(err);
  const msg = errorMessage(err).slice(0, 120);
  return `${cls}: ${msg}`;
}

const NOT_CONFIGURED_MSG = [
  "Vision tool is not configured.",
  "",
  "Use /vision to set the vision provider and model:",
  "  /vision config provider <provider>",
  "  /vision config model <model-id>",
  "",
  "The provider and model must be defined in ~/.pi/agent/models.json",
  'and the model should have `input: ["text", "image"]`.',
].join("\n");

const FALLBACK_MODEL_NOT_FOUND_MSG = (provider: string, model: string) =>
  [
    `Vision tool error: fallback model "${provider}/${model}" not found in the model registry.`,
    "",
    "Make sure the fallback provider + model are defined in ~/.pi/agent/models.json",
    'and the model has `input: ["text", "image"]`.',
    "Use /vision fallback <provider/model> to update or /vision fallback clear to remove.",
  ].join("\n");

/** Local-only mode refusal message (SPEC-5 §3.2). Cache hits still work; a
 *  cache miss refuses with this clear, actionable message. */
const LOCAL_ONLY_MSG = (cacheHint: string) =>
  [
    "Vision tool is in local-only mode — image bytes are not sent to any provider.",
    "",
    `This image has no cached description. ${cacheHint}`,
    "",
    "To delegate this image to a vision model:",
    "  /vision local-only off",
    "",
    "To inspect the cache:",
    "  /vision cache show",
  ].join("\n");

/** Write one audit entry (best-effort) if audit logging is enabled. */
function audit(config: VisionConfig, entry: AuditEntry): void {
  if (!config.auditLog) return;
  appendAuditEntry(resolveAuditPath(config.auditLogPath, getAgentDir()), entry);
}

/**
 * Run the full DELEGATE pipeline: preflight config/auth checks → load +
 * compress the image → cache check → (retry+fallback) call the vision model
 * → return its text response. Every failure returns a structured, actionable
 * error (SPEC-1 T5–T8 + SPEC-2 resilience).
 *
 * `cache` is optional: when omitted, caching is skipped (used by tests + the
 * v0.1.0 call shape). When provided + `config.cacheEnabled`, hits return
 * zero vision-model API calls.
 */
export async function delegateToVisionModel(
  ctx: ExtensionContext,
  config: VisionConfig,
  params: DelegateParams,
  signal: AbortSignal | undefined,
  cache?: VisionCache,
): Promise<DelegateResult> {
  if (!config.enabled) {
    return {
      ok: false,
      error: { code: "disabled", message: "Vision tool is disabled. Use /vision on to enable." },
    };
  }
  if (!isConfiguredForDelegation(config)) {
    return { ok: false, error: { code: "not_configured", message: NOT_CONFIGURED_MSG } };
  }

  const visionModel = ctx.modelRegistry.find(config.provider!, config.model!);
  if (!visionModel) {
    return {
      ok: false,
      error: {
        code: "model_not_found",
        message: [
          `Vision tool error: model "${config.provider}/${config.model}" not found in the model registry.`,
          "",
          "Make sure:",
          "1. The provider and model are defined in ~/.pi/agent/models.json",
          '2. The model has `input: ["text", "image"]`',
          "3. Use /vision show to check or /vision config to update the configuration",
        ].join("\n"),
      },
    };
  }

  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(visionModel);
  if (!auth.ok) {
    return {
      ok: false,
      error: {
        code: "auth_error",
        message: `Vision tool error: unable to resolve API key for "${config.provider}". ${auth.error}`,
      },
    };
  }

  const loaded = await loadImage(params.image_path, {
    compress: params.compress,
    maxDimension: config.maxDimension,
    jpegQuality: config.jpegQuality,
    cwd: ctx.cwd,
  });
  if (!loaded.ok) {
    return {
      ok: false,
      error: { code: loaded.error.code, message: formatImageError(loaded.error, params.image_path) },
    };
  }

  const modelId = `${config.provider}/${config.model}`;
  const baseDetails = {
    model: modelId,
    image_path: params.image_path,
    prompt: params.prompt,
    compressed: params.compress,
    reasoning: params.reasoning,
  };

  // ── Unified cache check + local-only gate + network call (SPEC-5 §1.6) ─
  // The two v0.4.0 branches (cache-enabled-miss + no-cache) are merged into
  // one network path so the local-only gate + the audit entry each have a
  // single insertion point. Behavior-preserving (cache-hit + cache-store-on-
  // success semantics identical to v0.4.0; T47 + T55 assert).
  const useCache = !!(cache && config.cacheEnabled);
  const key = useCache
    ? cacheKey(loaded.sourceHash, params.compress, config.maxDimension, config.jpegQuality, params.prompt, modelId, params.reasoning)
    : undefined;

  // Cache hit (allowed in local-only — the cache is local; 0 network calls).
  if (key && cache) {
    const hit = cache.get(key);
    if (hit) {
      audit(config, {
        ts: new Date().toISOString(),
        provider: config.provider ?? "(unset)",
        model: modelId,
        image_path: truncateImagePathForLog(params.image_path),
        source_hash: loaded.sourceHash,
        cached: true, fallback: false, fallback_model: undefined,
        ok: true, error_code: undefined, latency_ms: 0, local_only: config.localOnly,
      });
      return { ok: true, text: hit.text, details: { ...hit.details, ...baseDetails, cached: true, fallback: false } };
    }
  }

  // ── LOCAL-ONLY GATE (SPEC-5 §3.2) ────────────────────────────────────
  // Cache miss (or no cache) + local-only → refuse the network call. The
  // cache is local, so cache hits still work (cache-only mode above). Cache
  // miss → clear error, NO network call (structural guarantee).
  if (config.localOnly) {
    const cacheHint = config.cacheEnabled
      ? "Enable delegation (local-only off) to describe it, or re-use a previously-cached description."
      : "Enable delegation (local-only off) to describe it.";
    audit(config, {
      ts: new Date().toISOString(),
      provider: config.provider ?? "(unset)",
      model: modelId,
      image_path: truncateImagePathForLog(params.image_path),
      source_hash: loaded.sourceHash,
      cached: false, fallback: false, fallback_model: undefined,
      ok: false, error_code: "local_only", latency_ms: 0, local_only: true,
    });
    return { ok: false, error: { code: "local_only", message: LOCAL_ONLY_MSG(cacheHint) } };
  }

  // ── Network call (single path) ───────────────────────────────────────
  const t0 = performance.now();
  const result = await callWithRetryAndFallback(ctx, config, params, signal, visionModel, auth.apiKey, auth.headers, loaded.image, modelId, baseDetails);
  const latency_ms = Math.round(performance.now() - t0);

  // Cache store on success (unchanged semantics from v0.4.0).
  if (result.ok && useCache && key && cache) {
    cache.set(key, { text: result.text, details: { ...result.details, cached: false }, storedAt: Date.now() });
  }

  // ── Audit the network result (success / fallback / failure / abort) ──
  // PLAN-5 §1.6: `provider` = configured primary (the attempted route);
  // `model` = result.details.model (the responder); `fallback` +
  // `fallback_model` disambiguate. local_only is false here (the gate above
  // returned for local-only; the network path is only reached when off).
  audit(config, {
    ts: new Date().toISOString(),
    provider: config.provider ?? "(unset)",
    model: result.ok ? result.details.model : modelId,
    image_path: truncateImagePathForLog(params.image_path),
    source_hash: loaded.sourceHash,
    cached: false,
    fallback: result.ok ? result.details.fallback : false,
    fallback_model: result.ok && result.details.fallback ? result.details.model : undefined,
    ok: result.ok,
    error_code: result.ok ? undefined : result.error.code,
    latency_ms,
    local_only: false,
  });
  return result;
}

/**
 * The resilient call: retry the primary vision model with backoff, then fall
 * back to a configured secondary model on failure. Abort-aware (no retry, no
 * fallback on `ctx.signal` abort). Extracted so the cache-hit path + the
 * no-cache path share one implementation.
 */
async function callWithRetryAndFallback(
  ctx: ExtensionContext,
  config: VisionConfig,
  params: DelegateParams,
  signal: AbortSignal | undefined,
  primaryModel: Model<Api>,
  apiKey: string | undefined,
  headers: Record<string, string> | undefined,
  image: LoadedImage,
  modelId: string,
  baseDetails: Omit<DelegateSuccess["details"], "cached" | "fallback">,
): Promise<DelegateResult> {
  try {
    const text = await withRetry(
      () => callVisionModel(primaryModel, apiKey, headers, image, params.prompt, signal, params.reasoning, config.systemPrompt),
      { attempts: config.retryAttempts, backoffMs: config.retryBackoffMs, signal },
    );
    return { ok: true, text, details: { ...baseDetails, cached: false, fallback: false } };
  } catch (err) {
    if (err instanceof AbortError) {
      return { ok: false, error: { code: "aborted", message: "Vision tool aborted." } };
    }
    // Non-abort failure → try the fallback (if configured).
    if (!config.fallbackProvider || !config.fallbackModel) {
      return {
        ok: false,
        error: { code: "vision_call_error", message: `Vision tool error: ${errorMessage(err)}` },
      };
    }
    return runFallback(ctx, config, params, signal, image, err);
  }
}

/** Resolve + call the fallback vision model (one attempt, no retry). */
async function runFallback(
  ctx: ExtensionContext,
  config: VisionConfig,
  params: DelegateParams,
  signal: AbortSignal | undefined,
  image: LoadedImage,
  primaryErr: unknown,
): Promise<DelegateResult> {
  const fallbackId = `${config.fallbackProvider}/${config.fallbackModel}`;
  const fbModel = ctx.modelRegistry.find(config.fallbackProvider!, config.fallbackModel!);
  if (!fbModel) {
    return {
      ok: false,
      error: { code: "model_not_found", message: FALLBACK_MODEL_NOT_FOUND_MSG(config.fallbackProvider!, config.fallbackModel!) },
      details: { primaryError: primaryErrorTag(primaryErr), fallbackModel: fallbackId },
    };
  }
  const fbAuth = await ctx.modelRegistry.getApiKeyAndHeaders(fbModel);
  if (!fbAuth.ok) {
    return {
      ok: false,
      error: {
        code: "auth_error",
        message: `Vision tool error: unable to resolve API key for fallback "${config.fallbackProvider}". ${fbAuth.error}`,
      },
      details: { primaryError: primaryErrorTag(primaryErr), fallbackModel: fallbackId },
    };
  }
  try {
    const text = await callVisionModel(fbModel, fbAuth.apiKey, fbAuth.headers, image, params.prompt, signal, params.reasoning, config.systemPrompt);
    return {
      ok: true,
      text,
      details: {
        model: fallbackId,
        image_path: params.image_path,
        prompt: params.prompt,
        compressed: params.compress,
        reasoning: params.reasoning,
        cached: false,
        fallback: true,
      },
    };
  } catch (fbErr) {
    if (fbErr instanceof AbortError) {
      return { ok: false, error: { code: "aborted", message: "Vision tool aborted." } };
    }
    return {
      ok: false,
      error: { code: "vision_call_error", message: `Vision tool error (fallback ${fallbackId}): ${errorMessage(fbErr)}` },
      details: { primaryError: primaryErrorTag(primaryErr), fallbackModel: fallbackId },
    };
  }
}
// ── ModelRegistry-backed delegator (SPEC-2: for non-extension consumers like armory-fleet).
// Takes a ModelRegistry (the { find, getApiKeyAndHeaders } slice delegateToVisionModel reads)
// so consumers with a ModelRuntime construct `new ModelRegistry(modelRuntime)` themselves
// (ModelRegistry is exported by @earendil-works/pi-coding-agent) and pass it here.
// This keeps vision free of ModelRuntime coupling and is unit-testable with a trivial fake.

export interface ModelRegistryLike {
  find(provider: string, modelId: string): Model<Api> | undefined;
  getApiKeyAndHeaders(model: Model<Api>): Promise<unknown>;
}

export interface VisionDelegatorDeps {
  /** A ModelRegistry (or the minimal { find, getApiKeyAndHeaders } slice). */
  modelRegistry: ModelRegistryLike;
  /** The cwd for image path resolution. */
  cwd: string;
  /** The pi agent dir (where vision.json lives). */
  agentDir: string;
}

export interface VisionDelegator {
  delegate(params: DelegateParams, signal?: AbortSignal | undefined): Promise<DelegateResult>;
  config: VisionConfig;
}

export function createVisionDelegator(deps: VisionDelegatorDeps): VisionDelegator {
  const config = loadConfig(deps.agentDir);
  // delegateToVisionModel reads only ctx.modelRegistry + ctx.cwd; construct a minimal ctx.
  const ctx = { modelRegistry: deps.modelRegistry, cwd: deps.cwd } as unknown as ExtensionContext;
  return {
    config,
    delegate: (params, signal) => delegateToVisionModel(ctx, config, params, signal),
  };
}
