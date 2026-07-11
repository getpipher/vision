/**
 * Vision tool configuration — load/save `~/.pi/agent/vision.json`.
 *
 * The config shape mirrors pi-vision-tool's (zero migration friction for
 * users who know that config surface) but lives in our own file so it does
 * not collide with the community package during transition. Per SPEC-1 §9 Q3
 * (decided 2026-07-09): no auto-migration from `vision-tool.json` — users
 * reconfigure once via `/vision config`.
 *
 * Load is fault-tolerant: a missing or malformed file yields defaults so the
 * extension always loads. Save is atomic (tmp + rename) so a crash mid-write
 * never leaves a truncated config.
 */
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ModelThinkingLevel } from "@earendil-works/pi-ai";
import type { MarkerStyle } from "./marker.ts";

export type ReasoningLevel = ModelThinkingLevel;

export const MARKER_STYLES: readonly MarkerStyle[] = ["code", "bold", "plain"] as const;

export const PASTE_MODES = ["hint", "auto", "off"] as const;
export type PasteMode = (typeof PASTE_MODES)[number];

export const DEFAULT_AUTO_DELEGATE_PROMPT =
  "Describe this image concisely, focusing on visible content, text, diagrams, and layout.";

/** Hard cap on the number of images a single `describe_image` batch call can
 *  process. A safety bound, not a workflow knob — defends against an
 *  over-eager model passing an absurd array. (SPEC-4 §3.5.) */
export const MAX_BATCH_IMAGES = 50;

export const REASONING_LEVELS: readonly ReasoningLevel[] = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
] as const;

export interface VisionConfig {
  /** Vision model provider id (must exist in models.json). Required for DELEGATE. */
  provider: string | undefined;
  /** Vision model id under the provider. Required for DELEGATE. */
  model: string | undefined;
  /** Max image dimension (long edge) in pixels for compression. */
  maxDimension: number;
  /** JPEG re-encode quality (1–100) when compression is on. */
  jpegQuality: number;
  /** Default reasoning effort for delegation calls (overridable per call). */
  defaultReasoningEffort: ReasoningLevel;
  /** Master switch. When false, describe_image is hidden + errors if invoked. */
  enabled: boolean;
  // ── v0.2.0 (SPEC-2) ──────────────────────────────────────────────────────
  /** Custom system prompt prepended to the vision-model request (undefined = none, v0.1.0 shape). */
  systemPrompt: string | undefined;
  /** When true, successful delegation results are cached (0 tokens on hit). */
  cacheEnabled: boolean;
  /** When true, the cache also persists to disk (cross-session hits, LRU-evicted). */
  cachePersist: boolean;
  /** Max entries in the disk cache before LRU eviction. */
  cacheMaxEntries: number;
  /** Number of retries after the first failure (total attempts = retryAttempts + 1). */
  retryAttempts: number;
  /** Base backoff in ms for retry; delay = min(retryBackoffMs * 2^attempt, 8000). */
  retryBackoffMs: number;
  /** Fallback vision model provider (used when the primary exhausts retries / fails non-retryable). */
  fallbackProvider: string | undefined;
  /** Fallback vision model id under fallbackProvider. */
  fallbackModel: string | undefined;
  // ── v0.3.0 (SPEC-3) ──────────────────────────────────────────────────────
  /** Markdown style for [Image-#N] markers: "code" (inline code), "bold", or "plain". */
  markerStyle: MarkerStyle;
  /** How pasted images are handled when the primary model is text-only:
   *  "hint" (default, zero-token nudge), "auto" (auto-delegate), "off" (markers only). */
  textOnlyPasteMode: PasteMode;
  /** Generic prompt for auto-delegation in text-only + "auto" mode. */
  autoDelegatePrompt: string;
  /** Timeout (ms) for auto-delegation in the input hook (own AbortController). */
  autoDelegateTimeoutMs: number;
  // ── v0.3.3 (SPEC-3 gap #7) ───────────────────────────────────────────
  /** When true, compose-time auto-preview shows images above the editor while typing. */
  composePreview: boolean;
  /** Max width (in terminal cells) for the preview rendering. */
  previewMaxWidthCells: number;
  // ── v0.4.0 (SPEC-4) ──────────────────────────────────────────────────────
  /** Max number of image delegations to run in parallel (describe_image batch
   *  + paste auto mode). 1 = serial (escape hatch). 20 = aggressive. */
  batchConcurrency: number;
}

export const DEFAULT_CONFIG: VisionConfig = {
  provider: undefined,
  model: undefined,
  maxDimension: 1568,
  jpegQuality: 85,
  defaultReasoningEffort: "off",
  enabled: true,
  // v0.2.0 defaults
  systemPrompt: undefined,
  cacheEnabled: true,
  cachePersist: false,
  cacheMaxEntries: 256,
  retryAttempts: 2,
  retryBackoffMs: 500,
  fallbackProvider: undefined,
  fallbackModel: undefined,
  // v0.3.0 defaults
  markerStyle: "code",
  textOnlyPasteMode: "hint",
  autoDelegatePrompt: DEFAULT_AUTO_DELEGATE_PROMPT,
  autoDelegateTimeoutMs: 30000,
  // v0.3.3 defaults
  composePreview: true,
  previewMaxWidthCells: 80,
  // v0.4.0 defaults
  batchConcurrency: 5,
};

export const CONFIG_FILENAME = "vision.json";

/** Resolve the config file path inside a pi agent directory. */
export function configFilePath(agentDir: string): string {
  return join(agentDir, CONFIG_FILENAME);
}

function isReasoningLevel(value: unknown): value is ReasoningLevel {
  return typeof value === "string" && (REASONING_LEVELS as readonly string[]).includes(value);
}

function isMarkerStyle(value: unknown): value is MarkerStyle {
  return typeof value === "string" && (MARKER_STYLES as readonly string[]).includes(value);
}

function isPasteMode(value: unknown): value is PasteMode {
  return typeof value === "string" && (PASTE_MODES as readonly string[]).includes(value);
}

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.round(n)));
}

/** Non-empty trimmed string → string; empty/missing → undefined. */
function strOrUndef(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

/**
 * Merge a parsed partial config over the defaults, validating + clamping
 * every field so a malformed file can never produce an invalid `VisionConfig`.
 */
export function mergeConfig(partial: unknown): VisionConfig {
  const p = (partial ?? {}) as Partial<Record<string, unknown>>;
  return {
    provider: strOrUndef(p.provider),
    model: strOrUndef(p.model),
    maxDimension: clampInt(p.maxDimension, 1, 8000, DEFAULT_CONFIG.maxDimension),
    jpegQuality: clampInt(p.jpegQuality, 1, 100, DEFAULT_CONFIG.jpegQuality),
    defaultReasoningEffort: isReasoningLevel(p.defaultReasoningEffort)
      ? p.defaultReasoningEffort
      : DEFAULT_CONFIG.defaultReasoningEffort,
    enabled: typeof p.enabled === "boolean" ? p.enabled : DEFAULT_CONFIG.enabled,
    // v0.2.0 fields
    systemPrompt: strOrUndef(p.systemPrompt),
    cacheEnabled: typeof p.cacheEnabled === "boolean" ? p.cacheEnabled : DEFAULT_CONFIG.cacheEnabled,
    cachePersist: typeof p.cachePersist === "boolean" ? p.cachePersist : DEFAULT_CONFIG.cachePersist,
    cacheMaxEntries: clampInt(p.cacheMaxEntries, 1, 10000, DEFAULT_CONFIG.cacheMaxEntries),
    retryAttempts: clampInt(p.retryAttempts, 0, 10, DEFAULT_CONFIG.retryAttempts),
    retryBackoffMs: clampInt(p.retryBackoffMs, 0, 60000, DEFAULT_CONFIG.retryBackoffMs),
    fallbackProvider: strOrUndef(p.fallbackProvider),
    fallbackModel: strOrUndef(p.fallbackModel),
    // v0.3.0 fields
    markerStyle: isMarkerStyle(p.markerStyle) ? p.markerStyle : DEFAULT_CONFIG.markerStyle,
    textOnlyPasteMode: isPasteMode(p.textOnlyPasteMode) ? p.textOnlyPasteMode : DEFAULT_CONFIG.textOnlyPasteMode,
    autoDelegatePrompt: strOrUndef(p.autoDelegatePrompt) ?? DEFAULT_CONFIG.autoDelegatePrompt,
    autoDelegateTimeoutMs: clampInt(p.autoDelegateTimeoutMs, 1000, 120000, DEFAULT_CONFIG.autoDelegateTimeoutMs),
    // v0.3.3 fields
    composePreview: typeof p.composePreview === "boolean" ? p.composePreview : DEFAULT_CONFIG.composePreview,
    previewMaxWidthCells: clampInt(p.previewMaxWidthCells, 20, 200, DEFAULT_CONFIG.previewMaxWidthCells),
    // v0.4.0 fields
    batchConcurrency: clampInt(p.batchConcurrency, 1, 20, DEFAULT_CONFIG.batchConcurrency),
  };
}

/**
 * Load config from the agent dir. Returns defaults on any read/parse error
 * (the extension must still load so `/vision config` can fix it).
 */
export function loadConfig(agentDir: string): VisionConfig {
  try {
    const raw = readFileSync(configFilePath(agentDir), "utf8");
    return mergeConfig(JSON.parse(raw));
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

/**
 * Atomically write config to the agent dir (tmp file + rename). Creates the
 * file only; the agent dir itself is assumed to exist (pi ensures it).
 */
export function saveConfig(config: VisionConfig, agentDir: string): void {
  const file = configFilePath(agentDir);
  const tmp = `${file}.tmp`;
  writeFileSync(tmp, `${JSON.stringify(config, null, 2)}\n`, "utf8");
  renameSync(tmp, file);
}

/** Whether the config has the minimum required fields for DELEGATE mode. */
export function isConfiguredForDelegation(config: VisionConfig): boolean {
  return !!config.provider && !!config.model;
}

/**
 * Apply a single settings-panel edit to a config, returning a NEW config
 * (pure — no I/O). Used by the /vision interactive settings panel; the
 * caller handles save + tool-visibility resync. Values are the display
 * strings the SettingsList cycles/submits (e.g. "on"/"off", "1568px", "85",
 * "ollama/minimax-m3:cloud", a reasoning level).
 */
export function applySettingChange(
  config: VisionConfig,
  id: string,
  value: string,
): VisionConfig {
  switch (id) {
    case "enabled":
      return { ...config, enabled: value === "on" };
    case "model": {
      // "provider/id" → set both; bare id → set model only (keeps provider)
      const slash = value.indexOf("/");
      if (slash > 0 && slash < value.length - 1) {
        return { ...config, provider: value.slice(0, slash), model: value.slice(slash + 1) };
      }
      return { ...config, model: value.length > 0 ? value : undefined };
    }
    case "maxDimension": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, maxDimension: Math.min(8000, Math.max(1, n)) };
    }
    case "jpegQuality": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, jpegQuality: Math.min(100, Math.max(1, n)) };
    }
    case "reasoning":
      if (isReasoningLevel(value)) return { ...config, defaultReasoningEffort: value };
      return config;
    // ── v0.2.0 fields ──────────────────────────────────────────────────────
    case "systemPrompt":
      // Empty string (panel Input cleared) → undefined; otherwise the typed text.
      return { ...config, systemPrompt: value.trim().length > 0 ? value.trim() : undefined };
    case "cacheEnabled":
      return { ...config, cacheEnabled: value === "on" };
    case "cachePersist":
      return { ...config, cachePersist: value === "on" };
    case "cacheMaxEntries": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, cacheMaxEntries: Math.min(10000, Math.max(1, n)) };
    }
    case "retryAttempts": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, retryAttempts: Math.min(10, Math.max(0, n)) };
    }
    case "retryBackoffMs": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, retryBackoffMs: Math.min(60000, Math.max(0, n)) };
    }
    case "fallbackModel": {
      // "provider/id" → set both; bare id → set fallbackModel only (keeps fallbackProvider)
      const slash = value.indexOf("/");
      if (slash > 0 && slash < value.length - 1) {
        return { ...config, fallbackProvider: value.slice(0, slash), fallbackModel: value.slice(slash + 1) };
      }
      return { ...config, fallbackModel: value.length > 0 ? value : undefined };
    }
    // ── v0.3.0 fields ──────────────────────────────────────────────────────
    case "markerStyle":
      if (isMarkerStyle(value)) return { ...config, markerStyle: value };
      return config;
    case "textOnlyPasteMode":
      if (isPasteMode(value)) return { ...config, textOnlyPasteMode: value };
      return config;
    case "autoDelegatePrompt":
      // Empty string → default; otherwise the typed text.
      return { ...config, autoDelegatePrompt: value.trim().length > 0 ? value.trim() : DEFAULT_CONFIG.autoDelegatePrompt };
    case "autoDelegateTimeoutMs": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, autoDelegateTimeoutMs: Math.min(120000, Math.max(1000, n)) };
    }
    case "composePreview":
      return { ...config, composePreview: value === "on" };
    case "previewMaxWidthCells": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, previewMaxWidthCells: Math.min(200, Math.max(20, n)) };
    }
    // ── v0.4.0 fields ────────────────────────────────────────────────────
    case "batchConcurrency": {
      const n = parseInt(value, 10);
      if (!Number.isFinite(n)) return config;
      return { ...config, batchConcurrency: Math.min(20, Math.max(1, n)) };
    }
    default:
      return config;
  }
}
