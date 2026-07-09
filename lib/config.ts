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

export type ReasoningLevel = ModelThinkingLevel;

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
}

export const DEFAULT_CONFIG: VisionConfig = {
  provider: undefined,
  model: undefined,
  maxDimension: 1568,
  jpegQuality: 85,
  defaultReasoningEffort: "off",
  enabled: true,
};

export const CONFIG_FILENAME = "vision.json";

/** Resolve the config file path inside a pi agent directory. */
export function configFilePath(agentDir: string): string {
  return join(agentDir, CONFIG_FILENAME);
}

function isReasoningLevel(value: unknown): value is ReasoningLevel {
  return typeof value === "string" && (REASONING_LEVELS as readonly string[]).includes(value);
}

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.round(n)));
}

/**
 * Merge a parsed partial config over the defaults, validating + clamping
 * every field so a malformed file can never produce an invalid `VisionConfig`.
 */
export function mergeConfig(partial: unknown): VisionConfig {
  const p = (partial ?? {}) as Partial<Record<string, unknown>>;
  return {
    provider: typeof p.provider === "string" && p.provider.trim().length > 0 ? p.provider.trim() : undefined,
    model: typeof p.model === "string" && p.model.trim().length > 0 ? p.model.trim() : undefined,
    maxDimension: clampInt(p.maxDimension, 1, 8000, DEFAULT_CONFIG.maxDimension),
    jpegQuality: clampInt(p.jpegQuality, 1, 100, DEFAULT_CONFIG.jpegQuality),
    defaultReasoningEffort: isReasoningLevel(p.defaultReasoningEffort)
      ? p.defaultReasoningEffort
      : DEFAULT_CONFIG.defaultReasoningEffort,
    enabled: typeof p.enabled === "boolean" ? p.enabled : DEFAULT_CONFIG.enabled,
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
    default:
      return config;
  }
}
