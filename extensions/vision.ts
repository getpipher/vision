/**
 * @getpipher/vision — vision extension entry point.
 *
 * Registers the `describe_image` tool and the `/vision` slash command, and
 * keeps the tool's visibility in sync with the active primary model's
 * capability via `lib/capability.ts` (mechanism A: `setActiveTools`).
 *
 * - Multimodal primary → `describe_image` hidden (PASS-THROUGH; native image
 *   reasoning, 0 delegation). A minimal `input` hook in `paste.ts` (B-lite)
 *   guarantees path-referenced images reach the model.
 * - Text-only primary → `describe_image` visible (DELEGATE to the configured
 *   vision model via `lib/delegate.ts`).
 *
 * `/vision` (no arg) opens an interactive settings panel built on pi-tui's
 * `SettingsList` — the same engine pi's native `/settings` uses. Arrow keys
 * navigate, Enter cycles a value or opens a sub-picker (e.g. the vision-model
 * picker), Escape exits. Changes apply live (saved to vision.json + tool
 * visibility re-synced). Non-TUI modes fall back to a text status. Typed
 * subcommands (`/vision on`, `/vision model <id>`, …) remain for power users.
 */
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext, Theme } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { join } from "node:path";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";
import {
  Container,
  type Component,
  Input,
  Key,
  SettingsList,
  type SettingItem,
  SelectItem,
  SelectList,
  Text,
} from "@earendil-works/pi-tui";
import { isMultimodal, syncToolAvailability, TOOL_NAME } from "../lib/capability.ts";
import {
  applySettingChange,
  DEFAULT_CONFIG,
  loadConfig,
  MARKER_STYLES,
  PASTE_MODES,
  REASONING_LEVELS,
  saveConfig,
  type ReasoningLevel,
  type VisionConfig,
} from "../lib/config.ts";
import { delegateToVisionModel, type DelegateParams } from "../lib/delegate.ts";
import { VisionCache } from "../lib/cache.ts";
import { setSharedState } from "../lib/state.ts";
import { createPreviewComponent, makePreviewImage, detectProtocol, formatImageMetadata } from "../lib/preview.ts";
import { matchesKey } from "@earendil-works/pi-tui";
import { loadImage } from "../lib/image.ts";

/** Current config. Loaded on session_start, mutated by /vision, saved to disk. */
let config: VisionConfig = { ...DEFAULT_CONFIG };

/** Content-addressed delegation cache. Rebuilt on session_start + when
 *  cachePersist/cacheMaxEntries change. Memory-only when cachePersist is off. */
let cache: VisionCache = new VisionCache(undefined, DEFAULT_CONFIG.cacheMaxEntries);

function rebuildCache(): void {
  const dir = config.cachePersist ? join(getAgentDir(), "vision-cache") : undefined;
  cache = new VisionCache(dir, config.cacheMaxEntries);
  setSharedState(config, cache);
}

const SUBCOMMANDS = [
  "show",
  "on",
  "off",
  "provider",
  "model",
  "max-dim",
  "quality",
  "reasoning-effort",
  "system-prompt",
  "cache",
  "fallback",
  "clear",
  "paste-mode",
  "marker-style",
  "auto-prompt",
  "preview",
] as const;

function formatConfigStatus(c: VisionConfig): string {
  return [
    "Vision tool config:",
    `  enabled:         ${c.enabled}`,
    `  provider:        ${c.provider ?? "(not set)"}`,
    `  model:           ${c.model ?? "(not set)"}`,
    `  maxDimension:    ${c.maxDimension}px`,
    `  jpegQuality:     ${c.jpegQuality}`,
    `  reasoning:       ${c.defaultReasoningEffort}`,
    `  systemPrompt:    ${c.systemPrompt ? truncatePreview(c.systemPrompt, 40) : "(none)"}`,
    `  cache:           ${c.cacheEnabled ? "on" : "off"}${c.cachePersist ? " (persisted, max " + c.cacheMaxEntries + ")" : ""}`,
    `  retry:           ${c.retryAttempts} attempts, ${c.retryBackoffMs}ms backoff`,
    `  fallback:        ${c.fallbackProvider && c.fallbackModel ? c.fallbackProvider + "/" + c.fallbackModel : "(none)"}`,
    `  markerStyle:     ${c.markerStyle}`,
    `  textOnlyPaste:   ${c.textOnlyPasteMode}`,
    `  autoPrompt:      ${c.autoDelegatePrompt ? truncatePreview(c.autoDelegatePrompt, 40) : "(default)"}`,
    `  autoTimeout:     ${c.autoDelegateTimeoutMs}ms`,
    `  composePreview:  ${c.composePreview}`,
    `  previewMaxWidth: ${c.previewMaxWidthCells} cells`,
  ].join("\n");
}

/** Truncate a string for a settings-row preview, appending an ellipsis if it overflows. */
function truncatePreview(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length <= max ? t : `${t.slice(0, max - 1)}…`;
}

/** Display string for a setting row, from the current config. */
function renderValue(id: string): string {
  switch (id) {
    case "enabled":
      return config.enabled ? "on" : "off";
    case "model":
      return config.provider && config.model ? `${config.provider}/${config.model}` : "(not set)";
    case "maxDimension":
      return `${config.maxDimension}px`;
    case "jpegQuality":
      return `${config.jpegQuality}`;
    case "reasoning":
      return config.defaultReasoningEffort;
    case "systemPrompt":
      return config.systemPrompt ? truncatePreview(config.systemPrompt, 40) : "(none)";
    case "cacheEnabled":
      return config.cacheEnabled ? "on" : "off";
    case "cachePersist":
      return config.cachePersist ? "on" : "off";
    case "cacheMaxEntries":
      return `${config.cacheMaxEntries}`;
    case "retryAttempts":
      return `${config.retryAttempts}`;
    case "retryBackoffMs":
      return `${config.retryBackoffMs}ms`;
    case "fallbackModel":
      return config.fallbackProvider && config.fallbackModel ? `${config.fallbackProvider}/${config.fallbackModel}` : "(none)";
    case "markerStyle":
      return config.markerStyle;
    case "textOnlyPasteMode":
      return config.textOnlyPasteMode;
    case "autoDelegatePrompt":
      return config.autoDelegatePrompt ? truncatePreview(config.autoDelegatePrompt, 40) : "(default)";
    case "autoDelegateTimeoutMs":
      return `${config.autoDelegateTimeoutMs}ms`;
    case "composePreview":
      return config.composePreview ? "on" : "off";
    case "previewMaxWidthCells":
      return `${config.previewMaxWidthCells}`;
    default:
      return "";
  }
}

/** Re-sync tool visibility after a config change that could affect it. */
function resync(pi: ExtensionAPI, ctx: ExtensionCommandContext): void {
  syncToolAvailability(pi, ctx.model, { enabled: config.enabled });
}

/** Apply a setting edit, persist, re-sync visibility if needed, and rebuild
 *  the cache when cache-shape fields change. */
function applyAndSave(id: string, value: string, pi: ExtensionAPI, ctx: ExtensionCommandContext): void {
  config = applySettingChange(config, id, value);
  saveConfig(config, getAgentDir());
  setSharedState(config, cache);
  if (id === "enabled" || id === "model") resync(pi, ctx);
  if (id === "cachePersist" || id === "cacheMaxEntries") rebuildCache();
}

/** Vision-capable authed models from the registry (input includes "image"). */
function visionCapableModels(ctx: ExtensionContext): Model<Api>[] {
  return ctx.modelRegistry.getAvailable().filter((m) => m.input.includes("image"));
}

/** Open pi's native select picker over vision-capable models. Sets provider +
 *  model together. Used by `/vision model` (no arg), `/vision-use`, and the
 *  `alt+shift+v` hotkey as a quick pick. */
async function pickVisionModel(ctx: ExtensionContext): Promise<boolean> {
  const models = visionCapableModels(ctx);
  if (models.length === 0) {
    ctx.ui.notify(
      'No vision-capable models found. Define a model with `input: ["text","image"]` in ~/.pi/agent/models.json and configure its auth.',
      "warning",
    );
    return false;
  }
  const options = models.map((m) => `${m.provider}/${m.id}`);
  const choice = await ctx.ui.select("Pick a vision model:", options);
  if (!choice) return false;
  const slash = choice.indexOf("/");
  if (slash <= 0 || slash >= choice.length - 1) return false;
  config = { ...config, provider: choice.slice(0, slash), model: choice.slice(slash + 1) };
  saveConfig(config, getAgentDir());
  return true;
}

/**
 * Open the interactive `/vision` settings panel (pi-tui SettingsList — the
 * same engine `/settings` uses). TUI-only; non-TUI modes fall back to text.
 */
async function showVisionSettings(pi: ExtensionAPI, ctx: ExtensionCommandContext): Promise<void> {
  if (ctx.mode !== "tui") {
    ctx.ui.notify(formatConfigStatus(config), "info");
    return;
  }
  await ctx.ui.custom<boolean>((tui, theme, _kb, done) => {
    const container = new Container();
    container.addChild(new Text(theme.fg("accent", theme.bold("Vision tool settings")), 0, 0));

    const items: SettingItem[] = [
      {
        id: "enabled",
        label: "Enabled",
        currentValue: renderValue("enabled"),
        values: ["on", "off"],
        description: "Master switch. Off → describe_image hidden + actionable error if invoked.",
      },
      {
        id: "model",
        label: "Vision model",
        currentValue: renderValue("model"),
        description: "Model to delegate to (DELEGATE mode). Must have input: [text, image]. Enter opens a picker.",
        submenu: (_cur, subDone) => buildModelSubmenu(theme, ctx, subDone),
      },
      {
        id: "maxDimension",
        label: "Max dimension",
        currentValue: renderValue("maxDimension"),
        values: ["512px", "1024px", "1568px", "2048px", "4096px"],
        description: "Max long-edge pixels for compression.",
      },
      {
        id: "jpegQuality",
        label: "JPEG quality",
        currentValue: renderValue("jpegQuality"),
        values: ["70", "80", "85", "90", "95"],
        description: "Re-encode quality (1-100).",
      },
      {
        id: "reasoning",
        label: "Reasoning effort",
        currentValue: renderValue("reasoning"),
        values: [...REASONING_LEVELS],
        description: "Default reasoning effort for delegation calls.",
      },
      // ── v0.2.0 (SPEC-2) rows ────────────────────────────────────────────
      {
        id: "systemPrompt",
        label: "System prompt",
        currentValue: renderValue("systemPrompt"),
        description: "Vision-model framing prepended to the request. Enter to edit inline (single-line). For multi-line, use /vision system-prompt.",
        submenu: (cur, subDone) => buildSystemPromptInput(cur, subDone),
      },
      {
        id: "cacheEnabled",
        label: "Caching",
        currentValue: renderValue("cacheEnabled"),
        values: ["on", "off"],
        description: "When on, identical delegation calls return a cached description (0 tokens on hit).",
      },
      {
        id: "cachePersist",
        label: "Persist cache to disk",
        currentValue: renderValue("cachePersist"),
        values: ["on", "off"],
        description: "When on, the cache survives session restarts (LRU-evicted at max entries).",
      },
      {
        id: "cacheMaxEntries",
        label: "Cache max entries",
        currentValue: renderValue("cacheMaxEntries"),
        values: ["64", "128", "256", "512", "1024"],
        description: "Max disk-cache entries before LRU eviction.",
      },
      {
        id: "retryAttempts",
        label: "Retry attempts",
        currentValue: renderValue("retryAttempts"),
        values: ["0", "1", "2", "3", "5"],
        description: "Retries after the first failure (total attempts = this + 1). Only 5xx/429/network retry.",
      },
      {
        id: "retryBackoffMs",
        label: "Retry backoff (ms)",
        currentValue: renderValue("retryBackoffMs"),
        values: ["250", "500", "1000", "2000"],
        description: "Base backoff; delay = min(backoffMs * 2^attempt, 8000ms).",
      },
      {
        id: "fallbackModel",
        label: "Fallback vision model",
        currentValue: renderValue("fallbackModel"),
        description: "Secondary vision model tried when the primary exhausts retries or fails non-retryable. Enter opens a picker.",
        submenu: (_cur, subDone) => buildModelSubmenu(theme, ctx, subDone),
      },
      // ── v0.3.0 (SPEC-3) rows ────────────────────────────────────────────
      {
        id: "markerStyle",
        label: "Marker style",
        currentValue: renderValue("markerStyle"),
        values: [...MARKER_STYLES],
        description: "Markdown style for [Image-#N] markers: code (inline code), bold, or plain.",
      },
      {
        id: "textOnlyPasteMode",
        label: "Text-only paste mode",
        currentValue: renderValue("textOnlyPasteMode"),
        values: [...PASTE_MODES],
        description: "How pasted images are handled on a text-only primary: hint (nudge to call describe_image), auto (auto-delegate), off (markers only).",
      },
      {
        id: "autoDelegatePrompt",
        label: "Auto-delegate prompt",
        currentValue: renderValue("autoDelegatePrompt"),
        description: "Generic prompt for auto-delegation in text-only + auto mode. Enter to edit inline (single-line). For multi-line, use /vision auto-prompt.",
        submenu: (cur, subDone) => buildAutoPromptInput(cur, subDone),
      },
      {
        id: "autoDelegateTimeoutMs",
        label: "Auto-delegate timeout",
        currentValue: renderValue("autoDelegateTimeoutMs"),
        values: ["10000ms", "20000ms", "30000ms", "60000ms"],
        description: "Timeout for auto-delegation in the paste hook (per-image AbortController). Falls back to hint on timeout.",
      },
      // ── v0.3.3 (SPEC-3 gap #7) rows ──────────────────────────────────────
      {
        id: "composePreview",
        label: "Compose preview",
        currentValue: renderValue("composePreview"),
        values: ["on", "off"],
        description: "When on, images preview above the editor as you type a path (WhatsApp style). Text fallback on tmux/unsupported terminals.",
      },
      {
        id: "previewMaxWidthCells",
        label: "Preview max width",
        currentValue: renderValue("previewMaxWidthCells"),
        values: ["40", "60", "80", "100", "120"],
        description: "Max width (in terminal cells) for the image preview rendering.",
      },
    ];

    const settingsList = new SettingsList(
      items,
      12,
      {
        label: (text, selected) => (selected ? theme.fg("accent", theme.bold(text)) : text),
        value: (text, selected) => (selected ? theme.fg("accent", text) : theme.fg("muted", text)),
        description: (text) => theme.fg("dim", text),
        cursor: "❯",
        hint: (text) => theme.fg("dim", text),
      },
      (id, newValue) => {
        applyAndSave(id, newValue, pi, ctx);
        settingsList.updateValue(id, renderValue(id));
      },
      () => done(true),
    );

    container.addChild(settingsList);
    container.addChild(new Text(theme.fg("dim", "↑↓ navigate • enter edit/cycle • esc done"), 0, 0));

    return {
      render(width: number) {
        return container.render(width);
      },
      invalidate() {
        container.invalidate();
      },
      handleInput(data: string) {
        settingsList.handleInput(data);
        tui.requestRender();
      },
    } as Component & { dispose?(): void };
  });
}

/** Build the vision-model sub-picker shown when Enter is pressed on the model
 *  row of the settings panel. A SelectList over vision-capable authed models. */
function buildModelSubmenu(
  theme: Theme,
  ctx: ExtensionCommandContext,
  subDone: (selectedValue?: string) => void,
): Component {
  const models = visionCapableModels(ctx);
  const items: SelectItem[] =
    models.length > 0
      ? models.map((m) => ({ value: `${m.provider}/${m.id}`, label: `${m.provider}/${m.id}` }))
      : [{ value: "", label: "(no vision-capable models — add one to models.json)" }];
  const sl = new SelectList(items, 10, {
    selectedPrefix: (text) => theme.fg("accent", text),
    selectedText: (text) => theme.fg("accent", text),
    description: (text) => theme.fg("muted", text),
    scrollInfo: (text) => theme.fg("dim", text),
    noMatch: (text) => theme.fg("warning", text),
  });
  sl.onSelect = (item) => subDone(item.value || undefined);
  sl.onCancel = () => subDone();
  return sl;
}

/** Build the single-line system-prompt editor shown when Enter is pressed on
 *  the system-prompt row. An `Input` (the same component SettingsList uses
 *  for its own search box). Empty submit clears; Escape cancels. */
function buildSystemPromptInput(
  currentValue: string,
  subDone: (selectedValue?: string) => void,
): Component {
  const input = new Input();
  input.setValue(currentValue === "(none)" ? "" : currentValue);
  input.onSubmit = (value) => subDone(value); // "" commits → applySettingChange clears
  input.onEscape = () => subDone(); // undefined → cancel (no change)
  return input;
}

/** Build the single-line auto-delegate-prompt editor (mirrors buildSystemPromptInput). */
function buildAutoPromptInput(
  currentValue: string,
  subDone: (selectedValue?: string) => void,
): Component {
  const input = new Input();
  input.setValue(currentValue === "(default)" ? "" : currentValue);
  input.onSubmit = (value) => subDone(value);
  input.onEscape = () => subDone();
  return input;
}

export default function visionExtension(pi: ExtensionAPI): void {
  // ── Session lifecycle ───────────────────────────────────────────────────
  pi.on("session_start", (_event, ctx) => {
    config = loadConfig(getAgentDir());
    rebuildCache();
    setSharedState(config, cache);
    syncToolAvailability(pi, ctx.model, { enabled: config.enabled });
  });

  // Re-sync when the user switches models mid-session (/model, Ctrl+P, restore).
  pi.on("model_select", (event) => {
    syncToolAvailability(pi, event.model, { enabled: config.enabled });
  });

  // v0.1.0 holds no background resources; shutdown is a no-op (forward-compat
  // with SPEC-2 retry/caching state).
  pi.on("session_shutdown", () => {});

  // ── describe_image tool ─────────────────────────────────────────────────
  // Always registered; visibility is gated by syncToolAvailability so
  // multimodal models never see it (0 delegation is structurally impossible).
  pi.registerTool({
    name: TOOL_NAME,
    label: "Describe Image",
    description:
      "Analyze an image file and return a text description or answer questions about it. Delegates to a configured vision model when the active primary model cannot process images natively. Accepts a file path, data URL, or raw base64.",
    promptSnippet:
      "Analyze an image file and return a text description or answer questions about it",
    promptGuidelines: [
      "Use describe_image when you need to analyze an image file and the active model cannot process images natively. describe_image delegates to a configured vision model and returns its text response.",
    ],
    parameters: Type.Object({
      image_path: Type.String({
        description: "Path to the image file, a data: URL, or raw base64 data.",
      }),
      prompt: Type.String({
        description: "What to analyze, extract, or answer about the image.",
      }),
      compress: Type.Optional(
        Type.Boolean({
          description: "Optimize (resize + re-encode) the image before delegation. Default true.",
        }),
      ),
      reasoning: Type.Optional(
        StringEnum([...REASONING_LEVELS], {
          description:
            "Reasoning effort for the delegation call. Defaults to the configured defaultReasoningEffort.",
        }),
      ),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      // Defense-in-depth: under mechanism (A) the tool is hidden from
      // multimodal models, so this branch should never fire. It handles the
      // rare race where the model switched between the LLM deciding to call
      // the tool and the tool executing.
      if (isMultimodal(ctx.model)) {
        const id = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "unknown";
        return {
          content: [
            {
              type: "text" as const,
              text: `The active primary model (${id}) can process images natively. Use the \`read\` tool to view the image, then respond directly — no delegation needed.`,
            },
          ],
          details: { mode: "passthrough_redirect", model: id },
        };
      }

      const delegateParams: DelegateParams = {
        image_path: params.image_path,
        prompt: params.prompt,
        compress: params.compress ?? true,
        reasoning: (params.reasoning ?? config.defaultReasoningEffort) as ReasoningLevel,
      };

      const result = await delegateToVisionModel(ctx, config, delegateParams, signal, cache);
      if (result.ok) {
        return {
          content: [{ type: "text" as const, text: result.text }],
          details: { mode: "delegate", ...result.details },
        };
      }
      return {
        content: [{ type: "text" as const, text: result.error.message }],
        details: { mode: "delegate", error: result.error.code },
        isError: true,
      };
    },
  });

  // ── /vision slash command ──────────────────────────────────────────────
  pi.registerCommand("vision", {
    description:
      "Open the vision settings panel (like /settings). Subcommands: show, on, off, provider <p>, model [<id>], max-dim <px>, quality <1-100>, reasoning-effort <level>, system-prompt [<text>|clear], cache <clear|show>, fallback [<provider/model>|clear>, clear.",
    handler: async (args, ctx) => {
      const parts = args.trim().split(/\s+/).filter(Boolean);
      const sub = parts[0] ?? ""; // empty → open the settings panel
      const agentDir = getAgentDir();

      switch (sub) {
        case "": {
          // /vision (no arg) → interactive settings panel (TUI) or text status.
          await showVisionSettings(pi, ctx);
          return;
        }
        case "show": {
          ctx.ui.notify(formatConfigStatus(config), "info");
          return;
        }
        case "on": {
          config = { ...config, enabled: true };
          saveConfig(config, agentDir);
          resync(pi, ctx);
          ctx.ui.notify("Vision tool enabled.", "info");
          return;
        }
        case "off": {
          config = { ...config, enabled: false };
          saveConfig(config, agentDir);
          resync(pi, ctx);
          ctx.ui.notify("Vision tool disabled. Use /vision on to re-enable.", "info");
          return;
        }
        case "provider": {
          const value = parts[1];
          if (!value) {
            ctx.ui.notify("Usage: /vision provider <name>", "warning");
            return;
          }
          config = { ...config, provider: value };
          saveConfig(config, agentDir);
          resync(pi, ctx);
          ctx.ui.notify(`Vision provider set to ${value}.`, "info");
          return;
        }
        case "model": {
          const value = parts.slice(1).join(" ").trim();
          if (!value) {
            // Quick pick via native select dialog (works in RPC too).
            const picked = await pickVisionModel(ctx);
            if (picked) {
              resync(pi, ctx);
              ctx.ui.notify(`Vision model set to ${config.provider}/${config.model}.`, "info");
            }
            return;
          }
          config = { ...config, model: value };
          saveConfig(config, agentDir);
          resync(pi, ctx);
          ctx.ui.notify(`Vision model set to ${value}.`, "info");
          return;
        }
        case "max-dim": {
          const n = Number(parts[1]);
          if (!Number.isFinite(n)) {
            ctx.ui.notify("Usage: /vision max-dim <pixels>", "warning");
            return;
          }
          config = { ...config, maxDimension: Math.min(8000, Math.max(1, Math.round(n))) };
          saveConfig(config, agentDir);
          ctx.ui.notify(`Max dimension set to ${config.maxDimension}px.`, "info");
          return;
        }
        case "quality": {
          const n = Number(parts[1]);
          if (!Number.isFinite(n)) {
            ctx.ui.notify("Usage: /vision quality <1-100>", "warning");
            return;
          }
          config = { ...config, jpegQuality: Math.min(100, Math.max(1, Math.round(n))) };
          saveConfig(config, agentDir);
          ctx.ui.notify(`JPEG quality set to ${config.jpegQuality}.`, "info");
          return;
        }
        case "reasoning-effort": {
          const raw = parts[1];
          if (!raw || !(REASONING_LEVELS as readonly string[]).includes(raw)) {
            ctx.ui.notify(
              `Usage: /vision reasoning-effort <${REASONING_LEVELS.join("|")}>`,
              "warning",
            );
            return;
          }
          config = { ...config, defaultReasoningEffort: raw as ReasoningLevel };
          saveConfig(config, agentDir);
          ctx.ui.notify(`Default reasoning effort set to ${raw}.`, "info");
          return;
        }
        case "clear": {
          config = { ...DEFAULT_CONFIG };
          saveConfig(config, agentDir);
          rebuildCache();
          resync(pi, ctx);
          ctx.ui.notify("Vision config reset to defaults.", "info");
          return;
        }
        case "system-prompt": {
          const value = parts.slice(1).join(" ").trim();
          if (!value) {
            // No arg → multi-line editor (safe: command handler, not inside ctx.ui.custom).
            if (ctx.hasUI) {
              const edited = await ctx.ui.editor("Vision system prompt", config.systemPrompt ?? "");
              if (edited === undefined) return; // cancelled
              config = { ...config, systemPrompt: edited.trim().length > 0 ? edited.trim() : undefined };
            } else {
              ctx.ui.notify("Usage: /vision system-prompt <text> (or /vision system-prompt clear)", "warning");
              return;
            }
          } else if (value === "clear") {
            config = { ...config, systemPrompt: undefined };
          } else {
            config = { ...config, systemPrompt: value };
          }
          saveConfig(config, agentDir);
          ctx.ui.notify(config.systemPrompt ? "Vision system prompt set." : "Vision system prompt cleared.", "info");
          return;
        }
        case "cache": {
          const action = parts[1];
          if (action === "clear") {
            cache.clear();
            ctx.ui.notify("Vision cache cleared (memory + disk).", "info");
          } else if (action === "show") {
            const s = cache.stats();
            ctx.ui.notify(`Vision cache: ${s.memoryEntries} memory, ${s.diskEntries} disk (max ${s.maxEntries}, persisted ${s.persisted}). Memory is session-scoped; enable \"Persist cache to disk\" for cross-session hits.`, "info");
          } else {
            ctx.ui.notify("Usage: /vision cache <clear|show>", "warning");
          }
          return;
        }
        case "fallback": {
          const value = parts.slice(1).join(" ").trim();
          if (!value) {
            ctx.ui.notify("Usage: /vision fallback <provider/model> (or /vision fallback clear)", "warning");
            return;
          }
          if (value === "clear") {
            config = { ...config, fallbackProvider: undefined, fallbackModel: undefined };
            saveConfig(config, agentDir);
            ctx.ui.notify("Fallback vision model cleared.", "info");
            return;
          }
          const slash = value.indexOf("/");
          if (slash > 0 && slash < value.length - 1) {
            config = { ...config, fallbackProvider: value.slice(0, slash), fallbackModel: value.slice(slash + 1) };
          } else {
            config = { ...config, fallbackModel: value };
          }
          saveConfig(config, agentDir);
          ctx.ui.notify(`Fallback vision model set to ${config.fallbackProvider}/${config.fallbackModel}.`, "info");
          return;
        }
        case "paste-mode": {
          const value = parts[1];
          if (!value) {
            const order = PASTE_MODES as readonly string[];
            const next = order[(order.indexOf(config.textOnlyPasteMode) + 1) % order.length] ?? "hint";
            config = applySettingChange(config, "textOnlyPasteMode", next);
          } else {
            config = applySettingChange(config, "textOnlyPasteMode", value);
          }
          saveConfig(config, agentDir);
          setSharedState(config, cache);
          ctx.ui.notify(`Text-only paste mode set to ${config.textOnlyPasteMode}.`, "info");
          return;
        }
        case "marker-style": {
          const value = parts[1];
          if (!value) {
            ctx.ui.notify(`Marker style: ${config.markerStyle}. Valid: ${MARKER_STYLES.join(", ")}`, "info");
            return;
          }
          config = applySettingChange(config, "markerStyle", value);
          saveConfig(config, agentDir);
          setSharedState(config, cache);
          ctx.ui.notify(
            config.markerStyle === value ? `Marker style set to ${value}.` : `Invalid style. Valid: ${MARKER_STYLES.join(", ")}`,
            config.markerStyle === value ? "info" : "warning",
          );
          return;
        }
        case "auto-prompt": {
          const value = parts.slice(1).join(" ").trim();
          if (!value) {
            if (ctx.hasUI) {
              const edited = await ctx.ui.editor("Auto-delegate prompt", config.autoDelegatePrompt === DEFAULT_CONFIG.autoDelegatePrompt ? "" : config.autoDelegatePrompt);
              if (edited === undefined) return;
              config = applySettingChange(config, "autoDelegatePrompt", edited);
            } else {
              ctx.ui.notify("Usage: /vision auto-prompt <text> (or /vision auto-prompt clear)", "warning");
              return;
            }
          } else if (value === "clear") {
            config = applySettingChange(config, "autoDelegatePrompt", "");
          } else {
            config = applySettingChange(config, "autoDelegatePrompt", value);
          }
          saveConfig(config, agentDir);
          setSharedState(config, cache);
          ctx.ui.notify(config.autoDelegatePrompt === DEFAULT_CONFIG.autoDelegatePrompt ? "Auto-delegate prompt reset to default." : "Auto-delegate prompt set.", "info");
          return;
        }
        case "preview": {
          const path = parts.slice(1).join(" ").trim();
          if (!path) {
            ctx.ui.notify("Usage: /vision preview <image-path>", "warning");
            return;
          }
          // Clear compose preview widget if active (prevent interference)
          if (typeof ctx.ui.setWidget === "function") {
            ctx.ui.setWidget("vision-compose-preview", undefined);
          }
          // Helpful error if the user passed a marker instead of a path
          if (path.includes("[Image-#") || path.startsWith("`")) {
            ctx.ui.notify("Vision preview: pass a file path, not a marker. Example: /vision preview /tmp/screenshot.png", "warning");
            return;
          }
          if (ctx.mode !== "tui") {
            // Non-TUI: notify metadata as text
            const loaded = await loadImage(path, { compress: false, maxDimension: 1568, jpegQuality: 85, cwd: ctx.cwd });
            if (!loaded.ok) {
              ctx.ui.notify(`Vision preview error: could not load image "${path}".`, "error");
              return;
            }
            const img = makePreviewImage(loaded.image.data, loaded.image.mimeType, path);
            ctx.ui.notify(formatImageMetadata(img, detectProtocol()), "info");
            return;
          }
          // TUI: open a custom panel with the Image component
          const loaded = await loadImage(path, { compress: false, maxDimension: 1568, jpegQuality: 85, cwd: ctx.cwd });
          if (!loaded.ok) {
            ctx.ui.notify(`Vision preview error: could not load image "${path}" (${loaded.error.code}).`, "error");
            return;
          }
          const img = makePreviewImage(loaded.image.data, loaded.image.mimeType, path);
          await ctx.ui.custom((_tui, theme, keybindings, done) => {
            const component = createPreviewComponent(img, (c: string, t: string) => theme.fg(c as any, t), config.previewMaxWidthCells);
            return {
              render(width: number) {
                return component.render(width);
              },
              invalidate() {
                component.invalidate();
              },
              handleInput(data: string) {
                // Close on Escape
                if (data === "\x1b" || matchesKey(data, "escape") || matchesKey(data, "esc")) {
                  done(undefined);
                }
              },
              dispose() {
                component.dispose?.();
              },
            } as Component & { dispose?(): void };
          });
          return;
        }
        default: {
          ctx.ui.notify(
            `Unknown /vision subcommand: ${sub}\nAvailable: ${SUBCOMMANDS.join(", ")} (or just /vision for the panel)`,
            "warning",
          );
        }
      }
    },
  });

  // ── /vision-use command + ctrl+shift+i hotkey (SPEC-2 gap #5: inline switch) ─
  // Both switch the DELEGATE vision model mid-session without the full panel.
  // Tool visibility is unaffected (it tracks the PRIMARY model's capability,
  // not the vision model) so no resync is needed. The hotkey uses ctrl+shift+i
  // (not alt+) so it works on Mac terminals where Option≠Alt by default
  ﻿//  (e.g. Ghostty macos-option-as-alt=false). Rebindable via keybindings.json.
  pi.registerCommand("vision-use", {
    description:
      "Switch the DELEGATE vision model inline. No arg → picker; <provider/model> → set directly. (Hotkey: ctrl+shift+i)",
    handler: async (args, ctx) => {
      const value = args.trim();
      if (!value) {
        const picked = await pickVisionModel(ctx);
        if (picked) ctx.ui.notify(`Vision model set to ${config.provider}/${config.model}.`, "info");
        return;
      }
      const slash = value.indexOf("/");
      if (slash > 0 && slash < value.length - 1) {
        config = { ...config, provider: value.slice(0, slash), model: value.slice(slash + 1) };
      } else {
        config = { ...config, model: value };
      }
      saveConfig(config, getAgentDir());
      ctx.ui.notify(`Vision model set to ${config.provider}/${config.model}.`, "info");
    },
  });

  pi.registerShortcut(Key.ctrlShift("i"), {
    description: "Switch vision model (inline picker)",
    handler: async (ctx) => {
      const picked = await pickVisionModel(ctx);
      if (picked) ctx.ui.notify(`Vision model set to ${config.provider}/${config.model}.`, "info");
    },
  });
}