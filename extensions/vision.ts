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
 * `/vision model` with no argument opens pi's native picker
 * (`ctx.ui.select`) listing vision-capable authed models from the registry —
 * same UX quality as `/model`, scoped to vision-model selection. The typed
 * form (`/vision model <id>`) remains as a power-user fallback.
 */
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";
import { isMultimodal, syncToolAvailability, TOOL_NAME } from "../lib/capability.ts";
import {
  DEFAULT_CONFIG,
  loadConfig,
  REASONING_LEVELS,
  saveConfig,
  type ReasoningLevel,
  type VisionConfig,
} from "../lib/config.ts";
import { delegateToVisionModel, type DelegateParams } from "../lib/delegate.ts";

/** Current config. Loaded on session_start, mutated by /vision, saved to disk. */
let config: VisionConfig = { ...DEFAULT_CONFIG };

const SUBCOMMANDS = [
  "show",
  "on",
  "off",
  "provider",
  "model",
  "max-dim",
  "quality",
  "reasoning-effort",
  "clear",
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
  ].join("\n");
}

/** Re-sync tool visibility after a config change that could affect it. */
function resync(pi: ExtensionAPI, ctx: ExtensionCommandContext): void {
  syncToolAvailability(pi, ctx.model, { enabled: config.enabled });
}

/** Open pi's native picker over vision-capable authed models. Sets provider +
 *  model together. No-op (returns false) if the user cancels or there are no
 *  candidates. */
async function pickVisionModel(ctx: ExtensionCommandContext): Promise<boolean> {
  const models = ctx.modelRegistry.getAvailable().filter((m) => m.input.includes("image"));
  if (models.length === 0) {
    ctx.ui.notify(
      'No vision-capable models found. Define a model with `input: ["text","image"]` in ~/.pi/agent/models.json and configure its auth.',
      "warning",
    );
    return false;
  }
  const options = models.map((m) => `${m.provider}/${m.id}`);
  const choice = await ctx.ui.select("Pick a vision model:", options);
  if (!choice) return false; // user cancelled (or non-UI mode)
  const slash = choice.indexOf("/");
  if (slash <= 0 || slash >= choice.length - 1) return false;
  config = {
    ...config,
    provider: choice.slice(0, slash),
    model: choice.slice(slash + 1),
  };
  saveConfig(config, getAgentDir());
  return true;
}

export default function visionExtension(pi: ExtensionAPI): void {
  // ── Session lifecycle ───────────────────────────────────────────────────
  pi.on("session_start", (_event, ctx) => {
    config = loadConfig(getAgentDir());
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

      const result = await delegateToVisionModel(ctx, config, delegateParams, signal);
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
      "Configure the vision tool. `/vision model` (no arg) opens a picker of vision-capable models. Other: show, on, off, provider <p>, model <id>, max-dim <px>, quality <1-100>, reasoning-effort <level>, clear.",
    handler: async (args, ctx) => {
      const parts = args.trim().split(/\s+/).filter(Boolean);
      const sub = parts[0] ?? "show";
      const agentDir = getAgentDir();

      switch (sub) {
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
            // Interactive picker: list vision-capable authed models, set both
            // provider + model together.
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
          resync(pi, ctx);
          ctx.ui.notify("Vision config reset to defaults.", "info");
          return;
        }
        default: {
          ctx.ui.notify(
            `Unknown /vision subcommand: ${sub}\nAvailable: ${SUBCOMMANDS.join(", ")}`,
            "warning",
          );
        }
      }
    },
  });
}