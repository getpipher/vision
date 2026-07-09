/**
 * DELEGATE mode — call a configured vision model to analyze an image and
 * return a text description. Used only when the active primary model is
 * text-only (capability-aware gating in `lib/capability.ts` hides
 * `describe_image` from multimodal models, so this path never fires for
 * them under mechanism A).
 *
 * Clean-room: the OpenAI-compatible `/chat/completions` request shape with a
 * base64 data-URL image is standard API usage, not copied from pi-vision-tool.
 * v0.1.0 assumes the configured vision model exposes an OpenAI-compat
 * chat/completions endpoint (Ollama, OpenRouter, most providers do). API-type
 * awareness (anthropic-messages, etc.) is a SPEC-2 resilience concern.
 */
import type { Model, Api } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { isConfiguredForDelegation, type ReasoningLevel, type VisionConfig } from "./config.ts";
import { loadImage, type LoadedImage } from "./image.ts";

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
  };
}

export interface DelegateFailure {
  ok: false;
  error: { code: string; message: string };
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
 * image as a data URL + the user's prompt. Returns the model's text response.
 * Exported (and fetch-based) so tests can mock `globalThis.fetch`.
 */
export async function callVisionModel(
  visionModel: Model<Api>,
  apiKey: string | undefined,
  providerHeaders: Record<string, string> | undefined,
  image: LoadedImage,
  prompt: string,
  signal: AbortSignal | undefined,
  reasoning: ReasoningLevel,
): Promise<string> {
  const baseUrl = visionModel.baseUrl.replace(/\/+$/, "");
  const body: Record<string, unknown> = {
    model: visionModel.id,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: `data:${image.mimeType};base64,${image.data}` },
          },
          { type: "text", text: prompt },
        ],
      },
    ],
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

/**
 * Run the full DELEGATE pipeline: preflight config/auth checks → load +
 * compress the image → call the vision model → return its text response.
 * Every failure returns a structured, actionable error (SPEC-1 T5–T8).
 */
export async function delegateToVisionModel(
  ctx: ExtensionContext,
  config: VisionConfig,
  params: DelegateParams,
  signal: AbortSignal | undefined,
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

  try {
    const text = await callVisionModel(
      visionModel,
      auth.apiKey,
      auth.headers,
      loaded.image,
      params.prompt,
      signal,
      params.reasoning,
    );
    return {
      ok: true,
      text,
      details: {
        model: `${config.provider}/${config.model}`,
        image_path: params.image_path,
        prompt: params.prompt,
        compressed: params.compress,
        reasoning: params.reasoning,
      },
    };
  } catch (err) {
    return {
      ok: false,
      error: { code: "vision_call_error", message: `Vision tool error: ${errorMessage(err)}` },
    };
  }
}