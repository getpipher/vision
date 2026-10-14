/**
 * Wire formats for the vision call.
 *
 * `callVisionModel` used to POST an OpenAI-style `/chat/completions` request
 * to every provider, whatever its `api`. A model registered with
 * `google-generative-ai` or `anthropic-messages` therefore got a 404 from a
 * route that does not exist on that API. This module owns the per-API request
 * shape and response parsing so the call dispatches on `model.api` instead.
 *
 * Pure: no I/O, no pi runtime. Types only.
 *
 * Conventions follow pi-ai's own adapters:
 * - `google-generative-ai`: `baseUrl` already carries the version segment
 *   (e.g. `https://generativelanguage.googleapis.com/v1beta`).
 * - `anthropic-messages`: `baseUrl` has no version segment; the Anthropic SDK
 *   appends `/v1/messages`.
 * - `openai-responses`: `baseUrl` carries the version segment (e.g.
 *   `https://api.openai.com/v1`); `/responses` is appended.
 */
import type { Api, Model, ProviderHeaders } from "@earendil-works/pi-ai";
import type { ReasoningLevel } from "./config.ts";
import type { LoadedImage } from "./image.ts";

/** APIs the vision call can speak. Anything else is rejected before the network. */
export const SUPPORTED_VISION_APIS = [
  "openai-completions",
  "openai-responses",
  "google-generative-ai",
  "anthropic-messages",
] as const;

export type SupportedVisionApi = (typeof SUPPORTED_VISION_APIS)[number];

export function isSupportedVisionApi(api: unknown): api is SupportedVisionApi {
  return typeof api === "string" && (SUPPORTED_VISION_APIS as readonly string[]).includes(api);
}

/** Output cap shared by every API (matches the pre-existing OpenAI request). */
const MAX_OUTPUT_TOKENS = 4096;

const ANTHROPIC_VERSION = "2023-06-01";

/** Anthropic extended-thinking budgets per pi reasoning level (tokens). */
const ANTHROPIC_THINKING_BUDGETS: Record<Exclude<ReasoningLevel, "off">, number> = {
  minimal: 1024,
  low: 2048,
  medium: 8192,
  high: 16384,
  xhigh: 32768,
};

export interface VisionWireInput {
  model: Model<Api>;
  apiKey: string | undefined;
  providerHeaders: ProviderHeaders | undefined;
  image: LoadedImage;
  prompt: string;
  reasoning: ReasoningLevel;
  systemPrompt?: string | undefined;
}

export interface VisionWireRequest {
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

export interface VisionWireText {
  /** The model's answer, or undefined when the response carried none. */
  text: string | undefined;
  /** Why there is no text (block reason / finish reason), for the error message. */
  reason?: string | undefined;
}

/**
 * Actionable message for a model whose API this extension cannot call. Must not
 * match `/returned \d{3}:/` — `classifyError` would read it as an HTTP status.
 */
export function unsupportedApiMessage(model: Pick<Model<Api>, "provider" | "id" | "api">): string {
  return [
    `Vision model "${model.provider}/${model.id}" uses the "${model.api}" API, which the vision tool cannot call.`,
    `Supported APIs: ${SUPPORTED_VISION_APIS.join(", ")}.`,
    "Pick a model from a provider that uses one of them (/vision model), or register",
    "the same model under such a provider in ~/.pi/agent/models.json.",
  ].join("\n");
}

export class UnsupportedVisionApiError extends Error {
  constructor(model: Pick<Model<Api>, "provider" | "id" | "api">) {
    super(unsupportedApiMessage(model));
    this.name = "UnsupportedVisionApiError";
  }
}

function trimTrailingSlashes(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Merge provider headers over the API's own; a null value means "unset". */
function withProviderHeaders(
  base: Record<string, string>,
  providerHeaders: ProviderHeaders | undefined,
): Record<string, string> {
  const headers = { ...base };
  if (providerHeaders) {
    for (const [key, value] of Object.entries(providerHeaders)) {
      if (value !== null) headers[key] = value;
    }
  }
  return headers;
}

function openAiRequest(input: VisionWireInput): VisionWireRequest {
  const { model, apiKey, image, prompt, systemPrompt, reasoning } = input;
  const messages: unknown[] = [];
  if (systemPrompt && systemPrompt.length > 0) {
    messages.push({ role: "system", content: systemPrompt });
  }
  messages.push({
    role: "user",
    content: [
      { type: "image_url", image_url: { url: `data:${image.mimeType};base64,${image.data}` } },
      { type: "text", text: prompt },
    ],
  });
  const body: Record<string, unknown> = {
    model: model.id,
    messages,
    max_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0,
  };
  if (model.reasoning && reasoning !== "off") body.reasoning_effort = reasoning;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  return {
    url: `${trimTrailingSlashes(model.baseUrl)}/chat/completions`,
    headers: withProviderHeaders(headers, input.providerHeaders),
    body,
  };
}

/**
 * OpenAI Responses API: `{baseUrl}/responses` with `input` items. The image and
 * prompt are `input_image` / `input_text` parts, the system prompt goes in
 * `instructions`, and reasoning is the nested `reasoning.effort`.
 */
function responsesRequest(input: VisionWireInput): VisionWireRequest {
  const { model, apiKey, image, prompt, systemPrompt, reasoning } = input;
  const body: Record<string, unknown> = {
    model: model.id,
    input: [
      {
        role: "user",
        content: [
          { type: "input_image", image_url: `data:${image.mimeType};base64,${image.data}` },
          { type: "input_text", text: prompt },
        ],
      },
    ],
    max_output_tokens: MAX_OUTPUT_TOKENS,
  };
  if (systemPrompt && systemPrompt.length > 0) body.instructions = systemPrompt;
  // `temperature` is deliberately omitted: reasoning models reject it.
  if (model.reasoning && reasoning !== "off") body.reasoning = { effort: reasoning };

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  return {
    url: `${trimTrailingSlashes(model.baseUrl)}/responses`,
    headers: withProviderHeaders(headers, input.providerHeaders),
    body,
  };
}

/** Gemini `thinkingLevel` for a pi level, honouring the model's own map. */
function googleThinkingLevel(model: Model<Api>, level: ReasoningLevel): string | undefined {
  if (!model.reasoning || level === "off") return undefined;
  const mapped = model.thinkingLevelMap?.[level];
  if (mapped === null) return undefined; // the model marks this level unsupported
  if (typeof mapped === "string") return mapped;
  return level === "xhigh" ? "high" : level;
}

function googleRequest(input: VisionWireInput): VisionWireRequest {
  const { model, apiKey, image, prompt, systemPrompt, reasoning } = input;
  const generationConfig: Record<string, unknown> = {
    temperature: 0,
    maxOutputTokens: MAX_OUTPUT_TOKENS,
  };
  const thinkingLevel = googleThinkingLevel(model, reasoning);
  if (thinkingLevel) generationConfig.thinkingConfig = { thinkingLevel };

  const body: Record<string, unknown> = {
    contents: [
      {
        role: "user",
        parts: [{ inlineData: { mimeType: image.mimeType, data: image.data } }, { text: prompt }],
      },
    ],
    generationConfig,
  };
  if (systemPrompt && systemPrompt.length > 0) {
    body.systemInstruction = { parts: [{ text: systemPrompt }] };
  }

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers["x-goog-api-key"] = apiKey;
  const id = encodeURIComponent(model.id.replace(/^models\//, ""));
  return {
    url: `${trimTrailingSlashes(model.baseUrl)}/models/${id}:generateContent`,
    headers: withProviderHeaders(headers, input.providerHeaders),
    body,
  };
}

function anthropicRequest(input: VisionWireInput): VisionWireRequest {
  const { model, apiKey, image, prompt, systemPrompt, reasoning } = input;
  const body: Record<string, unknown> = {
    model: model.id,
    max_tokens: MAX_OUTPUT_TOKENS,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: image.mimeType, data: image.data } },
          { type: "text", text: prompt },
        ],
      },
    ],
  };
  if (systemPrompt && systemPrompt.length > 0) body.system = systemPrompt;
  // `temperature` is deliberately omitted: newer Claude models reject it, and
  // extended thinking requires it unset or 1.
  if (model.reasoning && reasoning !== "off") {
    const budget = ANTHROPIC_THINKING_BUDGETS[reasoning];
    body.thinking = { type: "enabled", budget_tokens: budget };
    body.max_tokens = budget + MAX_OUTPUT_TOKENS; // must exceed the thinking budget
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "anthropic-version": ANTHROPIC_VERSION,
  };
  if (apiKey) headers["x-api-key"] = apiKey;
  return {
    url: `${trimTrailingSlashes(model.baseUrl)}/v1/messages`,
    headers: withProviderHeaders(headers, input.providerHeaders),
    body,
  };
}

/** Build the HTTP request for `input.model.api`. Throws on an unsupported API. */
export function buildVisionRequest(input: VisionWireInput): VisionWireRequest {
  switch (input.model.api) {
    case "openai-completions":
      return openAiRequest(input);
    case "openai-responses":
      return responsesRequest(input);
    case "google-generative-ai":
      return googleRequest(input);
    case "anthropic-messages":
      return anthropicRequest(input);
    default:
      throw new UnsupportedVisionApiError(input.model);
  }
}

type Json = Record<string, any>;

function openAiText(json: Json): VisionWireText {
  const choice = json.choices?.[0];
  const msg = choice?.message;
  const text = msg?.content || msg?.reasoning_content;
  return { text: text || undefined, reason: choice?.finish_reason };
}

function responsesText(json: Json): VisionWireText {
  const output: Json[] = Array.isArray(json.output) ? json.output : [];
  const text = output
    .filter((item) => item.type === "message" && Array.isArray(item.content))
    .flatMap((item) => item.content as Json[])
    .filter((part) => part.type === "output_text" && typeof part.text === "string")
    .map((part) => part.text as string)
    .join("");
  return { text: text || undefined, reason: json.incomplete_details?.reason };
}

function googleText(json: Json): VisionWireText {
  const candidate = json.candidates?.[0];
  const parts: Json[] = Array.isArray(candidate?.content?.parts) ? candidate.content.parts : [];
  const joined = (wantThought: boolean) =>
    parts
      .filter((p) => typeof p.text === "string" && Boolean(p.thought) === wantThought)
      .map((p) => p.text as string)
      .join("");
  // Prefer the visible answer; fall back to thought text like the OpenAI
  // path falls back to `reasoning_content`.
  const text = joined(false) || joined(true);
  return { text: text || undefined, reason: json.promptFeedback?.blockReason ?? candidate?.finishReason };
}

function anthropicText(json: Json): VisionWireText {
  const blocks: Json[] = Array.isArray(json.content) ? json.content : [];
  const text = blocks
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("");
  return { text: text || undefined, reason: json.stop_reason };
}

/** Pull the answer text out of a parsed response body for `api`. */
export function extractVisionText(api: Api, json: unknown): VisionWireText {
  const body: Json = json && typeof json === "object" ? (json as Json) : {};
  switch (api) {
    case "google-generative-ai":
      return googleText(body);
    case "anthropic-messages":
      return anthropicText(body);
    case "openai-responses":
      return responsesText(body);
    default:
      return openAiText(body);
  }
}
