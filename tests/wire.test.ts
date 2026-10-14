/**
 * Unit tests for `lib/wire.ts` — the per-API request/response shapes the
 * vision call dispatches on. Pure: no fetch, no pi runtime.
 *
 * Regression context: `callVisionModel` once POSTed `/chat/completions` to
 * every provider, so a model on `google-generative-ai` or `anthropic-messages`
 * came back as an opaque 404.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import type { Api, Model } from "@earendil-works/pi-ai";
import {
  SUPPORTED_VISION_APIS,
  UnsupportedVisionApiError,
  buildVisionRequest,
  extractVisionText,
  isSupportedVisionApi,
  unsupportedApiMessage,
  type VisionWireInput,
} from "../lib/wire.ts";

function makeModel(api: string, overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    id: "m-1",
    name: "M 1",
    api: api as Api,
    provider: "prov",
    baseUrl: "https://example.test/base",
    reasoning: false,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 100000,
    maxTokens: 4096,
    ...overrides,
  };
}

function makeInput(model: Model<Api>, overrides: Partial<VisionWireInput> = {}): VisionWireInput {
  return {
    model,
    apiKey: "key-1",
    providerHeaders: undefined,
    image: { data: "QUJD", mimeType: "image/png" },
    prompt: "describe",
    reasoning: "off",
    ...overrides,
  };
}

// ── api support ──────────────────────────────────────────────────────────

test("isSupportedVisionApi: accepts the four supported APIs only", () => {
  for (const api of SUPPORTED_VISION_APIS) assert.equal(isSupportedVisionApi(api), true);
  assert.deepEqual([...SUPPORTED_VISION_APIS], [
    "openai-completions",
    "openai-responses",
    "google-generative-ai",
    "anthropic-messages",
  ]);
  for (const api of ["openai-codex-responses", "google-vertex", "bedrock-converse-stream", "", undefined, null, 5]) {
    assert.equal(isSupportedVisionApi(api), false, String(api));
  }
});

test("buildVisionRequest: unsupported api throws an actionable error (no HTTP status shape)", () => {
  const model = makeModel("bedrock-converse-stream");
  assert.throws(() => buildVisionRequest(makeInput(model)), UnsupportedVisionApiError);
  const msg = unsupportedApiMessage(model);
  assert.match(msg, /prov\/m-1/);
  assert.match(msg, /"bedrock-converse-stream"/);
  for (const api of SUPPORTED_VISION_APIS) assert.ok(msg.includes(api), api);
  // classifyError reads /returned (\d{3}):/ as an HTTP status — must not match.
  assert.doesNotMatch(msg, /returned \d{3}:/);
});

// ── openai-completions ───────────────────────────────────────────────────

test("openai: /chat/completions, Bearer auth, image_url data URL, temperature 0", () => {
  const req = buildVisionRequest(makeInput(makeModel("openai-completions", { baseUrl: "https://example.test/v1///" })));
  assert.equal(req.url, "https://example.test/v1/chat/completions");
  assert.equal(req.headers.Authorization, "Bearer key-1");
  assert.equal(req.headers["Content-Type"], "application/json");
  const body = req.body as any;
  assert.equal(body.model, "m-1");
  assert.equal(body.max_tokens, 4096);
  assert.equal(body.temperature, 0);
  assert.equal(body.messages[0].content[0].image_url.url, "data:image/png;base64,QUJD");
  assert.equal(body.messages[0].content[1].text, "describe");
  assert.equal("reasoning_effort" in body, false);
});

test("openai: system prompt first; reasoning_effort only when model.reasoning and level != off", () => {
  const withSys = buildVisionRequest(makeInput(makeModel("openai-completions"), { systemPrompt: "be terse" })).body as any;
  assert.deepEqual(withSys.messages[0], { role: "system", content: "be terse" });

  const on = buildVisionRequest(makeInput(makeModel("openai-completions", { reasoning: true }), { reasoning: "high" })).body as any;
  assert.equal(on.reasoning_effort, "high");
  const off = buildVisionRequest(makeInput(makeModel("openai-completions", { reasoning: true }), { reasoning: "off" })).body as any;
  assert.equal("reasoning_effort" in off, false);
  const notReasoning = buildVisionRequest(makeInput(makeModel("openai-completions"), { reasoning: "high" })).body as any;
  assert.equal("reasoning_effort" in notReasoning, false);
});

test("openai: no api key → no Authorization header; null provider headers are dropped, others override", () => {
  const req = buildVisionRequest(
    makeInput(makeModel("openai-completions"), {
      apiKey: undefined,
      providerHeaders: { "X-Org": "acme", "X-Unset": null, "Content-Type": "application/vnd.test+json" } as any,
    }),
  );
  assert.equal("Authorization" in req.headers, false);
  assert.equal(req.headers["X-Org"], "acme");
  assert.equal("X-Unset" in req.headers, false);
  assert.equal(req.headers["Content-Type"], "application/vnd.test+json");
});

// ── openai-responses ─────────────────────────────────────────────────────

const RESPONSES_USER_ITEM = {
  role: "user",
  content: [
    { type: "input_image", image_url: "data:image/png;base64,QUJD" },
    { type: "input_text", text: "describe" },
  ],
};

test("responses: {baseUrl}/responses, Bearer auth, input_image + input_text parts, max_output_tokens", () => {
  const req = buildVisionRequest(makeInput(makeModel("openai-responses", { baseUrl: "https://example.test/v1///" })));
  assert.equal(req.url, "https://example.test/v1/responses");
  assert.equal(req.headers.Authorization, "Bearer key-1");
  assert.equal(req.headers["Content-Type"], "application/json");
  assert.equal(req.body.model, "m-1");
  assert.equal(req.body.max_output_tokens, 4096);
  assert.deepEqual(req.body.input, [RESPONSES_USER_ITEM]);
  assert.equal("messages" in req.body, false);
  assert.equal("temperature" in req.body, false);
});

test("responses: system prompt → instructions, not an input item; reasoning is nested reasoning.effort only when model.reasoning and level != off", () => {
  const withSys = buildVisionRequest(makeInput(makeModel("openai-responses"), { systemPrompt: "be terse" })).body;
  assert.equal(withSys.instructions, "be terse");
  assert.deepEqual(withSys.input, [RESPONSES_USER_ITEM]);

  const on = buildVisionRequest(makeInput(makeModel("openai-responses", { reasoning: true }), { reasoning: "high" })).body;
  assert.deepEqual(on.reasoning, { effort: "high" });
  assert.equal("reasoning_effort" in on, false);
  const off = buildVisionRequest(makeInput(makeModel("openai-responses", { reasoning: true }), { reasoning: "off" })).body;
  assert.equal("reasoning" in off, false);
  const notReasoning = buildVisionRequest(makeInput(makeModel("openai-responses"), { reasoning: "high" })).body;
  assert.equal("reasoning" in notReasoning, false);
});

// ── google-generative-ai ─────────────────────────────────────────────────

test("google: {baseUrl}/models/{id}:generateContent, x-goog-api-key, inlineData part", () => {
  const model = makeModel("google-generative-ai", { id: "gemini-x", baseUrl: "https://gw.test/v1beta/" });
  const req = buildVisionRequest(makeInput(model));
  assert.equal(req.url, "https://gw.test/v1beta/models/gemini-x:generateContent");
  assert.equal(req.headers["x-goog-api-key"], "key-1");
  assert.equal("Authorization" in req.headers, false);
  const body = req.body as any;
  assert.deepEqual(body.contents, [
    { role: "user", parts: [{ inlineData: { mimeType: "image/png", data: "QUJD" } }, { text: "describe" }] },
  ]);
  assert.equal(body.generationConfig.temperature, 0);
  assert.equal(body.generationConfig.maxOutputTokens, 4096);
  assert.equal("thinkingConfig" in body.generationConfig, false);
  assert.equal("systemInstruction" in body, false);
});

test("google: strips a leading models/ from the id; system prompt → systemInstruction", () => {
  const model = makeModel("google-generative-ai", { id: "models/gemini-x", baseUrl: "https://gw.test/v1beta" });
  const req = buildVisionRequest(makeInput(model, { systemPrompt: "be terse" }));
  assert.equal(req.url, "https://gw.test/v1beta/models/gemini-x:generateContent");
  assert.deepEqual((req.body as any).systemInstruction, { parts: [{ text: "be terse" }] });
});

test("google: thinkingLevel honours thinkingLevelMap, xhigh → high, null → omitted, off → omitted", () => {
  const level = (model: Model<Api>, reasoning: VisionWireInput["reasoning"]) =>
    ((buildVisionRequest(makeInput(model, { reasoning })).body as any).generationConfig.thinkingConfig ?? {}).thinkingLevel;

  const plain = makeModel("google-generative-ai", { reasoning: true });
  assert.equal(level(plain, "low"), "low");
  assert.equal(level(plain, "xhigh"), "high");
  assert.equal(level(plain, "off"), undefined);

  const mapped = makeModel("google-generative-ai", { reasoning: true, thinkingLevelMap: { low: "minimal", high: null } });
  assert.equal(level(mapped, "low"), "minimal");
  assert.equal(level(mapped, "high"), undefined);

  const notReasoning = makeModel("google-generative-ai");
  assert.equal(level(notReasoning, "high"), undefined);
});

// ── anthropic-messages ───────────────────────────────────────────────────

test("anthropic: {baseUrl}/v1/messages, x-api-key + anthropic-version, base64 image block, no temperature", () => {
  const model = makeModel("anthropic-messages", { id: "claude-x", baseUrl: "https://gw.test/" });
  const req = buildVisionRequest(makeInput(model));
  assert.equal(req.url, "https://gw.test/v1/messages");
  assert.equal(req.headers["x-api-key"], "key-1");
  assert.equal(req.headers["anthropic-version"], "2023-06-01");
  assert.equal("Authorization" in req.headers, false);
  const body = req.body as any;
  assert.equal(body.model, "claude-x");
  assert.equal(body.max_tokens, 4096);
  assert.equal("temperature" in body, false);
  assert.equal("thinking" in body, false);
  assert.equal("system" in body, false);
  assert.deepEqual(body.messages[0].content, [
    { type: "image", source: { type: "base64", media_type: "image/png", data: "QUJD" } },
    { type: "text", text: "describe" },
  ]);
});

test("anthropic: system prompt → top-level system", () => {
  const body = buildVisionRequest(makeInput(makeModel("anthropic-messages"), { systemPrompt: "be terse" })).body as any;
  assert.equal(body.system, "be terse");
});

test("anthropic: reasoning on → enabled thinking, max_tokens above the budget; off/non-reasoning → none", () => {
  const on = buildVisionRequest(makeInput(makeModel("anthropic-messages", { reasoning: true }), { reasoning: "low" })).body as any;
  assert.deepEqual(on.thinking, { type: "enabled", budget_tokens: 2048 });
  assert.ok(on.max_tokens > on.thinking.budget_tokens);

  const off = buildVisionRequest(makeInput(makeModel("anthropic-messages", { reasoning: true }), { reasoning: "off" })).body as any;
  assert.equal("thinking" in off, false);
  assert.equal(off.max_tokens, 4096);

  const notReasoning = buildVisionRequest(makeInput(makeModel("anthropic-messages"), { reasoning: "high" })).body as any;
  assert.equal("thinking" in notReasoning, false);
});

// ── extractVisionText ────────────────────────────────────────────────────

test("extract openai: content, then reasoning_content, else undefined + finish_reason", () => {
  assert.equal(extractVisionText("openai-completions" as Api, { choices: [{ message: { content: "hi" } }] }).text, "hi");
  assert.equal(
    extractVisionText("openai-completions" as Api, { choices: [{ message: { content: "", reasoning_content: "think" } }] }).text,
    "think",
  );
  const none = extractVisionText("openai-completions" as Api, { choices: [{ message: {}, finish_reason: "length" }] });
  assert.equal(none.text, undefined);
  assert.equal(none.reason, "length");
});

test("extract responses: joins output_text of message items, skipping reasoning and refusal parts; incomplete → reason", () => {
  const api = "openai-responses" as Api;
  const ok = extractVisionText(api, {
    status: "completed",
    output: [
      { type: "reasoning", summary: [] },
      {
        type: "message",
        content: [
          { type: "output_text", text: "a" },
          { type: "refusal", refusal: "no" },
          { type: "output_text", text: "b" },
        ],
      },
    ],
  });
  assert.equal(ok.text, "ab");
  const incomplete = extractVisionText(api, {
    status: "incomplete",
    incomplete_details: { reason: "max_output_tokens" },
    output: [{ type: "reasoning", summary: [] }],
  });
  assert.equal(incomplete.text, undefined);
  assert.equal(incomplete.reason, "max_output_tokens");
});

test("extract google: joins visible parts, ignores thought parts, falls back to thoughts", () => {
  const api = "google-generative-ai" as Api;
  const parts = (...p: unknown[]) => ({ candidates: [{ content: { parts: p } }] });
  assert.equal(extractVisionText(api, parts({ text: "a" }, { text: "b" })).text, "ab");
  assert.equal(extractVisionText(api, parts({ text: "secret", thought: true }, { text: "answer" })).text, "answer");
  assert.equal(extractVisionText(api, parts({ text: "only thought", thought: true })).text, "only thought");
});

test("extract google: blocked prompt / empty candidate → undefined with the reason", () => {
  const api = "google-generative-ai" as Api;
  const blocked = extractVisionText(api, { promptFeedback: { blockReason: "SAFETY" } });
  assert.equal(blocked.text, undefined);
  assert.equal(blocked.reason, "SAFETY");
  const maxTokens = extractVisionText(api, { candidates: [{ finishReason: "MAX_TOKENS", content: { parts: [] } }] });
  assert.equal(maxTokens.text, undefined);
  assert.equal(maxTokens.reason, "MAX_TOKENS");
});

test("extract anthropic: joins text blocks, skips thinking blocks; none → undefined + stop_reason", () => {
  const api = "anthropic-messages" as Api;
  const ok = extractVisionText(api, {
    content: [{ type: "thinking", thinking: "hmm" }, { type: "text", text: "a" }, { type: "text", text: "b" }],
    stop_reason: "end_turn",
  });
  assert.equal(ok.text, "ab");
  const none = extractVisionText(api, { content: [{ type: "thinking", thinking: "hmm" }], stop_reason: "max_tokens" });
  assert.equal(none.text, undefined);
  assert.equal(none.reason, "max_tokens");
});

test("extract: non-object body never throws", () => {
  for (const api of SUPPORTED_VISION_APIS) {
    assert.equal(extractVisionText(api as Api, null).text, undefined);
    assert.equal(extractVisionText(api as Api, "oops").text, undefined);
  }
});
