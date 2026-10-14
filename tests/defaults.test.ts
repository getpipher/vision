/**
 * Unit tests for `lib/defaults.ts` (SPEC-5 §3.3 / PLAN-5 §1.4, step 2).
 *
 * `autoDetectDefaults` scans vision-capable models + picks a workflow-fit
 * primary (preferring the `Ollama` provider per AGENTS.md "Ollama Cloud
 * primary") + a frontier fallback (the first vision model under a *different*
 * provider). Pure over `Model<Api>[]` — no I/O, no pi runtime. Deterministic
 * (sorted by `(provider, id)` so the registry's iteration order doesn't
 * matter).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import type { Api, Model } from "@earendil-works/pi-ai";
import { autoDetectDefaults, PREFERRED_PRIMARY_PROVIDER } from "../lib/defaults.ts";

function makeModel(provider: string, id: string, input: ("text" | "image")[] = ["text"]): Model<Api> {
  return {
    id,
    name: id,
    provider,
    api: "openai-completions" as Api,
    reasoning: false,
    input,
    contextWindow: 200000,
    maxTokens: 4096,
  } as Model<Api>;
}

const vision = (p: string, id: string) => makeModel(p, id, ["text", "image"]);
const text = (p: string, id: string) => makeModel(p, id, ["text"]);

test("PREFERRED_PRIMARY_PROVIDER is \"Ollama\"", () => {
  assert.equal(PREFERRED_PRIMARY_PROVIDER, "Ollama");
});

test("no vision models → all undefined (no-op)", () => {
  const result = autoDetectDefaults([
    text("Ollama", "llama3.1:8b"),
    text("OpenRouter", "gpt-4o"),
  ]);
  assert.equal(result.provider, undefined);
  assert.equal(result.model, undefined);
});

test("only Ollama vision models → primary = first by sorted id, no fallback", () => {
  const result = autoDetectDefaults([
    vision("Ollama", "qwen3.5:cloud"),
    vision("Ollama", "minimax-m3:cloud"),
  ]);
  assert.equal(result.provider, "Ollama");
  assert.equal(result.model, "minimax-m3:cloud", "sorted by id: minimax < qwen");
});

test("Ollama + OpenRouter vision → primary Ollama, NO auto-fallback (v0.5.1: user sets fallback explicitly)", () => {
  const result = autoDetectDefaults([
    vision("Ollama", "minimax-m3:cloud"),
    vision("OpenRouter", "gpt-4o"),
  ]);
  assert.equal(result.provider, "Ollama");
  assert.equal(result.model, "minimax-m3:cloud");
  // v0.5.1: auto-detect returns ONLY the primary. No fallback fields.
  assert.equal("fallbackProvider" in result, false, "no fallback field in DetectedDefaults");
  assert.equal("fallbackModel" in result, false);
});

test("no Ollama vision, but OpenRouter vision → primary = first OpenRouter vision", () => {
  const result = autoDetectDefaults([
    vision("OpenRouter", "gpt-4o"),
    vision("OpenRouter", "claude-sonnet"),
  ]);
  assert.equal(result.provider, "OpenRouter");
  assert.equal(result.model, "claude-sonnet", "sorted by id: claude < gpt");
});

test("only one provider with vision → fallback undefined (no other provider)", () => {
  const result = autoDetectDefaults([
    vision("Ollama", "minimax-m3:cloud"),
    vision("Ollama", "qwen3.5:cloud"),
    text("Ollama", "glm-5.2:cloud"),
    text("OpenRouter", "gpt-4o"), // text-only OpenRouter doesn't count
  ]);
  assert.equal(result.provider, "Ollama");
  assert.equal(result.model, "minimax-m3:cloud");
});

test("three providers with vision → primary Ollama (v0.5.1: no auto-fallback)", () => {
  const result = autoDetectDefaults([
    vision("Ollama", "minimax-m3:cloud"),
    vision("OpenRouter", "gpt-4o"),
    vision("Anthropic", "claude-sonnet"),
  ]);
  assert.equal(result.provider, "Ollama");
  assert.equal(result.model, "minimax-m3:cloud");
  assert.equal("fallbackProvider" in result, false, "v0.5.1: no fallback field");
});

test("determinism: shuffled input → same output (sort normalizes)", () => {
  const models = [
    vision("Ollama", "minimax-m3:cloud"),
    vision("Ollama", "qwen3.5:cloud"),
    vision("OpenRouter", "gpt-4o"),
    text("Ollama", "glm-5.2:cloud"),
  ];
  const a = autoDetectDefaults([...models]);
  const b = autoDetectDefaults([...models].reverse());
  const c = autoDetectDefaults([models[2]!, models[0]!, models[3]!, models[1]!]);
  assert.deepEqual(a, b);
  assert.deepEqual(a, c);
  assert.equal(a.model, "minimax-m3:cloud");
});

test(":cloud preference is implicit (sort by id), not a separate filter", () => {
  // Both Ollama vision models — one :cloud, one :local. The sort by id
  // decides. This documents that the :cloud preference is aspirational, NOT
  // enforced (PLAN-5 §1.4). If RECTOR adds a local Ollama vision model he
  // doesn't want as primary, he sets it explicitly.
  const result = autoDetectDefaults([
    vision("Ollama", "minimax-m3:cloud"),
    vision("Ollama", "llava:local"),
  ]);
  // "llava:local" < "minimax-m3:cloud" lexicographically (l < m).
  assert.equal(result.model, "llava:local", "sort by id wins; :cloud not enforced");
});

test("empty input → all undefined", () => {
  const result = autoDetectDefaults([]);
  assert.equal(result.provider, undefined);
  assert.equal(result.model, undefined);
});

test("primary model without a provider field → handled (sort key empty string)", () => {
  // Defensive: a malformed model with no provider. Should not crash.
  const result = autoDetectDefaults([vision("", "lonely")]);
  assert.equal(result.provider, "");
  assert.equal(result.model, "lonely");
});

test("models on an api the vision call cannot speak are never picked (would 404)", () => {
  const onApi = (api: string, p: string, id: string) =>
    ({ ...vision(p, id), api: api as Api }) as Model<Api>;
  // Both sort before "quantforge-openai" — either used to win and 404 on the first call.
  const result = autoDetectDefaults([
    onApi("bedrock-converse-stream", "aws-bedrock", "claude-x"),
    onApi("google-vertex", "aws-vertex", "gem-x"),
    onApi("openai-completions", "quantforge-openai", "gemini-x"),
  ]);
  assert.equal(result.provider, "quantforge-openai");
  assert.equal(result.model, "gemini-x");
});

test("supported non-openai apis (google, anthropic) remain eligible", () => {
  const google = { ...vision("g", "gem"), api: "google-generative-ai" as Api } as Model<Api>;
  assert.deepEqual(autoDetectDefaults([google]), { provider: "g", model: "gem" });
  const anthropic = { ...vision("a", "claude"), api: "anthropic-messages" as Api } as Model<Api>;
  assert.deepEqual(autoDetectDefaults([anthropic]), { provider: "a", model: "claude" });
});

test("openai-responses models remain eligible", () => {
  const responses = { ...vision("r", "gpt-x"), api: "openai-responses" as Api } as Model<Api>;
  assert.deepEqual(autoDetectDefaults([responses]), { provider: "r", model: "gpt-x" });
});

test("only unsupported-api vision models → all undefined (no-op)", () => {
  const unsupported = { ...vision("p", "m"), api: "bedrock-converse-stream" as Api } as Model<Api>;
  assert.deepEqual(autoDetectDefaults([unsupported]), { provider: undefined, model: undefined });
});
