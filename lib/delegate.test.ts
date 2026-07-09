import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
  callVisionModel,
  delegateToVisionModel,
} from "./delegate.ts";
import { DEFAULT_CONFIG } from "./config.ts";

const PNG_1x1_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M8AAAMBEg1+mP0AAAAASUVORK5CYII=";
const PNG_BYTES = Buffer.from(PNG_1x1_B64, "base64");

function makeVisionModel(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    id: "minimax-m3:cloud",
    name: "MiniMax M3",
    api: "openai-completions" as Api,
    provider: "ollama",
    baseUrl: "http://localhost:11434/v1",
    reasoning: false,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 512000,
    maxTokens: 8192,
    ...overrides,
  };
}

interface FetchCall {
  url: string;
  init: RequestInit;
}

function mockFetch(response: { status: number; body: unknown }): {
  calls: FetchCall[];
  restore: () => void;
} {
  const calls: FetchCall[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    return new Response(JSON.stringify(response.body), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof globalThis.fetch;
  return { calls, restore: () => { globalThis.fetch = original; } };
}

function mockFetchError(status: number, body: string): {
  calls: FetchCall[];
  restore: () => void;
} {
  const calls: FetchCall[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    return new Response(body, { status });
  }) as typeof globalThis.fetch;
  return { calls, restore: () => { globalThis.fetch = original; } };
}

function makeCtx(opts: {
  model?: Model<Api> | undefined;
  authOk?: boolean;
  authError?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  cwd?: string;
}): ExtensionContext {
  const model = opts.model ?? makeVisionModel();
  return {
    cwd: opts.cwd ?? "/tmp",
    modelRegistry: {
      find: () => opts.model === null ? undefined : model,
      getApiKeyAndHeaders: async () =>
        opts.authOk === false
          ? { ok: false, error: opts.authError ?? "no api key" }
          : { ok: true, apiKey: opts.apiKey ?? "test-key", headers: opts.headers },
    } as unknown as ExtensionContext["modelRegistry"],
  } as unknown as ExtensionContext;
}

function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-delegate-"));
}

test("callVisionModel: sends chat/completions POST with image data URL + prompt", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a red square" } }] } });
  try {
    const text = await callVisionModel(
      makeVisionModel(),
      "key-123",
      undefined,
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "describe this",
      undefined,
      "off",
    );
    assert.equal(text, "a red square");
    assert.equal(m.calls.length, 1);
    assert.equal(m.calls[0]!.url, "http://localhost:11434/v1/chat/completions");
    const init = m.calls[0]!.init;
    assert.equal(init.method, "POST");
    const headers = init.headers as Record<string, string>;
    assert.equal(headers.Authorization, "Bearer key-123");
    assert.equal(headers["Content-Type"], "application/json");
    const body = JSON.parse(init.body as string);
    assert.equal(body.model, "minimax-m3:cloud");
    assert.equal(body.messages[0].role, "user");
    assert.equal(body.messages[0].content[0].type, "image_url");
    assert.ok(body.messages[0].content[0].image_url.url.startsWith("data:image/png;base64,"));
    assert.equal(body.messages[0].content[1].text, "describe this");
    assert.equal(body.temperature, 0);
  } finally {
    m.restore();
  }
});

test("callVisionModel: falls back to reasoning_content when content is empty", async () => {
  const m = mockFetch({
    status: 200,
    body: { choices: [{ message: { content: "", reasoning_content: "thought: it is blue" } }] },
  });
  try {
    const text = await callVisionModel(
      makeVisionModel(),
      undefined,
      undefined,
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "what color",
      undefined,
      "off",
    );
    assert.equal(text, "thought: it is blue");
  } finally {
    m.restore();
  }
});

test("callVisionModel: includes reasoning_effort when model.reasoning + level != off", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    await callVisionModel(
      makeVisionModel({ reasoning: true }),
      "k",
      undefined,
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "p",
      undefined,
      "high",
    );
    const body = JSON.parse(m.calls[0]!.init.body as string);
    assert.equal(body.reasoning_effort, "high");
  } finally {
    m.restore();
  }
});

test("callVisionModel: omits reasoning_effort when level is off", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    await callVisionModel(
      makeVisionModel({ reasoning: true }),
      "k",
      undefined,
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "p",
      undefined,
      "off",
    );
    const body = JSON.parse(m.calls[0]!.init.body as string);
    assert.equal(body.reasoning_effort, undefined);
  } finally {
    m.restore();
  }
});

test("callVisionModel: merges provider headers", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    await callVisionModel(
      makeVisionModel(),
      "k",
      { "X-Custom": "yes" },
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "p",
      undefined,
      "off",
    );
    const headers = m.calls[0]!.init.headers as Record<string, string>;
    assert.equal(headers["X-Custom"], "yes");
  } finally {
    m.restore();
  }
});

test("callVisionModel: HTTP error throws with status + body excerpt", async () => {
  const m = mockFetchError(500, "internal server boom");
  try {
    await assert.rejects(
      callVisionModel(
        makeVisionModel(),
        "k",
        undefined,
        { data: PNG_1x1_B64, mimeType: "image/png" },
        "p",
        undefined,
        "off",
      ),
      /500: internal server boom/,
    );
  } finally {
    m.restore();
  }
});

test("callVisionModel: no content in response throws", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: {} }] } });
  try {
    await assert.rejects(
      callVisionModel(
        makeVisionModel(),
        "k",
        undefined,
        { data: PNG_1x1_B64, mimeType: "image/png" },
        "p",
        undefined,
        "off",
      ),
      /no content/,
    );
  } finally {
    m.restore();
  }
});

test("delegateToVisionModel: disabled → actionable error", async () => {
  const ctx = makeCtx({});
  const r = await delegateToVisionModel(ctx, { ...DEFAULT_CONFIG, enabled: false }, {
    image_path: "/x.png", prompt: "p", compress: true, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "disabled");
    assert.match(r.error.message, /\/vision on/);
  }
});

test("delegateToVisionModel: not configured → actionable error with /vision config", async () => {
  const ctx = makeCtx({});
  const r = await delegateToVisionModel(ctx, DEFAULT_CONFIG, {
    image_path: "/x.png", prompt: "p", compress: true, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "not_configured");
    assert.match(r.error.message, /\/vision config provider/);
  }
});

test("delegateToVisionModel: model not found → actionable error", async () => {
  const ctx = makeCtx({ model: null as unknown as Model<Api> });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "bogus" };
  const r = await delegateToVisionModel(ctx, cfg, {
    image_path: "/x.png", prompt: "p", compress: true, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "model_not_found");
    assert.match(r.error.message, /models\.json/);
  }
});

test("delegateToVisionModel: auth failure → actionable error", async () => {
  const ctx = makeCtx({ authOk: false, authError: "key not set" });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
  const r = await delegateToVisionModel(ctx, cfg, {
    image_path: "/x.png", prompt: "p", compress: false, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "auth_error");
    assert.match(r.error.message, /key not set/);
  }
});

test("delegateToVisionModel: image not found → actionable error with path", async () => {
  const ctx = makeCtx({});
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
  const r = await delegateToVisionModel(ctx, cfg, {
    image_path: "/nonexistent-vision-test.png", prompt: "p", compress: false, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "not_found");
    assert.match(r.error.message, /nonexistent-vision-test\.png/);
  }
});

test("delegateToVisionModel: success → returns text + details", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a tiny image" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
    const r = await delegateToVisionModel(ctx, cfg, {
      image_path: file, prompt: "describe", compress: false, reasoning: "off",
    }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.text, "a tiny image");
      assert.equal(r.details.model, "ollama/minimax-m3:cloud");
      assert.equal(r.details.image_path, file);
      assert.equal(r.details.compressed, false);
    }
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});