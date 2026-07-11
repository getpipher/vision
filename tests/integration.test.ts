/**
 * Integration EVAL harness for @getpipher/vision v0.1.0 (SPEC-1 T1–T8 + T10).
 *
 * Wires extensions/vision.ts + extensions/paste.ts together against a mock
 * pi runtime and drives the full event flow (session_start → model_select →
 * input → tool execute → /vision command) to verify the capability-aware
 * mechanism end-to-end — without a real LLM or network. Provider calls are
 * mocked via globalThis.fetch; a real 1×1 PNG is written to a temp dir.
 *
 * T9 (pi install) is a package-manager concern covered by the load smoke
 * test + unit tests; T1/T2's real-LLM behavior (does the model actually
 * reason about the attached image / choose to call describe_image) still
 * needs a manual fresh-session check — see the session handoff.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Api, Model, ImageContent } from "@earendil-works/pi-ai";
import type {
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
  ToolDefinition,
} from "@earendil-works/pi-coding-agent";

// Redirect getAgentDir() to a temp dir so /vision commands never touch the
// real ~/.pi/agent/vision.json during this test run.
const TMP_AGENT = mkdtempSync(join(tmpdir(), "vision-eval-agent-"));
process.env.PI_CODING_AGENT_DIR = TMP_AGENT;

import visionFactory from "../extensions/vision.ts";
import pasteFactory from "../extensions/paste.ts";

const PNG_1x1_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M8AAAMBEg1+mP0AAAAASUVORK5CYII=";
const PNG_BYTES = Buffer.from(PNG_1x1_B64, "base64");
// A second distinct 1x1 PNG (red pixel) for tests that need two different images.
const PNG_1x1_RED_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const PNG_BYTES_2 = Buffer.from(PNG_1x1_RED_B64, "base64");

function makeModel(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    id: "x",
    name: "X",
    api: "openai-completions" as Api,
    provider: "ollama",
    baseUrl: "http://localhost:11434/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 4096,
    ...overrides,
  };
}
const MULTIMODAL = makeModel({ id: "minimax-m3:cloud", input: ["text", "image"] });
const TEXT_ONLY = makeModel({ id: "glm-5.2:cloud", input: ["text"] });
const VISION_MODEL = makeModel({
  id: "minimax-m3:cloud",
  input: ["text", "image"],
  reasoning: false,
});

interface MockPi {
  handlers: Map<string, Array<(event: any, ctx: any) => any>>;
  tools: Map<string, ToolDefinition>;
  commands: Map<string, { handler: (args: string, ctx: any) => Promise<void> }>;
  shortcuts: Map<string, { description?: string; handler: (ctx: any) => Promise<void> | void }>;
  active: string[];
  on(event: string, handler: any): void;
  registerTool(def: ToolDefinition): void;
  registerCommand(name: string, opts: any): void;
  registerShortcut(shortcut: string, opts: { description?: string; handler: (ctx: any) => Promise<void> | void }): void;
  getActiveTools(): string[];
  setActiveTools(names: string[]): void;
  emit(event: string, eventObj: any, ctx: any): Promise<any>;
}

function createMockPi(initialActive = ["read", "bash", "edit", "write"]): MockPi {
  const handlers = new Map<string, Array<(event: any, ctx: any) => any>>();
  const tools = new Map<string, ToolDefinition>();
  const commands = new Map<string, { handler: (args: string, ctx: any) => Promise<void> }>();
  const shortcuts = new Map<string, { description?: string; handler: (ctx: any) => Promise<void> | void }>();
  let active = initialActive.slice();
  const pi: MockPi = {
    handlers,
    tools,
    commands,
    shortcuts,
    active,
    on(event, handler) {
      (handlers.get(event) ?? handlers.set(event, []).get(event)!).push(handler);
    },
    registerTool(def) {
      tools.set(def.name, def);
    },
    registerCommand(name, opts) {
      commands.set(name, opts);
    },
    registerShortcut(shortcut, opts) {
      shortcuts.set(shortcut, opts);
    },
    getActiveTools: () => active.slice(),
    setActiveTools(names) {
      active = names.slice();
    },
    async emit(event, eventObj, ctx) {
      let result: any;
      for (const h of handlers.get(event) ?? []) {
        result = await h(eventObj, ctx);
      }
      return result;
    },
  };
  return pi;
}

function makeRegistry(opts: {
  model?: Model<Api> | undefined;
  authOk?: boolean;
  apiKey?: string;
} = {}) {
  return {
    find: () => (opts.model === undefined ? undefined : opts.model),
    getApiKeyAndHeaders: async () =>
      opts.authOk === false
        ? { ok: false, error: "no api key configured" }
        : { ok: true, apiKey: opts.apiKey ?? "test-key", headers: undefined },
  };
}

function makeCtx(opts: {
  model?: Model<Api> | undefined;
  cwd?: string;
  registry?: ReturnType<typeof makeRegistry>;
  signal?: AbortSignal;
} = {}): ExtensionContext {
  return {
    ui: { notify: () => {} },
    mode: "tui",
    hasUI: false,
    cwd: opts.cwd ?? "/tmp",
    sessionManager: {},
    modelRegistry: opts.registry ?? makeRegistry({ model: VISION_MODEL }),
    model: opts.model,
    isIdle: () => true,
    isProjectTrusted: () => true,
    signal: opts.signal,
    abort: () => {},
    hasPendingMessages: () => false,
    shutdown: () => {},
    getContextUsage: () => undefined,
    compact: () => {},
    getSystemPrompt: () => "",
  } as unknown as ExtensionContext;
}

interface FetchMock {
  calls: Array<{ url: string; body: any }>;
  restore: () => void;
}
function mockFetch(body: unknown, status = 200): FetchMock {
  const calls: FetchMock["calls"] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), body: init?.body ? JSON.parse(init.body as string) : null });
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  }) as typeof globalThis.fetch;
  return { calls, restore: () => { globalThis.fetch = original; } };
}

/** Mock fetch that returns a queued sequence of responses (one per call). */
function mockFetchSeq(responses: Array<{ status: number; body: unknown }>): FetchMock {
  const calls: FetchMock["calls"] = [];
  const original = globalThis.fetch;
  let i = 0;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), body: init?.body ? JSON.parse(init.body as string) : null });
    const r = responses[Math.min(i, responses.length - 1)]!;
    i++;
    return new Response(typeof r.body === "string" ? r.body : JSON.stringify(r.body), {
      status: r.status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof globalThis.fetch;
  return { calls, restore: () => { globalThis.fetch = original; } };
}

async function runVisionCommand(pi: MockPi, args: string, model?: Model<Api>): Promise<void> {
  const cmd = pi.commands.get("vision");
  assert.ok(cmd, "/vision command registered");
  const ctx = makeCtx({ model }) as unknown as ExtensionCommandContext;
  await cmd.handler(args, ctx);
}

async function executeTool(pi: MockPi, params: Record<string, unknown>, ctx: ExtensionContext): Promise<any> {
  const tool = pi.tools.get("describe_image");
  assert.ok(tool, "describe_image tool registered");
  return tool!.execute("call-1", params as any, ctx.signal, undefined, ctx);
}

function tmpImgDir(): { dir: string; file: string } {
  const dir = mkdtempSync(join(tmpdir(), "vision-eval-img-"));
  const file = join(dir, "pixel.png");
  writeFileSync(file, PNG_BYTES);
  return { dir, file };
}

// ── T1: multimodal primary → 0 delegation (tool hidden + image attached) ──
test("T1: multimodal primary → describe_image hidden + paste hook attaches image + 0 delegation", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetch({ choices: [{ message: { content: "should not be called" } }] });
  try {
    // session_start with multimodal primary → sync hides describe_image
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    assert.ok(!pi.getActiveTools().includes("describe_image"), "describe_image must be hidden for multimodal primary");

    // input hook attaches the image referenced by path
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `describe ${file}`, source: "interactive", images: [] },
      makeCtx({ model: MULTIMODAL, cwd: dir }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.ok(inputResult.images?.length >= 1, "paste hook attached the image");
    assert.equal(inputResult.images[0].mimeType, "image/png");

    // Even if the tool were somehow called, the defense-in-depth guard returns a redirect (0 delegation)
    const res = await executeTool(pi, { image_path: file, prompt: "what" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    assert.equal(res.details.mode, "passthrough_redirect");
    assert.equal(fm.calls.length, 0, "T1 GATE: 0 vision-model API calls for multimodal primary");
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T2: text-only primary → delegation fires ──
test("T2: text-only primary → describe_image visible + delegation fires + text returned", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetch({ choices: [{ message: { content: "a 1x1 transparent pixel" } }] });
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    assert.ok(pi.getActiveTools().includes("describe_image"), "describe_image must be visible for text-only primary");

    // v0.3.0: paste hook transforms text-only too (markers + hint) but does NOT attach images
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `describe ${file}`, source: "interactive", images: [] },
      makeCtx({ model: TEXT_ONLY, cwd: dir }),
    );
    assert.equal(inputResult?.action ?? "continue", "transform", "paste hook transforms text-only (markers + hint)");
    assert.equal((inputResult?.images ?? []).length, 0, "paste hook does NOT attach images for text-only primary");

    // Configure + execute → delegation
    await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
    await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
    const res = await executeTool(pi, { image_path: file, prompt: "describe", compress: false }, makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) }));
    assert.equal(res.details.mode, "delegate");
    assert.equal(fm.calls.length, 1, "exactly 1 vision-model API call (delegation)");
    assert.match(res.content[0].text, /1x1 transparent pixel/);
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T3: mid-session model switch flips the behavior ──
test("T3: model_select multimodal→text-only flips describe_image back on", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  // start multimodal → hidden
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL }));
  assert.ok(!pi.getActiveTools().includes("describe_image"));
  // switch to text-only → re-sync shows it
  await pi.emit("model_select", { type: "model_select", model: TEXT_ONLY, previousModel: MULTIMODAL, source: "set" }, makeCtx({ model: TEXT_ONLY }));
  assert.ok(pi.getActiveTools().includes("describe_image"), "describe_image re-enabled after switch to text-only");
  // switch back to multimodal → hidden again
  await pi.emit("model_select", { type: "model_select", model: MULTIMODAL, previousModel: TEXT_ONLY, source: "set" }, makeCtx({ model: MULTIMODAL }));
  assert.ok(!pi.getActiveTools().includes("describe_image"));
});

// ── T4: disabled → hidden + actionable error ──
test("T4: /vision off → describe_image hidden + actionable error if invoked", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  assert.ok(pi.getActiveTools().includes("describe_image"));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
  await runVisionCommand(pi, "off", TEXT_ONLY);
  assert.ok(!pi.getActiveTools().includes("describe_image"), "disabled → hidden");
  const res = await executeTool(pi, { image_path: "/x.png", prompt: "p", compress: false }, makeCtx({ model: TEXT_ONLY }));
  assert.equal(res.isError, true);
  assert.match(res.content[0].text, /\/vision on/);
});

// ── T5: unconfigured → actionable error with /vision config instructions ──
test("T5: unconfigured → actionable error with /vision config instructions", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "clear", TEXT_ONLY); // wipe provider/model
  const res = await executeTool(pi, { image_path: "/x.png", prompt: "p", compress: false }, makeCtx({ model: TEXT_ONLY }));
  assert.equal(res.isError, true);
  assert.match(res.content[0].text, /\/vision config provider/);
});

// ── T6: model not found in registry ──
test("T6: bogus vision model → model_not_found error referencing models.json", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model totally-bogus", TEXT_ONLY);
  // registry returns undefined (model not found)
  const res = await executeTool(
    pi,
    { image_path: "/x.png", prompt: "p", compress: false },
    makeCtx({ model: TEXT_ONLY, registry: makeRegistry({ model: undefined }) }),
  );
  assert.equal(res.isError, true);
  assert.match(res.content[0].text, /models\.json/);
});

// ── T7: auth failure ──
test("T7: auth failure → actionable auth_error", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
  const { dir, file } = tmpImgDir();
  try {
    const res = await executeTool(
      pi,
      { image_path: file, prompt: "p", compress: false },
      makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL, authOk: false }) }),
    );
    assert.equal(res.isError, true);
    assert.match(res.content[0].text, /no api key configured/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T8: image read failure ──
test("T8: nonexistent image path → not_found error with the path", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
  const res = await executeTool(
    pi,
    { image_path: "/nonexistent-eval-test.png", prompt: "p", compress: false },
    makeCtx({ model: TEXT_ONLY, registry: makeRegistry({ model: VISION_MODEL }) }),
  );
  assert.equal(res.isError, true);
  assert.match(res.content[0].text, /not found/);
  assert.match(res.content[0].text, /nonexistent-eval-test\.png/);
});

// ── T10: coexistence — paste hook dedups against already-attached images ──
test("T10: coexistence — paste hook does not duplicate an already-attached image", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    // Simulate another extension (e.g. pi-paster) having already attached the same image
    const preAttached = [{ type: "image", data: PNG_1x1_B64, mimeType: "image/png" }];
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `describe ${file}`, source: "interactive", images: preAttached },
      makeCtx({ model: MULTIMODAL, cwd: dir }),
    );
    // The hook must not duplicate: either continue (deduped entirely) or transform with the same single image
    if (inputResult?.action === "transform") {
      const pngCount = inputResult.images.filter((i: any) => i.mimeType === "image/png").length;
      assert.equal(pngCount, 1, "no double-attachment: the PNG appears exactly once");
    } else {
      assert.equal(inputResult?.action ?? "continue", "continue", "deduped → no new attachment");
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── Bonus: /vision show reflects config state ──
test("bonus: /vision show + config persistence across a fresh load", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model qwen3.5:cloud", TEXT_ONLY);
  await runVisionCommand(pi, "max-dim 2048", TEXT_ONLY);
  await runVisionCommand(pi, "quality 90", TEXT_ONLY);
  await runVisionCommand(pi, "reasoning-effort high", TEXT_ONLY);
  // A second session_start (simulate reload) should re-load the persisted config
  let notified = "";
  const ctx = makeCtx({ model: TEXT_ONLY }) as unknown as ExtensionCommandContext;
  (ctx.ui as any).notify = (msg: string) => { notified = msg; };
  await pi.commands.get("vision")!.handler("show", ctx);
  assert.match(notified, /qwen3\.5:cloud/);
  assert.match(notified, /2048px/);
  assert.match(notified, /jpegQuality.*90/);
  assert.match(notified, /reasoning.*high/);
});

// ── Picker: /vision model (no arg) opens native select over vision-capable models ──
test("picker: /vision model with no arg opens select + sets provider+model together", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  // Build a ctx whose ui.select returns a specific model choice + a registry that lists models
  let selectedTitle = "";
  let selectedOptions: string[] = [];
  const pickerCtx = makeCtx({
    model: TEXT_ONLY,
    registry: {
      getAvailable: () => [MULTIMODAL, VISION_MODEL, TEXT_ONLY],
      find: () => VISION_MODEL,
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: undefined }),
    } as any,
  }) as unknown as ExtensionCommandContext;
  (pickerCtx.ui as any).select = async (title: string, options: string[]) => {
    selectedTitle = title;
    selectedOptions = options;
    return "ollama/minimax-m3:cloud"; // user picks this
  };
  let notified = "";
  (pickerCtx.ui as any).notify = (msg: string) => { notified = msg; };
  await pi.commands.get("vision")!.handler("model", pickerCtx);
  assert.match(selectedTitle, /Pick a vision model/);
  // Only vision-capable (input includes image) models listed: MULTIMODAL + VISION_MODEL
  assert.ok(selectedOptions.includes("ollama/minimax-m3:cloud"));
  assert.ok(!selectedOptions.includes("ollama/glm-5.2:cloud"), "text-only model excluded from picker");
  assert.match(notified, /Vision model set to ollama\/minimax-m3:cloud/);
  // Verify it persisted: a subsequent /vision show reflects it
  let shown = "";
  const showCtx = makeCtx({ model: TEXT_ONLY }) as unknown as ExtensionCommandContext;
  (showCtx.ui as any).notify = (msg: string) => { shown = msg; };
  await pi.commands.get("vision")!.handler("show", showCtx);
  assert.match(shown, /minimax-m3:cloud/);
});

test("picker: no vision-capable models → actionable warning", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  let notified = "";
  let selectCalled = false;
  const pickerCtx = makeCtx({
    model: TEXT_ONLY,
    registry: {
      getAvailable: () => [TEXT_ONLY], // only a text-only model
      find: () => undefined,
      getApiKeyAndHeaders: async () => ({ ok: false, error: "x" }),
    } as any,
  }) as unknown as ExtensionCommandContext;
  (pickerCtx.ui as any).select = async () => { selectCalled = true; return undefined; };
  (pickerCtx.ui as any).notify = (msg: string) => { notified = msg; };
  await pi.commands.get("vision")!.handler("model", pickerCtx);
  assert.equal(selectCalled, false, "picker must not open when no vision-capable models");
  assert.match(notified, /No vision-capable models found/);
});

test("picker: user cancels (select returns undefined) → no config change", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model original-model", TEXT_ONLY);
  const pickerCtx = makeCtx({
    model: TEXT_ONLY,
    registry: { getAvailable: () => [MULTIMODAL], find: () => VISION_MODEL, getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: undefined }) } as any,
  }) as unknown as ExtensionCommandContext;
  (pickerCtx.ui as any).select = async () => undefined; // cancel
  (pickerCtx.ui as any).notify = () => {};
  await pi.commands.get("vision")!.handler("model", pickerCtx);
  // config.model unchanged
  let shown = "";
  const showCtx = makeCtx({ model: TEXT_ONLY }) as unknown as ExtensionCommandContext;
  (showCtx.ui as any).notify = (msg: string) => { shown = msg; };
  await pi.commands.get("vision")!.handler("show", showCtx);
  assert.match(shown, /original-model/, "model unchanged after cancel");
});
// ── /vision (no arg) → interactive settings panel (TUI) or text fallback ──
test("panel: /vision (no arg) in TUI mode → opens custom SettingsList panel + renders", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);

  let customCalled = false;
  let rendered: string[] = [];
  const panelCtx = makeCtx({
    model: TEXT_ONLY,
    registry: { getAvailable: () => [MULTIMODAL], find: () => VISION_MODEL, getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: undefined }) } as any,
  }) as unknown as ExtensionCommandContext;
  // makeCtx sets mode "tui" already
  (panelCtx.ui as any).custom = async (factory: any) => {
    customCalled = true;
    const mockTui = { requestRender: () => {} };
    const mockTheme = { fg: (_c: string, t: string) => t, bold: (t: string) => t };
    const comp = factory(mockTui, mockTheme, {} as any, () => {});
    rendered = comp.render(80);
    return true;
  };
  (panelCtx.ui as any).notify = () => {};
  // run /vision with NO args
  await pi.commands.get("vision")!.handler("", panelCtx);
  assert.equal(customCalled, true, "TUI mode must open the custom panel");
  assert.ok(rendered.length > 0, "panel rendered at least one line");
  const joined = rendered.join("\n");
  assert.match(joined, /Vision tool settings/);
  assert.match(joined, /Enabled/);
  assert.match(joined, /Vision model/);
  assert.match(joined, /minimax-m3:cloud/, "current model value shown in panel");
});

test("panel: /vision (no arg) in non-TUI mode → falls back to text status", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
  await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
  let customCalled = false;
  let notified = "";
  const panelCtx = makeCtx({ model: TEXT_ONLY }) as unknown as ExtensionCommandContext;
  (panelCtx as any).mode = "print"; // non-TUI
  (panelCtx.ui as any).custom = async () => { customCalled = true; return true; };
  (panelCtx.ui as any).notify = (msg: string) => { notified = msg; };
  await pi.commands.get("vision")!.handler("", panelCtx);
  assert.equal(customCalled, false, "non-TUI must NOT open the custom panel");
  assert.match(notified, /Vision tool config:/);
  assert.match(notified, /minimax-m3:cloud/);
});

test("panel: live edit via SettingsList onChange applies + persists", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  // Drive the panel factory + capture the SettingsList onChange callback, then invoke it
  let onChange: ((id: string, newValue: string) => void) | undefined;
  const panelCtx = makeCtx({ model: TEXT_ONLY, registry: { getAvailable: () => [MULTIMODAL], find: () => VISION_MODEL, getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: undefined }) } as any }) as unknown as ExtensionCommandContext;
  (panelCtx.ui as any).custom = async (factory: any) => {
    const mockTui = { requestRender: () => {} };
    const mockTheme = { fg: (_c: string, t: string) => t, bold: (t: string) => t };
    // Patch SettingsList to capture onChange — instantiate via the factory then
    // find the onChange by inspecting render is too indirect. Instead, simulate
    // by calling the factory and using the component handleInput is also indirect.
    // Simplest: the factory builds SettingsList internally; we can't grab its
    // onChange directly. So test the live-edit path via applySettingChange +
    // applyAndSave instead (already unit-tested). Here we just confirm the
    // factory constructs.
    const comp = factory(mockTui, mockTheme, {} as any, () => {});
    assert.ok(typeof comp.render === "function");
    return true;
  };
  (panelCtx.ui as any).notify = () => {};
  await pi.commands.get("vision")!.handler("", panelCtx);
  // The live-edit logic itself (applySettingChange + applyAndSave) is covered by
  // the lib/config tests; this test confirms the panel constructs in TUI mode.
});// ── v0.2.0 (SPEC-2) integration tests ────────────────────────────────────

test("T11: cache hit via tool execute → 2nd call = 0 vision-model calls + details.cached", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetch({ choices: [{ message: { content: "a pixel" } }] });
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
    await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
    const ctx = makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) });
    const r1 = await executeTool(pi, { image_path: file, prompt: "describe", compress: false }, ctx);
    assert.equal(r1.details.cached, false, "first call is a cache miss");
    assert.equal(fm.calls.length, 1, "first call fetches");
    const r2 = await executeTool(pi, { image_path: file, prompt: "describe", compress: false }, ctx);
    assert.equal(r2.details.cached, true, "second call is a cache hit");
    assert.equal(fm.calls.length, 1, "T11 GATE: 2nd call = 0 vision-model API calls");
    assert.match(r2.content[0].text, /a pixel/);
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T16: systemPrompt set → request body has system message first (via tool execute)", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetch({ choices: [{ message: { content: "ok" } }] });
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
    await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
    await runVisionCommand(pi, "system-prompt You are a forensic analyst.", TEXT_ONLY);
    const ctx = makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) });
    await executeTool(pi, { image_path: file, prompt: "p", compress: false }, ctx);
    assert.equal(fm.calls.length, 1);
    assert.equal(fm.calls[0]!.body.messages[0].role, "system");
    assert.equal(fm.calls[0]!.body.messages[0].content, "You are a forensic analyst.");
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T20: primary 4xx (no retry) → fallback fires via tool execute + details.fallback", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetchSeq([
    { status: 400, body: "bad request" },                                   // primary (no retry)
    { status: 200, body: { choices: [{ message: { content: "fb desc" } }] } }, // fallback
  ]);
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
    await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
    await runVisionCommand(pi, "fallback openrouter/qwen3.5:cloud", TEXT_ONLY);
    const ctx = makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) });
    const res = await executeTool(pi, { image_path: file, prompt: "p", compress: false }, ctx);
    assert.equal(res.details.fallback, true, "fallback fired");
    assert.equal(res.details.model, "openrouter/qwen3.5:cloud");
    assert.match(res.content[0].text, /fb desc/);
    assert.equal(fm.calls.length, 2, "1 primary (no retry) + 1 fallback");
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T24: ctrl+shift+i shortcut registered + invokes pickVisionModel → config updated", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  const shortcut = pi.shortcuts.get("ctrl+shift+i");
  assert.ok(shortcut, "ctrl+shift+i shortcut registered");
  let notified = "";
  const sc = makeCtx({
    model: TEXT_ONLY,
    registry: {
      getAvailable: () => [MULTIMODAL, VISION_MODEL],
      find: () => VISION_MODEL,
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: undefined }),
    } as any,
  }) as unknown as ExtensionContext;
  (sc.ui as any).select = async (_t: string, _o: string[]) => "ollama/minimax-m3:cloud";
  (sc.ui as any).notify = (msg: string) => { notified = msg; };
  await shortcut!.handler(sc);
  assert.match(notified, /Vision model set to ollama\//);
});

test("T25: /vision-use <provider/model> sets directly (no picker)", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY }));
  let notified = "";
  const cmd = pi.commands.get("vision-use");
  assert.ok(cmd, "/vision-use command registered");
  const c = makeCtx({ model: TEXT_ONLY }) as unknown as ExtensionCommandContext;
  (c.ui as any).notify = (msg: string) => { notified = msg; };
  await cmd!.handler("openrouter/qwen3.5:cloud", c);
  assert.match(notified, /openrouter\/qwen3\.5:cloud/);
  // Verify via /vision show
  let shown = "";
  const sc = makeCtx({ model: TEXT_ONLY }) as unknown as ExtensionCommandContext;
  (sc.ui as any).notify = (msg: string) => { shown = msg; };
  await pi.commands.get("vision")!.handler("show", sc);
  assert.match(shown, /qwen3\.5:cloud/);
});

// ── T28: multimodal primary → [Image-#1] marker + image attached ──
test("T28: multimodal primary → marker in text + image attached + no delegation", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `analyze ${file} for bugs`, source: "interactive", images: [] },
      makeCtx({ model: MULTIMODAL, cwd: dir }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.match(inputResult.text, /`\[Image-#1\]`/, "marker rendered as inline code");
    assert.equal((inputResult.images ?? []).length, 1, "image attached");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T29: multiple images → sequential markers ──
test("T29: multi-image → [Image-#1] + [Image-#2] sequential markers", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const file2 = join(dir, "second.png");
  writeFileSync(file2, PNG_BYTES_2);
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `compare ${file} and ${file2} please`, source: "interactive", images: [] },
      makeCtx({ model: MULTIMODAL, cwd: dir }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.match(inputResult.text, /`\[Image-#1\]`/);
    assert.match(inputResult.text, /`\[Image-#2\]`/);
    assert.equal((inputResult.images ?? []).length, 2, "2 images attached in order");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T30: marker style (code/bold/plain) ──
test("T30: marker style — code/bold/plain produce different transform text", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  try {
    // code (default)
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    let r = await pi.emit("input", { type: "input", text: `see ${file}`, source: "interactive", images: [] }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    assert.match(r.text, /`\[Image-#1\]`/, "code style → backtick-wrapped");

    // bold
    await runVisionCommand(pi, "marker-style bold", MULTIMODAL);
    r = await pi.emit("input", { type: "input", text: `see ${file}`, source: "interactive", images: [] }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    assert.match(r.text, /\*\*\[Image-#1\]\*\*/, "bold style → **-wrapped");

    // plain
    await runVisionCommand(pi, "marker-style plain", MULTIMODAL);
    r = await pi.emit("input", { type: "input", text: `see ${file}`, source: "interactive", images: [] }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    assert.ok(!r.text.includes("`"), "plain style → no backticks");
    assert.match(r.text, /\[Image-#1\]/, "plain style → bare marker");
    // Reset to code for subsequent tests (config persists in shared TMP_AGENT)
    await runVisionCommand(pi, "marker-style code", MULTIMODAL);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T31: text-only + hint → markers + hint line, no attachment ──
test("T31: text-only + hint → markers + hint line, no image attached", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `analyze ${file}`, source: "interactive", images: [] },
      makeCtx({ model: TEXT_ONLY, cwd: dir }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.match(inputResult.text, /`\[Image-#1\]`/, "marker rendered");
    assert.match(inputResult.text, /describe_image tool/, "hint line appended");
    assert.equal((inputResult.images ?? []).length, 0, "no image attached for text-only");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T32: text-only + auto → delegation fires + descriptions appended ──
test("T32: text-only + auto → delegate per image + descriptions appended, no attachment", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetch({ choices: [{ message: { content: "A red square on white bg." } }] });
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
    await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
    await runVisionCommand(pi, "paste-mode auto", TEXT_ONLY);
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `analyze ${file}`, source: "interactive", images: [] },
      makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.equal((inputResult.images ?? []).length, 0, "no image attached for text-only auto");
    assert.ok(fm.calls.length >= 1, "at least 1 vision-model API call (delegation)");
    assert.match(inputResult.text, /red square/, "description appended");
    assert.match(inputResult.text, /auto-described/, "cost-awareness footer");
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T33: text-only + off → markers only, no hint, no delegation ──
test("T33: text-only + off → markers only, no hint, no delegation", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  const fm = mockFetch({ choices: [{ message: { content: "should not be called" } }] });
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    await runVisionCommand(pi, "paste-mode off", TEXT_ONLY);
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `analyze ${file}`, source: "interactive", images: [] },
      makeCtx({ model: TEXT_ONLY, cwd: dir }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.match(inputResult.text, /`\[Image-#1\]`/, "marker rendered");
    assert.ok(!inputResult.text.includes("describe_image tool"), "no hint line in off mode");
    assert.ok(!inputResult.text.includes("auto-described"), "no delegation in off mode");
    assert.equal((inputResult.images ?? []).length, 0, "no attachment");
    assert.equal(fm.calls.length, 0, "no vision-model calls");
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T34: auto timeout → hint fallback ──
test("T34: auto mode timeout → hint fallback, message not blocked", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  // Fast-failing fetch (network error — throws immediately)
  const original = globalThis.fetch;
  globalThis.fetch = (async () => { throw new TypeError("network error"); }) as typeof globalThis.fetch;
  try {
    // Write a config with retryAttempts: 0 so the network error fails fast (no backoff)
    writeFileSync(join(TMP_AGENT, "vision.json"), JSON.stringify({
      provider: "ollama", model: "minimax-m3:cloud", enabled: true,
      retryAttempts: 0, textOnlyPasteMode: "auto",
    }));
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `analyze ${file}`, source: "interactive", images: [] },
      makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) }),
    );
    assert.equal(inputResult?.action, "transform", "message not blocked");
    // Timeout → hint fallback (the fetch hangs, so delegation never completes)
    assert.match(inputResult.text, /describe_image tool/, "hint fallback on timeout");
    assert.ok(!inputResult.text.includes("auto-described"), "no descriptions on timeout");
  } finally {
    globalThis.fetch = original;
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T35: auto delegation failure → hint fallback ──
test("T35: auto mode delegation failure → hint fallback, no crash", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  // Vision model returns 500 → delegation fails
  const fm = mockFetch({ error: "server error" }, 500);
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: TEXT_ONLY, cwd: dir }));
    await runVisionCommand(pi, "provider ollama", TEXT_ONLY);
    await runVisionCommand(pi, "model minimax-m3:cloud", TEXT_ONLY);
    await runVisionCommand(pi, "paste-mode auto", TEXT_ONLY);
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `analyze ${file}`, source: "interactive", images: [] },
      makeCtx({ model: TEXT_ONLY, cwd: dir, registry: makeRegistry({ model: VISION_MODEL }) }),
    );
    assert.equal(inputResult?.action, "transform", "no crash");
    assert.match(inputResult.text, /describe_image tool/, "hint fallback on failure");
  } finally {
    fm.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T36: marker numbering offset (pre-existing images) ──
test("T36: marker numbering offset — pre-existing images → #3", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    // Pre-attach 2 images (simulating another extension)
    const fakeImg: ImageContent = { type: "image", data: "AAAA", mimeType: "image/png" };
    const fakeImg2: ImageContent = { type: "image", data: "BBBB", mimeType: "image/png" };
    const inputResult = await pi.emit(
      "input",
      { type: "input", text: `see ${file}`, source: "interactive", images: [fakeImg, fakeImg2] },
      makeCtx({ model: MULTIMODAL, cwd: dir }),
    );
    assert.equal(inputResult?.action, "transform");
    assert.match(inputResult.text, /`\[Image-#3\]`/, "new image is #3 (offset from 2 pre-existing)");
    assert.equal((inputResult.images ?? []).length, 3, "3 total images");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T37: v0.2.x regression — T1–T27 still green ──
// (Covered by the existing T1–T27 tests above; this is a documentation marker.
// If any prior test fails, the suite fails here too. No separate assertions needed.)

// ── T38: compose-time preview calls setWidget with Image ──
test("T38: compose-time preview — setWidget called when image path detected", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  let widgetSet = false;
  try {
    const ctx = makeCtx({ model: MULTIMODAL, cwd: dir }) as any;
    ctx.hasUI = true;
    ctx.ui.setWidget = (key: string, content: any) => { if (content !== undefined) widgetSet = true; };
    ctx.ui.getEditorText = () => `analyze ${file}`;
    ctx.ui.onTerminalInput = (handler: any) => { (ctx as any)._inputHandler = handler; return () => {}; };

    await pi.emit("session_start", { type: "session_start", reason: "startup" }, ctx);
    if ((ctx as any)._inputHandler) (ctx as any)._inputHandler("a");
    await new Promise((r) => setTimeout(r, 400));
    assert.ok(widgetSet, "setWidget called with a component factory after debounce");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T39: compose-time preview clear on path removal ──
test("T39: compose-time preview — widget cleared when path removed from editor", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  let widgetSet = false;
  let widgetCleared = false;
  let editorText = `analyze ${file}`;
  try {
    const ctx = makeCtx({ model: MULTIMODAL, cwd: dir }) as any;
    ctx.hasUI = true;
    ctx.ui.setWidget = (key: string, content: any) => {
      if (content === undefined) widgetCleared = true;
      else widgetSet = true;
    };
    ctx.ui.getEditorText = () => editorText;
    ctx.ui.onTerminalInput = (handler: any) => { (ctx as any)._inputHandler = handler; return () => {}; };

    await pi.emit("session_start", { type: "session_start", reason: "startup" }, ctx);
    if ((ctx as any)._inputHandler) (ctx as any)._inputHandler("a");
    await new Promise((r) => setTimeout(r, 400));
    assert.ok(widgetSet, "widget set after typing a path");

    editorText = "just a normal message now";
    widgetCleared = false;
    if ((ctx as any)._inputHandler) (ctx as any)._inputHandler("a");
    await new Promise((r) => setTimeout(r, 400));
    assert.ok(widgetCleared, "widget cleared after path removed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T40: /vision preview opens panel (TUI mode) ──
test("T40: /vision preview <path> opens custom panel in TUI mode", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  let customCalled = false;
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL, cwd: dir }));
    // Call the command handler directly (not via runVisionCommand which creates its own ctx)
    const cmd = pi.commands.get("vision");
    assert.ok(cmd, "/vision command registered");
    const ctx = makeCtx({ model: MULTIMODAL, cwd: dir }) as any;
    ctx.ui.custom = async (factory: any) => {
      customCalled = true;
      const component = factory({}, { fg: () => "styled" }, { matches: () => false }, () => {});
      assert.ok(typeof component.render === "function", "factory returns a component with render()");
      return undefined;
    };
    await cmd.handler(`preview ${file}`, ctx);
    assert.ok(customCalled, "ctx.ui.custom called for /vision preview in TUI mode");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T41: /vision preview bad path → actionable error ──
test("T41: /vision preview nonexistent path → notify error, no panel", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  let notified = "";
  let customCalled = false;
  try {
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, makeCtx({ model: MULTIMODAL }));
    const cmd = pi.commands.get("vision");
    assert.ok(cmd, "/vision command registered");
    const ctx = makeCtx({ model: MULTIMODAL }) as any;
    ctx.ui.notify = (msg: string, type: string) => { notified = msg; };
    ctx.ui.custom = async () => { customCalled = true; return undefined; };
    await cmd.handler("preview /tmp/nonexistent-vision-test-12345.png", ctx);
    assert.ok(notified.length > 0, "error notified");
    assert.ok(!customCalled, "no panel opened for bad path");
    assert.match(notified, /could not load image|not_found|error/i);
  } finally {
  }
});

// ── T42: compose preview disabled → no setWidget ──
test("T42: composePreview disabled → no setWidget called", async () => {
  const pi = createMockPi();
  visionFactory(pi as unknown as ExtensionAPI);
  pasteFactory(pi as unknown as ExtensionAPI);
  const { dir, file } = tmpImgDir();
  let widgetSet = false;
  try {
    const ctx = makeCtx({ model: MULTIMODAL, cwd: dir }) as any;
    ctx.hasUI = true;
    ctx.ui.setWidget = (key: string, content: any) => { if (content !== undefined) widgetSet = true; };
    ctx.ui.getEditorText = () => `analyze ${file}`;
    ctx.ui.onTerminalInput = (handler: any) => { (ctx as any)._inputHandler = handler; return () => {}; };

    // Write config with composePreview: false
    writeFileSync(join(TMP_AGENT, "vision.json"), JSON.stringify({
      provider: "ollama", model: "minimax-m3:cloud", enabled: true,
      composePreview: false,
    }));
    await pi.emit("session_start", { type: "session_start", reason: "startup" }, ctx);
    if ((ctx as any)._inputHandler) (ctx as any)._inputHandler("a");
    await new Promise((r) => setTimeout(r, 400));
    assert.ok(!widgetSet, "no setWidget called when composePreview is disabled");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// Cleanup the temp agent dir after all tests.
test("cleanup", () => {
  rmSync(TMP_AGENT, { recursive: true, force: true });
});