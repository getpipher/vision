import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  configFilePath,
  CONFIG_FILENAME,
  DEFAULT_CONFIG,
  isConfiguredForDelegation,
  loadConfig,
  mergeConfig,
  saveConfig,
  applySettingChange,
} from "../lib/config.ts";

function tmpAgentDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-cfg-"));
}

test("DEFAULT_CONFIG has the expected shape", () => {
  assert.deepEqual(DEFAULT_CONFIG, {
    provider: undefined,
    model: undefined,
    maxDimension: 1568,
    jpegQuality: 85,
    defaultReasoningEffort: "off",
    enabled: true,
    systemPrompt: undefined,
    cacheEnabled: true,
    cachePersist: false,
    cacheMaxEntries: 256,
    retryAttempts: 2,
    retryBackoffMs: 500,
    fallbackProvider: undefined,
    fallbackModel: undefined,
  });
});

test("mergeConfig: empty/unknown input → defaults", () => {
  assert.deepEqual(mergeConfig(undefined), DEFAULT_CONFIG);
  assert.deepEqual(mergeConfig({}), DEFAULT_CONFIG);
  assert.deepEqual(mergeConfig(null), DEFAULT_CONFIG);
});

test("mergeConfig: valid full config passes through", () => {
  const c = mergeConfig({
    provider: "ollama",
    model: "minimax-m3:cloud",
    maxDimension: 1024,
    jpegQuality: 90,
    defaultReasoningEffort: "high",
    enabled: false,
  });
  assert.equal(c.provider, "ollama");
  assert.equal(c.model, "minimax-m3:cloud");
  assert.equal(c.maxDimension, 1024);
  assert.equal(c.jpegQuality, 90);
  assert.equal(c.defaultReasoningEffort, "high");
  assert.equal(c.enabled, false);
});

test("mergeConfig: empty-string provider/model → undefined (not empty string)", () => {
  const c = mergeConfig({ provider: "", model: "  " });
  assert.equal(c.provider, undefined);
  assert.equal(c.model, undefined);
});

test("mergeConfig: clamps maxDimension to [1, 8000]", () => {
  assert.equal(mergeConfig({ maxDimension: 0 }).maxDimension, 1);
  assert.equal(mergeConfig({ maxDimension: 99999 }).maxDimension, 8000);
  assert.equal(mergeConfig({ maxDimension: 1234.7 }).maxDimension, 1235);
  assert.equal(mergeConfig({ maxDimension: "256" }).maxDimension, 256);
  assert.equal(
    mergeConfig({ maxDimension: "not-a-number" }).maxDimension,
    DEFAULT_CONFIG.maxDimension,
  );
});

test("mergeConfig: clamps jpegQuality to [1, 100]", () => {
  assert.equal(mergeConfig({ jpegQuality: 0 }).jpegQuality, 1);
  assert.equal(mergeConfig({ jpegQuality: 150 }).jpegQuality, 100);
  assert.equal(mergeConfig({ jpegQuality: 85.4 }).jpegQuality, 85);
});

test("mergeConfig: invalid reasoning → default off", () => {
  assert.equal(mergeConfig({ defaultReasoningEffort: "bogus" }).defaultReasoningEffort, "off");
  assert.equal(mergeConfig({ defaultReasoningEffort: 5 }).defaultReasoningEffort, "off");
  assert.equal(mergeConfig({ defaultReasoningEffort: "high" }).defaultReasoningEffort, "high");
});

test("mergeConfig: non-boolean enabled → default true", () => {
  assert.equal(mergeConfig({ enabled: "yes" }).enabled, true);
  assert.equal(mergeConfig({ enabled: 0 }).enabled, true);
  assert.equal(mergeConfig({ enabled: false }).enabled, false);
});

test("configFilePath joins agent dir + filename", () => {
  assert.equal(configFilePath("/home/u/.pi/agent"), "/home/u/.pi/agent/vision.json");
  assert.equal(configFilePath("/tmp/x"), "/tmp/x/vision.json");
});

test("loadConfig: missing file → defaults (no throw)", () => {
  const dir = tmpAgentDir();
  try {
    assert.deepEqual(loadConfig(dir), DEFAULT_CONFIG);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadConfig: malformed JSON → defaults (no throw)", () => {
  const dir = tmpAgentDir();
  try {
    writeFileSync(join(dir, CONFIG_FILENAME), "{ not valid json");
    assert.deepEqual(loadConfig(dir), DEFAULT_CONFIG);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadConfig: valid file → merged config", () => {
  const dir = tmpAgentDir();
  try {
    writeFileSync(
      join(dir, CONFIG_FILENAME),
      JSON.stringify({ provider: "ollama", model: "qwen3.5:cloud", maxDimension: 2048 }),
    );
    const c = loadConfig(dir);
    assert.equal(c.provider, "ollama");
    assert.equal(c.model, "qwen3.5:cloud");
    assert.equal(c.maxDimension, 2048);
    assert.equal(c.jpegQuality, DEFAULT_CONFIG.jpegQuality);
    assert.equal(c.enabled, true);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("saveConfig: writes valid JSON, round-trips through loadConfig", () => {
  const dir = tmpAgentDir();
  try {
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
    saveConfig(cfg, dir);
    const raw = readFileSync(join(dir, CONFIG_FILENAME), "utf8");
    assert.ok(raw.includes('"provider": "ollama"'));
    assert.ok(raw.endsWith("\n"), "file should end with a newline");
    assert.deepEqual(loadConfig(dir), cfg);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("saveConfig: atomic (no .tmp leftover)", () => {
  const dir = tmpAgentDir();
  try {
    saveConfig(DEFAULT_CONFIG, dir);
    const files = readdirSync(dir);
    assert.deepEqual(files, [CONFIG_FILENAME]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("isConfiguredForDelegation", () => {
  assert.equal(isConfiguredForDelegation(DEFAULT_CONFIG), false);
  assert.equal(isConfiguredForDelegation({ ...DEFAULT_CONFIG, provider: "ollama" }), false);
  assert.equal(isConfiguredForDelegation({ ...DEFAULT_CONFIG, model: "x" }), false);
  assert.equal(
    isConfiguredForDelegation({ ...DEFAULT_CONFIG, provider: "ollama", model: "x" }),
    true,
  );
});

test("applySettingChange: enabled on/off", () => {
  const base = { ...DEFAULT_CONFIG, enabled: true };
  assert.equal(applySettingChange(base, "enabled", "off").enabled, false);
  assert.equal(applySettingChange(base, "enabled", "on").enabled, true);
  assert.equal(applySettingChange(base, "enabled", "garbage").enabled, false);
});

test("applySettingChange: model provider/id splits both", () => {
  const r = applySettingChange(DEFAULT_CONFIG, "model", "ollama/minimax-m3:cloud");
  assert.equal(r.provider, "ollama");
  assert.equal(r.model, "minimax-m3:cloud");
});

test("applySettingChange: model bare id keeps provider", () => {
  const base = { ...DEFAULT_CONFIG, provider: "ollama", model: "old" };
  const r = applySettingChange(base, "model", "new-model");
  assert.equal(r.provider, "ollama");
  assert.equal(r.model, "new-model");
});

test("applySettingChange: maxDimension parses + clamps", () => {
  assert.equal(applySettingChange(DEFAULT_CONFIG, "maxDimension", "2048px").maxDimension, 2048);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "maxDimension", "99999").maxDimension, 8000);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "maxDimension", "0").maxDimension, 1);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "maxDimension", "notanum").maxDimension, DEFAULT_CONFIG.maxDimension);
});

test("applySettingChange: jpegQuality parses + clamps", () => {
  assert.equal(applySettingChange(DEFAULT_CONFIG, "jpegQuality", "90").jpegQuality, 90);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "jpegQuality", "150").jpegQuality, 100);
});

test("applySettingChange: reasoning sets valid level, rejects invalid", () => {
  assert.equal(applySettingChange(DEFAULT_CONFIG, "reasoning", "high").defaultReasoningEffort, "high");
  assert.equal(applySettingChange(DEFAULT_CONFIG, "reasoning", "bogus").defaultReasoningEffort, "off");
});

test("applySettingChange: unknown id → unchanged", () => {
  assert.deepEqual(applySettingChange(DEFAULT_CONFIG, "nope", "x"), DEFAULT_CONFIG);
});

// ── v0.2.0 fields (SPEC-2) ──────────────────────────────────────────────────

test("mergeConfig: v0.1.0-shape config (6 fields) loads with v0.2.0 defaults (forward-compat)", () => {
  const c = mergeConfig({
    provider: "ollama",
    model: "minimax-m3:cloud",
    maxDimension: 1568,
    jpegQuality: 85,
    defaultReasoningEffort: "off",
    enabled: true,
  });
  assert.equal(c.systemPrompt, undefined);
  assert.equal(c.cacheEnabled, true);
  assert.equal(c.cachePersist, false);
  assert.equal(c.cacheMaxEntries, 256);
  assert.equal(c.retryAttempts, 2);
  assert.equal(c.retryBackoffMs, 500);
  assert.equal(c.fallbackProvider, undefined);
  assert.equal(c.fallbackModel, undefined);
});

test("mergeConfig: v0.2.0 fields pass through + validate", () => {
  const c = mergeConfig({
    systemPrompt: "  You are an analyst.  ",
    cacheEnabled: false,
    cachePersist: true,
    cacheMaxEntries: 512,
    retryAttempts: 5,
    retryBackoffMs: 1000,
    fallbackProvider: "openrouter",
    fallbackModel: "qwen3.5:cloud",
  });
  assert.equal(c.systemPrompt, "You are an analyst.", "trimmed");
  assert.equal(c.cacheEnabled, false);
  assert.equal(c.cachePersist, true);
  assert.equal(c.cacheMaxEntries, 512);
  assert.equal(c.retryAttempts, 5);
  assert.equal(c.retryBackoffMs, 1000);
  assert.equal(c.fallbackProvider, "openrouter");
  assert.equal(c.fallbackModel, "qwen3.5:cloud");
});

test("mergeConfig: empty systemPrompt → undefined", () => {
  assert.equal(mergeConfig({ systemPrompt: "   " }).systemPrompt, undefined);
  assert.equal(mergeConfig({ systemPrompt: "" }).systemPrompt, undefined);
});

test("mergeConfig: clamps cacheMaxEntries to [1, 10000]", () => {
  assert.equal(mergeConfig({ cacheMaxEntries: 0 }).cacheMaxEntries, 1);
  assert.equal(mergeConfig({ cacheMaxEntries: 999999 }).cacheMaxEntries, 10000);
  assert.equal(mergeConfig({ cacheMaxEntries: "128" }).cacheMaxEntries, 128);
  assert.equal(mergeConfig({ cacheMaxEntries: "x" }).cacheMaxEntries, 256);
});

test("mergeConfig: clamps retryAttempts to [0, 10] + retryBackoffMs to [0, 60000]", () => {
  assert.equal(mergeConfig({ retryAttempts: -1 }).retryAttempts, 0);
  assert.equal(mergeConfig({ retryAttempts: 99 }).retryAttempts, 10);
  assert.equal(mergeConfig({ retryBackoffMs: -5 }).retryBackoffMs, 0);
  assert.equal(mergeConfig({ retryBackoffMs: 999999 }).retryBackoffMs, 60000);
});

test("mergeConfig: non-boolean cache flags → defaults", () => {
  assert.equal(mergeConfig({ cacheEnabled: "yes" }).cacheEnabled, true);
  assert.equal(mergeConfig({ cachePersist: 1 }).cachePersist, false);
  assert.equal(mergeConfig({ cacheEnabled: false }).cacheEnabled, false);
});

test("applySettingChange: systemPrompt set / clear", () => {
  const set = applySettingChange(DEFAULT_CONFIG, "systemPrompt", "You are a forensic analyst.");
  assert.equal(set.systemPrompt, "You are a forensic analyst.");
  const cleared = applySettingChange(set, "systemPrompt", "");
  assert.equal(cleared.systemPrompt, undefined, "empty string clears");
  const trimmed = applySettingChange(DEFAULT_CONFIG, "systemPrompt", "  hi  ");
  assert.equal(trimmed.systemPrompt, "hi");
});

test("applySettingChange: cacheEnabled / cachePersist on/off", () => {
  assert.equal(applySettingChange(DEFAULT_CONFIG, "cacheEnabled", "off").cacheEnabled, false);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "cacheEnabled", "on").cacheEnabled, true);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "cachePersist", "on").cachePersist, true);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "cachePersist", "off").cachePersist, false);
});

test("applySettingChange: cacheMaxEntries / retryAttempts / retryBackoffMs parse + clamp", () => {
  assert.equal(applySettingChange(DEFAULT_CONFIG, "cacheMaxEntries", "512").cacheMaxEntries, 512);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "cacheMaxEntries", "99999").cacheMaxEntries, 10000);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "retryAttempts", "5").retryAttempts, 5);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "retryAttempts", "99").retryAttempts, 10);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "retryBackoffMs", "1000").retryBackoffMs, 1000);
  assert.equal(applySettingChange(DEFAULT_CONFIG, "retryBackoffMs", "notanum").retryBackoffMs, DEFAULT_CONFIG.retryBackoffMs);
});

test("applySettingChange: fallbackModel provider/id splits both; bare id keeps fallbackProvider", () => {
  const r = applySettingChange(DEFAULT_CONFIG, "fallbackModel", "openrouter/qwen3.5:cloud");
  assert.equal(r.fallbackProvider, "openrouter");
  assert.equal(r.fallbackModel, "qwen3.5:cloud");
  const base = { ...DEFAULT_CONFIG, fallbackProvider: "openrouter", fallbackModel: "old" };
  const bare = applySettingChange(base, "fallbackModel", "new-fb");
  assert.equal(bare.fallbackProvider, "openrouter");
  assert.equal(bare.fallbackModel, "new-fb");
  const cleared = applySettingChange(base, "fallbackModel", "");
  assert.equal(cleared.fallbackModel, undefined);
});
