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
} from "./config.ts";

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