// delegator.test.mts — verifies createVisionDelegator + the public exports surface.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isMultimodal, loadConfig, createVisionDelegator } from "../src/index.ts";

// A fake ModelRegistry slice (the { find, getApiKeyAndHeaders } delegateToVisionModel reads).
const fakeRegistry = {
  find: () => undefined,
  getApiKeyAndHeaders: async () => ({ ok: false, error: "no auth" }),
};

test("isMultimodal + loadConfig + createVisionDelegator are exported", () => {
  assert.equal(typeof isMultimodal, "function");
  assert.equal(typeof loadConfig, "function");
  assert.equal(typeof createVisionDelegator, "function");
});

test("createVisionDelegator returns a delegator with config + delegate fn", () => {
  const d = createVisionDelegator({ modelRegistry: fakeRegistry as any, cwd: "/tmp", agentDir: "/tmp" });
  assert.equal(typeof d.delegate, "function");
  assert.equal(typeof d.config, "object");
  assert.equal(typeof d.config.defaultReasoningEffort, "string");
});

test("delegator.delegate on an unconfigured vision.json returns an actionable error (no crash)", async () => {
  const d = createVisionDelegator({ modelRegistry: fakeRegistry as any, cwd: "/tmp", agentDir: "/tmp/no-vision-config-here" });
  // default config has enabled:true but no provider/model → not_configured (isConfiguredForDelegation false)
  const result = await d.delegate({ image_path: "/nonexistent.png", prompt: "describe", compress: false, reasoning: "off" });
  assert.equal(result.ok, false);
  if (!result.ok) {
    // not_configured is the expected path (no provider/model in default config); model_not_found/disabled also acceptable
    assert.ok(["not_configured", "model_not_found", "disabled"].includes(result.error.code));
  }
});