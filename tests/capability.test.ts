import { test } from "node:test";
import assert from "node:assert/strict";
import type { Api, Model } from "@earendil-works/pi-ai";
import {
  isMultimodal,
  syncToolAvailability,
  TOOL_NAME,
  type ToolAvailabilityController,
} from "../lib/capability.ts";

function makeModel(input: ("text" | "image")[]): Model<Api> {
  return {
    id: "test-model",
    name: "Test",
    api: "openai-completions" as Api,
    provider: "test",
    baseUrl: "http://x",
    reasoning: false,
    input,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 4096,
  };
}

interface PiMock {
  pi: ToolAvailabilityController;
  sets: number;
  active: string[];
}

/** Minimal controller stub that records get/setActiveTools calls.
 *  `active` and `sets` are live getters over the closure state so tests can
 *  observe the result of a setActiveTools call (a plain property would
 *  snapshot the initial array reference and go stale on rebind). */
function makePi(initial: string[]): PiMock {
  let state = initial.slice();
  let sets = 0;
  const pi: ToolAvailabilityController = {
    getActiveTools: () => state.slice(),
    setActiveTools: (names: string[]) => {
      state = names.slice();
      sets += 1;
    },
  };
  return {
    pi,
    get sets() {
      return sets;
    },
    get active() {
      return state;
    },
  };
}

test("isMultimodal truth table", () => {
  assert.equal(isMultimodal(undefined), false);
  assert.equal(isMultimodal(makeModel(["text"])), false);
  assert.equal(isMultimodal(makeModel(["text", "image"])), true);
  assert.equal(isMultimodal(makeModel([])), false);
  assert.equal(isMultimodal(makeModel(["image"])), true);
});

test("TOOL_NAME is stable", () => {
  assert.equal(TOOL_NAME, "describe_image");
});

test("syncToolAvailability: text-only + enabled + absent → adds tool", () => {
  const m = makePi(["read", "bash"]);
  syncToolAvailability(m.pi, makeModel(["text"]), { enabled: true });
  assert.deepEqual(m.active.sort(), ["bash", "describe_image", "read"].sort());
  assert.equal(m.sets, 1);
});

test("syncToolAvailability: text-only + enabled + present → no-op", () => {
  const m = makePi(["read", "describe_image"]);
  syncToolAvailability(m.pi, makeModel(["text"]), { enabled: true });
  assert.equal(m.sets, 0, "must not call setActiveTools when already correct");
});

test("syncToolAvailability: multimodal + enabled + present → removes tool", () => {
  const m = makePi(["read", "describe_image", "bash"]);
  syncToolAvailability(m.pi, makeModel(["text", "image"]), { enabled: true });
  assert.deepEqual(m.active.sort(), ["bash", "read"].sort());
  assert.equal(m.sets, 1);
});

test("syncToolAvailability: multimodal + enabled + absent → no-op", () => {
  const m = makePi(["read", "bash"]);
  syncToolAvailability(m.pi, makeModel(["text", "image"]), { enabled: true });
  assert.equal(m.sets, 0);
});

test("syncToolAvailability: text-only + disabled + present → removes tool", () => {
  const m = makePi(["read", "describe_image"]);
  syncToolAvailability(m.pi, makeModel(["text"]), { enabled: false });
  assert.deepEqual(m.active, ["read"]);
  assert.equal(m.sets, 1);
});

test("syncToolAvailability: text-only + disabled + absent → no-op", () => {
  const m = makePi(["read"]);
  syncToolAvailability(m.pi, makeModel(["text"]), { enabled: false });
  assert.equal(m.sets, 0);
});

test("syncToolAvailability: multimodal + disabled + present → removes tool", () => {
  const m = makePi(["describe_image", "bash"]);
  syncToolAvailability(m.pi, makeModel(["text", "image"]), { enabled: false });
  assert.deepEqual(m.active, ["bash"]);
  assert.equal(m.sets, 1);
});

test("syncToolAvailability: undefined model + enabled + absent → adds tool (text-only default)", () => {
  const m = makePi(["read"]);
  syncToolAvailability(m.pi, undefined, { enabled: true });
  assert.deepEqual(m.active.sort(), ["describe_image", "read"].sort());
  assert.equal(m.sets, 1);
});

test("syncToolAvailability: preserves other extensions' tools when toggling", () => {
  const m = makePi(["read", "bash", "custom_tool", "describe_image"]);
  syncToolAvailability(m.pi, makeModel(["text", "image"]), { enabled: true });
  assert.deepEqual(m.active.sort(), ["bash", "custom_tool", "read"].sort());
});

test("syncToolAvailability: does not duplicate the tool if already present", () => {
  const m = makePi(["read", "describe_image"]);
  syncToolAvailability(m.pi, makeModel(["text"]), { enabled: true });
  assert.equal(m.active.filter((n) => n === TOOL_NAME).length, 1);
  assert.equal(m.sets, 0, "already present + text-only → no-op, no duplicate");
});