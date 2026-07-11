import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeImagePaths } from "../extensions/vision.ts";

// ── basic shapes ──────────────────────────────────────────────────────────

test("normalizeImagePaths: empty input → []", () => {
  assert.deepEqual(normalizeImagePaths({}), []);
  assert.deepEqual(normalizeImagePaths({ prompt: "x" } as any), []);
});

test("normalizeImagePaths: single image_path string → [path]", () => {
  assert.deepEqual(normalizeImagePaths({ image_path: "/tmp/a.png" }), ["/tmp/a.png"]);
});

test("normalizeImagePaths: image_paths array → as-is (order preserved)", () => {
  assert.deepEqual(
    normalizeImagePaths({ image_paths: ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"] }),
    ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"],
  );
});

// ── merge (both params present) ───────────────────────────────────────────

test("normalizeImagePaths: both image_path + image_paths → merged (image_paths first)", () => {
  const out = normalizeImagePaths({
    image_paths: ["/tmp/a.png", "/tmp/b.png"],
    image_path: "/tmp/c.png",
  });
  assert.deepEqual(out, ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"]);
});

test("normalizeImagePaths: merge dedups (same path in both fields) → first wins, order preserved", () => {
  const out = normalizeImagePaths({
    image_paths: ["/tmp/a.png", "/tmp/b.png"],
    image_path: "/tmp/a.png", // duplicate of image_paths[0]
  });
  assert.deepEqual(out, ["/tmp/a.png", "/tmp/b.png"]);
});

test("normalizeImagePaths: dedup within image_paths (case-sensitive, first wins)", () => {
  assert.deepEqual(
    normalizeImagePaths({ image_paths: ["/tmp/a.png", "/tmp/a.png", "/tmp/b.png", "/tmp/A.png"] }),
    ["/tmp/a.png", "/tmp/b.png", "/tmp/A.png"],
  );
});

// ── schema-tolerant coercion (the edit.js:36 stringified-array gotcha) ────

test("normalizeImagePaths: stringified image_paths (JSON string) → parsed", () => {
  assert.deepEqual(
    normalizeImagePaths({ image_paths: JSON.stringify(["/tmp/a.png", "/tmp/b.png"]) }),
    ["/tmp/a.png", "/tmp/b.png"],
  );
});

test("normalizeImagePaths: stringified image_path (JSON string) → parsed as single-element array", () => {
  // image_path as a JSON string of an array → parsed to that array
  assert.deepEqual(
    normalizeImagePaths({ image_path: JSON.stringify(["/tmp/a.png", "/tmp/b.png"]) }),
    ["/tmp/a.png", "/tmp/b.png"],
  );
});

test("normalizeImagePaths: string starting with '[' but invalid JSON → treated as a single path", () => {
  // A malformed JSON string that starts with '[' should not crash; treat as one path.
  const out = normalizeImagePaths({ image_paths: "[not valid json" });
  assert.deepEqual(out, ["[not valid json"]);
});

// (A JSON-stringified *single string* as image_paths is an unrealistic model
// behavior; the realistic cases are a plain path string or a stringified
// array, both covered above. Skipping that edge case.)

// ── empty / whitespace filtering ──────────────────────────────────────────

test("normalizeImagePaths: filters empty + whitespace-only entries", () => {
  assert.deepEqual(
    normalizeImagePaths({ image_paths: ["/tmp/a.png", "", "   ", "/tmp/b.png"] }),
    ["/tmp/a.png", "/tmp/b.png"],
  );
});

test("normalizeImagePaths: trims whitespace around paths", () => {
  assert.deepEqual(
    normalizeImagePaths({ image_paths: ["  /tmp/a.png  ", "/tmp/b.png"] }),
    ["/tmp/a.png", "/tmp/b.png"],
  );
});

test("normalizeImagePaths: all-empty image_paths + no image_path → []", () => {
  assert.deepEqual(normalizeImagePaths({ image_paths: ["", "   "] }), []);
});

test("normalizeImagePaths: image_path empty string → filtered", () => {
  assert.deepEqual(normalizeImagePaths({ image_path: "" }), []);
  assert.deepEqual(normalizeImagePaths({ image_path: "   " }), []);
});

// ── non-string entries (defensive — model sends wrong types) ──────────────

test("normalizeImagePaths: filters non-string entries in array", () => {
  assert.deepEqual(
    normalizeImagePaths({ image_paths: ["/tmp/a.png", 123, null, "/tmp/b.png", true] as any }),
    ["/tmp/a.png", "/tmp/b.png"],
  );
});

test("normalizeImagePaths: undefined / null fields → []", () => {
  assert.deepEqual(normalizeImagePaths({ image_path: undefined }), []);
  assert.deepEqual(normalizeImagePaths({ image_paths: null as any }), []);
  assert.deepEqual(normalizeImagePaths({ image_path: null as any, image_paths: undefined }), []);
});

// ── no path-normalization (intentional — §1.6) ────────────────────────────

test("normalizeImagePaths: no path normalization (/tmp/a.png and /tmp/./a.png are distinct)", () => {
  // Dedup is exact-string; canonicalization is loadImage's job downstream.
  const out = normalizeImagePaths({ image_paths: ["/tmp/a.png", "/tmp/./a.png"] });
  assert.deepEqual(out, ["/tmp/a.png", "/tmp/./a.png"]);
});