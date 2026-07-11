import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildDescriptionsBlock,
  buildHintLine,
  buildBatchToolResult,
  isMarkerStyle,
  MARKER_STYLES,
  renderMarkers,
  styleMarker,
} from "../lib/marker.ts";

// ── styleMarker ──────────────────────────────────────────────────────────

test("styleMarker: code style wraps in backticks", () => {
  assert.equal(styleMarker(1, "code"), "`[Image-#1]`");
  assert.equal(styleMarker(10, "code"), "`[Image-#10]`");
});

test("styleMarker: bold style wraps in **", () => {
  assert.equal(styleMarker(1, "bold"), "**[Image-#1]**");
  assert.equal(styleMarker(3, "bold"), "**[Image-#3]**");
});

test("styleMarker: plain style has no wrapping", () => {
  assert.equal(styleMarker(1, "plain"), "[Image-#1]");
  assert.equal(styleMarker(7, "plain"), "[Image-#7]");
});

test("styleMarker: raw number used as-is (0-based → 1-based conversion is in renderMarkers)", () => {
  assert.equal(styleMarker(0, "code"), "`[Image-#0]`");
  assert.equal(styleMarker(1, "code"), "`[Image-#1]`");
});

// ── isMarkerStyle + MARKER_STYLES ─────────────────────────────────────────

test("isMarkerStyle: valid styles", () => {
  for (const s of MARKER_STYLES) {
    assert.ok(isMarkerStyle(s), `${s} should be valid`);
  }
  assert.equal(MARKER_STYLES.length, 3);
});

test("isMarkerStyle: invalid values", () => {
  assert.ok(!isMarkerStyle("accent"));
  assert.ok(!isMarkerStyle("warning"));
  assert.ok(!isMarkerStyle(""));
  assert.ok(!isMarkerStyle(undefined));
  assert.ok(!isMarkerStyle(123));
});

// ── renderMarkers: basic ─────────────────────────────────────────────────

test("renderMarkers: single token replaced (index 0 → #1)", () => {
  const text = "analyze /tmp/a.png for bugs";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "code");
  assert.equal(out, "analyze `[Image-#1]` for bugs");
});

test("renderMarkers: multiple tokens → sequential markers", () => {
  const text = "compare /tmp/a.png and /tmp/b.jpeg please";
  const resolved = new Map([
    ["/tmp/a.png", { index: 0 }],
    ["/tmp/b.jpeg", { index: 1 }],
  ]);
  const out = renderMarkers(text, ["/tmp/a.png", "/tmp/b.jpeg"], resolved, "code");
  assert.equal(out, "compare `[Image-#1]` and `[Image-#2]` please");
});

test("renderMarkers: offset numbering (pre-existing images)", () => {
  const text = "see /tmp/c.png here";
  const resolved = new Map([["/tmp/c.png", { index: 2 }]]); // 2 pre-existing → #3
  const out = renderMarkers(text, ["/tmp/c.png"], resolved, "code");
  assert.equal(out, "see `[Image-#3]` here");
});

test("renderMarkers: unresolvable token left as-is", () => {
  const text = "analyze /tmp/missing.png and /tmp/a.png";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/missing.png", "/tmp/a.png"], resolved, "code");
  assert.equal(out, "analyze /tmp/missing.png and `[Image-#1]`");
});

test("renderMarkers: no resolvable tokens → text unchanged", () => {
  const text = "just a normal message";
  const out = renderMarkers(text, [], new Map(), "code");
  assert.equal(out, "just a normal message");
});

test("renderMarkers: bold style", () => {
  const text = "see /tmp/a.png";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "bold");
  assert.equal(out, "see **[Image-#1]**");
});

test("renderMarkers: plain style", () => {
  const text = "see /tmp/a.png";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "plain");
  assert.equal(out, "see [Image-#1]");
});

// ── renderMarkers: edge cases ────────────────────────────────────────────

test("renderMarkers: token at string start", () => {
  const text = "/tmp/a.png is the image";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "code");
  assert.equal(out, "`[Image-#1]` is the image");
});

test("renderMarkers: token at string end", () => {
  const text = "the image is /tmp/a.png";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "code");
  assert.equal(out, "the image is `[Image-#1]`");
});

test("renderMarkers: token is the entire text", () => {
  const text = "/tmp/a.png";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "code");
  assert.equal(out, "`[Image-#1]`");
});

test("renderMarkers: same token appears twice → both replaced", () => {
  const text = "/tmp/a.png then /tmp/a.png again";
  const resolved = new Map([["/tmp/a.png", { index: 0 }]]);
  const out = renderMarkers(text, ["/tmp/a.png"], resolved, "code");
  assert.equal(out, "`[Image-#1]` then `[Image-#1]` again");
});

test("renderMarkers: token adjacent to punctuation", () => {
  const text = "see (/tmp/a.png) and /tmp/b.png.";
  const resolved = new Map([
    ["/tmp/a.png", { index: 0 }],
    ["/tmp/b.png", { index: 1 }],
  ]);
  const out = renderMarkers(text, ["/tmp/a.png", "/tmp/b.png"], resolved, "code");
  assert.equal(out, "see (`[Image-#1]`) and `[Image-#2]`.");
});

test("renderMarkers: overlapping tokens (longer wins)", () => {
  const text = "file /tmp/a.png.bak here";
  const resolved = new Map([
    ["/tmp/a.png", { index: 0 }],
    ["/tmp/a.png.bak", { index: 1 }],
  ]);
  const out = renderMarkers(text, ["/tmp/a.png", "/tmp/a.png.bak"], resolved, "code");
  assert.equal(out, "file `[Image-#2]` here");
});

test("renderMarkers: empty text", () => {
  const out = renderMarkers("", ["/tmp/a.png"], new Map(), "code");
  assert.equal(out, "");
});

// ── buildHintLine (v0.4.0: lists paths + names batch affordance) ───────────

test("buildHintLine: single image → singular noun, one path, no batch clause", () => {
  const line = buildHintLine([{ token: "/tmp/a.png", index: 0 }]);
  assert.ok(line.startsWith("1 image referenced."), "singular noun");
  assert.ok(line.includes("analyze it."), "singular verb");
  assert.ok(!line.includes("image_paths"), "no batch affordance for 1 image");
  assert.ok(line.includes("  /tmp/a.png"), "path listed indented");
});

test("buildHintLine: multiple images → plural noun, N paths, batch affordance", () => {
  const line = buildHintLine([
    { token: "/tmp/a.png", index: 0 },
    { token: "/tmp/b.jpeg", index: 1 },
  ]);
  assert.ok(line.startsWith("2 images referenced."), "plural noun");
  assert.ok(line.includes("analyze them"), "plural verb");
  assert.ok(line.includes("image_paths"), "names the batch affordance");
  assert.ok(line.includes("  /tmp/a.png"), "path 1 listed");
  assert.ok(line.includes("  /tmp/b.jpeg"), "path 2 listed");
});

test("buildHintLine: zero images (defensive)", () => {
  const line = buildHintLine([]);
  assert.equal(line, "0 images referenced.");
});

test("buildHintLine: paths are extractable via regex", () => {
  const line = buildHintLine([
    { token: "/tmp/a.png", index: 0 },
    { token: "/tmp/pi-clipboard-3f1c.png", index: 1 },
  ]);
  const paths = [...line.matchAll(/^  (.+)$/gm)].map((m) => m[1]);
  assert.deepEqual(paths, ["/tmp/a.png", "/tmp/pi-clipboard-3f1c.png"]);
});

test("buildHintLine: preserves token order (index not used for ordering)", () => {
  // The caller passes tokens in marker order; output lists them in that order.
  const line = buildHintLine([
    { token: "/tmp/first.png", index: 0 },
    { token: "/tmp/second.png", index: 1 },
  ]);
  const firstIdx = line.indexOf("/tmp/first.png");
  const secondIdx = line.indexOf("/tmp/second.png");
  assert.ok(firstIdx < secondIdx && firstIdx > -1, "first before second");
});

// ── buildBatchToolResult (v0.4.0: structured per-image tool result) ───────

test("buildBatchToolResult: header + per-image sections in input order", () => {
  const out = buildBatchToolResult(
    ["/tmp/a.png", "/tmp/b.jpeg", "/tmp/c.png"],
    [
      { ok: true, text: "A red square.", cached: false, fallback: false },
      { ok: true, text: "A blue circle.", cached: false, fallback: false },
      { ok: true, text: "A green triangle.", cached: false, fallback: false },
    ],
  );
  assert.ok(out.startsWith("[Batch: 3 image(s)]"), "header with count");
  assert.ok(out.includes("[Image 1] /tmp/a.png"), "image 1 header");
  assert.ok(out.includes("[Image 2] /tmp/b.jpeg"), "image 2 header");
  assert.ok(out.includes("[Image 3] /tmp/c.png"), "image 3 header");
  assert.ok(out.includes("A red square."));
  assert.ok(out.includes("A blue circle."));
  assert.ok(out.includes("A green triangle."));
  // Order check: image 1 before 2 before 3
  const i1 = out.indexOf("[Image 1]");
  const i2 = out.indexOf("[Image 2]");
  const i3 = out.indexOf("[Image 3]");
  assert.ok(i1 < i2 && i2 < i3, "sections in input order");
});

test("buildBatchToolResult: cached tag", () => {
  const out = buildBatchToolResult(
    ["/tmp/a.png"],
    [{ ok: true, text: "desc", cached: true, fallback: false }],
  );
  assert.ok(out.includes("[Image 1] (cached) /tmp/a.png"), "cached tag on header");
});

test("buildBatchToolResult: fallback tag includes model", () => {
  const out = buildBatchToolResult(
    ["/tmp/a.png"],
    [{ ok: true, text: "desc", cached: false, fallback: true, fallbackModel: "ollama/glm4v:cloud" }],
  );
  assert.ok(out.includes("[Image 1] (fallback: ollama/glm4v:cloud) /tmp/a.png"), "fallback tag with model");
});

test("buildBatchToolResult: failed image → [error: code — message] section, not whole-batch fail", () => {
  const out = buildBatchToolResult(
    ["/tmp/good.png", "/tmp/bad.png", "/tmp/good2.png"],
    [
      { ok: true, text: "good 1", cached: false, fallback: false },
      { ok: false, errorCode: "not_found", message: "image not found at /tmp/bad.png" },
      { ok: true, text: "good 2", cached: false, fallback: false },
    ],
  );
  assert.ok(out.includes("[Image 2] /tmp/bad.png"), "failed image header present");
  assert.ok(out.includes("[error: not_found — image not found at /tmp/bad.png]"), "error section");
  assert.ok(out.includes("good 1") && out.includes("good 2"), "successful descriptions preserved");
});

test("buildBatchToolResult: all fail → still returns full text (caller sets isError)", () => {
  const out = buildBatchToolResult(
    ["/tmp/bad1.png", "/tmp/bad2.png"],
    [
      { ok: false, errorCode: "not_found", message: "missing 1" },
      { ok: false, errorCode: "read_error", message: "missing 2" },
    ],
  );
  assert.ok(out.startsWith("[Batch: 2 image(s)]"));
  assert.ok(out.includes("[error: not_found — missing 1]"));
  assert.ok(out.includes("[error: read_error — missing 2]"));
});

test("buildBatchToolResult: empty paths (defensive)", () => {
  const out = buildBatchToolResult([], []);
  assert.ok(out.startsWith("["), "returns something non-empty for safety");
  assert.ok(!out.includes("[Image"), "no per-image sections for empty input");
});

// ── buildDescriptionsBlock ──────────────────────────────────────────────

test("buildDescriptionsBlock: single image", () => {
  const out = buildDescriptionsBlock(
    [{ token: "/tmp/a.png", index: 0, text: "A red square.", cached: false }],
    "ollama/minimax-m3:cloud",
  );
  assert.ok(out.startsWith("\n\n"));
  assert.ok(out.includes("[`[Image-#1]` /tmp/a.png]: A red square."));
  assert.ok(out.includes("auto-described via ollama/minimax-m3:cloud"));
  assert.ok(out.includes('Set textOnlyPasteMode to "hint"'));
});

test("buildDescriptionsBlock: multiple images", () => {
  const out = buildDescriptionsBlock(
    [
      { token: "/tmp/a.png", index: 0, text: "A red square.", cached: false },
      { token: "/tmp/b.jpeg", index: 1, text: "A blue circle.", cached: true },
    ],
    "ollama/minimax-m3:cloud",
  );
  assert.ok(out.includes("[`[Image-#1]` /tmp/a.png]: A red square."));
  assert.ok(out.includes("[`[Image-#2]` /tmp/b.jpeg]: A blue circle. (cached)"));
  assert.ok(out.includes("2 image(s) auto-described"));
});

test("buildDescriptionsBlock: cached tag appears only when cached", () => {
  const out = buildDescriptionsBlock(
    [{ token: "/tmp/a.png", index: 0, text: "desc", cached: false }],
    "m",
  );
  assert.ok(!out.includes("(cached)"));
});

test("buildDescriptionsBlock: empty → empty string", () => {
  assert.equal(buildDescriptionsBlock([], "m"), "");
});