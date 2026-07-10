import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildDescriptionsBlock,
  buildHintLine,
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

// ── buildHintLine ────────────────────────────────────────────────────────

test("buildHintLine: singular", () => {
  const line = buildHintLine(1);
  assert.equal(
    line,
    "[1 image referenced. The active model cannot process images natively — use the describe_image tool to analyze them.]",
  );
});

test("buildHintLine: plural", () => {
  const line = buildHintLine(3);
  assert.equal(
    line,
    "[3 images referenced. The active model cannot process images natively — use the describe_image tool to analyze them.]",
  );
});

test("buildHintLine: zero (edge case)", () => {
  const line = buildHintLine(0);
  assert.ok(line.includes("0 images referenced"));
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