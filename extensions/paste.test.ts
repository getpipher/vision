import { test } from "node:test";
import assert from "node:assert/strict";
import { findImagePathTokens } from "./paste.ts";

test("findImagePathTokens: absolute path", () => {
  assert.deepEqual(findImagePathTokens("analyze /tmp/screenshot.png"), ["/tmp/screenshot.png"]);
});

test("findImagePathTokens: home path", () => {
  assert.deepEqual(findImagePathTokens("see ~/Pictures/diagram.jpg"), ["~/Pictures/diagram.jpg"]);
});

test("findImagePathTokens: relative ./ and ../ paths", () => {
  assert.deepEqual(findImagePathTokens("./a.png and ../b/c.gif"), ["./a.png", "../b/c.gif"]);
});

test("findImagePathTokens: multiple paths in one message", () => {
  const out = findImagePathTokens("compare /x.png and /y.jpeg please");
  assert.deepEqual(out, ["/x.png", "/y.jpeg"]);
});

test("findImagePathTokens: dedups repeated paths", () => {
  const out = findImagePathTokens("/x.png then /x.png again");
  assert.deepEqual(out, ["/x.png"]);
});

test("findImagePathTokens: bare filename without separator is NOT matched", () => {
  assert.deepEqual(findImagePathTokens("look at photo.png for me"), []);
  assert.deepEqual(findImagePathTokens("photo.png"), []);
});

test("findImagePathTokens: non-image extension is NOT matched", () => {
  assert.deepEqual(findImagePathTokens("/tmp/notes.txt"), []);
  assert.deepEqual(findImagePathTokens("/tmp/data.json"), []);
});

test("findImagePathTokens: all supported extensions", () => {
  const out = findImagePathTokens("/a.png /b.jpg /c.jpeg /d.gif /e.webp /f.bmp");
  assert.equal(out.length, 6);
});

test("findImagePathTokens: empty / no paths", () => {
  assert.deepEqual(findImagePathTokens(""), []);
  assert.deepEqual(findImagePathTokens("just a normal message with no images"), []);
});

test("findImagePathTokens: path adjacent to punctuation", () => {
  const out = findImagePathTokens("see (/tmp/x.png) and /tmp/y.png.");
  assert.ok(out.includes("/tmp/x.png"));
  assert.ok(out.some((p) => p.startsWith("/tmp/y.png")));
});

test("findImagePathTokens: URL with image ext is matched by regex but filtered later", () => {
  // The regex matches the path-like portion; resolveImageFile's existsSync
  // check filters non-local paths. Here we only assert the regex surfaces it.
  const out = findImagePathTokens("https://example.com/cat.png");
  assert.ok(out.length >= 1, "regex surfaces URL path-like token; existsSync filters it downstream");
});