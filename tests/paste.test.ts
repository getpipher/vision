import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { findImagePathTokens, resolveImageFile } from "../extensions/paste.ts";

// 1×1 transparent PNG (same fixture as tests/image.test.ts).
const PNG_BYTES = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M8AAAMBEg1+mP0AAAAASUVORK5CYII=",
  "base64",
);

function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-paste-"));
}

/** Run `fn` with $HOME pointed at a fresh temp dir, restoring it afterwards. */
function withTempHome(fn: (home: string) => void): void {
  const home = tmpDir();
  const origHome = process.env.HOME;
  process.env.HOME = home;
  try {
    fn(home);
  } finally {
    if (origHome === undefined) delete process.env.HOME;
    else process.env.HOME = origHome;
    rmSync(home, { recursive: true, force: true });
  }
}

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
test("findImagePathTokens: escaped-space path (terminal drag-paste on macOS)", () => {
  const out = findImagePathTokens("/var/folders/NSIRD_screencaptureui/Screenshot\\ 2026-07-10\\ at\\ 10.30.43.png");
  assert.equal(out.length, 1);
  assert.ok(out[0]!.includes("Screenshot\\ 2026-07-10\\ at\\ 10.30.43.png"));
});

test("findImagePathTokens: escaped-space path followed by more text", () => {
  const out = findImagePathTokens("analyze /tmp/My\\ Screenshot.png for bugs");
  assert.equal(out.length, 1);
  assert.equal(out[0], "/tmp/My\\ Screenshot.png");
});

test("findImagePathTokens: escaped + regular paths mixed", () => {
  const out = findImagePathTokens("compare /tmp/a.png and /tmp/My\\ B.jpeg");
  assert.equal(out.length, 2);
  assert.equal(out[0], "/tmp/a.png");
  assert.equal(out[1], "/tmp/My\\ B.jpeg");
});

// ── resolveImageFile: tilde expansion (fix) ────────────────────────────────
// Previously `~/x.png` was resolved as `<cwd>/~/x.png`, so a home-relative
// path pasted into the editor never resolved: no [Image-#N] marker, no hint
// line, no compose preview, no auto-delegation. It only appeared to work
// because the LLM expanded the tilde itself before calling describe_image.

test("resolveImageFile: ~/ resolves against $HOME, not cwd", () => {
  withTempHome((home) => {
    const file = join(home, "shot.png");
    writeFileSync(file, PNG_BYTES);
    assert.equal(resolveImageFile("~/shot.png", join(home, "unrelated")), file);
  });
});

test("resolveImageFile: ~/ with escaped spaces (drag-paste of a home path)", () => {
  withTempHome((home) => {
    const file = join(home, "My Screenshot.png");
    writeFileSync(file, PNG_BYTES);
    assert.equal(resolveImageFile("~/My\\ Screenshot.png", "/tmp"), file);
  });
});

test("resolveImageFile: absolute + relative paths still resolve", () => {
  const dir = tmpDir();
  try {
    const file = join(dir, "a.png");
    writeFileSync(file, PNG_BYTES);
    assert.equal(resolveImageFile(file, "/tmp"), file);
    assert.equal(resolveImageFile("./a.png", dir), file);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("resolveImageFile: nonexistent path → undefined", () => {
  assert.equal(resolveImageFile("/nope/definitely-missing.png", "/tmp"), undefined);
});
