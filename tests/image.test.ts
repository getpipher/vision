import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { detectMimeType, hashBytes, loadImage, MAX_IMAGE_BYTES } from "../lib/image.ts";

// 1×1 transparent PNG — decodes to bytes starting with the PNG signature
// (89 50 4E 47 0D 0A 1A 0A).
const PNG_1x1_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M8AAAMBEg1+mP0AAAAASUVORK5CYII=";
const PNG_BYTES = Buffer.from(PNG_1x1_B64, "base64");

const LOAD_OPTS = {
  compress: false,
  maxDimension: 1568,
  jpegQuality: 85,
  cwd: "/tmp",
};

function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-img-"));
}

test("detectMimeType: PNG magic bytes", () => {
  assert.equal(detectMimeType(PNG_BYTES), "image/png");
});

test("detectMimeType: JPEG magic bytes", () => {
  const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
  assert.equal(detectMimeType(jpeg), "image/jpeg");
});

test("detectMimeType: GIF magic bytes", () => {
  const gif = Buffer.from([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00]);
  assert.equal(detectMimeType(gif), "image/gif");
});

test("detectMimeType: WebP magic bytes (RIFF....WEBP)", () => {
  const webp = Buffer.from([
    0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
  ]);
  assert.equal(detectMimeType(webp), "image/webp");
});

test("detectMimeType: unknown bytes → undefined", () => {
  assert.equal(detectMimeType(Buffer.from("hello world", "utf8")), undefined);
  assert.equal(detectMimeType(Buffer.alloc(0)), undefined);
});

test("MAX_IMAGE_BYTES is 64 MB", () => {
  assert.equal(MAX_IMAGE_BYTES, 64 * 1024 * 1024);
});

test("loadImage: file path → reads + detects PNG", async () => {
  const dir = tmpDir();
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const r = await loadImage(file, { ...LOAD_OPTS, cwd: dir });
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.image.mimeType, "image/png");
      assert.equal(r.image.data, PNG_1x1_B64);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: relative path resolves against cwd", async () => {
  const dir = tmpDir();
  try {
    writeFileSync(join(dir, "pixel.png"), PNG_BYTES);
    const r = await loadImage("./pixel.png", { ...LOAD_OPTS, cwd: dir });
    assert.equal(r.ok, true);
    if (r.ok) assert.equal(r.image.mimeType, "image/png");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: data URL → parses base64 + mime", async () => {
  const r = await loadImage(`data:image/png;base64,${PNG_1x1_B64}`, LOAD_OPTS);
  assert.equal(r.ok, true);
  if (r.ok) {
    assert.equal(r.image.mimeType, "image/png");
    assert.equal(r.image.data, PNG_1x1_B64);
  }
});

test("loadImage: data URL without explicit mime → detects from bytes", async () => {
  const r = await loadImage(`data:;base64,${PNG_1x1_B64}`, LOAD_OPTS);
  assert.equal(r.ok, true);
  if (r.ok) assert.equal(r.image.mimeType, "image/png");
});

test("loadImage: raw base64 → decodes + detects mime", async () => {
  const r = await loadImage(PNG_1x1_B64, LOAD_OPTS);
  assert.equal(r.ok, true);
  if (r.ok) assert.equal(r.image.mimeType, "image/png");
});

test("loadImage: nonexistent path → not_found", async () => {
  const r = await loadImage(join(tmpdir(), "nonexistent-vision-test.png"), LOAD_OPTS);
  assert.equal(r.ok, false);
  if (!r.ok) assert.equal(r.error.code, "not_found");
});

test("loadImage: directory path → not_a_file", async () => {
  const dir = tmpDir();
  try {
    const r = await loadImage(dir, { ...LOAD_OPTS, cwd: dir });
    assert.equal(r.ok, false);
    if (!r.ok) assert.equal(r.error.code, "not_a_file");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: file with unknown format → unsupported_format", async () => {
  const dir = tmpDir();
  try {
    const file = join(dir, "fake.png");
    writeFileSync(file, "this is not an image");
    const r = await loadImage(file, { ...LOAD_OPTS, cwd: dir });
    assert.equal(r.ok, false);
    if (!r.ok) assert.equal(r.error.code, "unsupported_format");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: malformed data URL → invalid_data_url", async () => {
  const r = await loadImage("data:not-a-data-url", LOAD_OPTS);
  assert.equal(r.ok, false);
  if (!r.ok) assert.equal(r.error.code, "invalid_data_url");
});

test("loadImage: non-base64 data URL → invalid_data_url", async () => {
  const r = await loadImage("data:image/png,hello", LOAD_OPTS);
  assert.equal(r.ok, false);
  if (!r.ok) assert.equal(r.error.code, "invalid_data_url");
});

test("loadImage: compress=true returns ok (compressed or gracefully degraded)", async () => {
  const dir = tmpDir();
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const r = await loadImage(file, { compress: true, maxDimension: 100, jpegQuality: 80, cwd: dir });
    assert.equal(r.ok, true, "compress must never turn a valid image into a failure");
    if (r.ok) {
      assert.ok(r.image.data.length > 0);
      assert.ok(r.image.mimeType.startsWith("image/"));
      assert.equal(r.sourceHash.length, 64, "sourceHash is sha256 hex (64 chars)");
      assert.match(r.sourceHash, /^[0-9a-f]{64}$/, "sourceHash is lowercase hex");
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: sourceHash is stable across two loads of the same file", async () => {
  const dir = tmpDir();
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const a = await loadImage(file, { ...LOAD_OPTS, cwd: dir });
    const b = await loadImage(file, { ...LOAD_OPTS, cwd: dir });
    assert.equal(a.ok, true);
    assert.equal(b.ok, true);
    if (a.ok && b.ok) assert.equal(a.sourceHash, b.sourceHash, "same bytes → same hash");
    if (a.ok) assert.equal(a.sourceHash, hashBytes(PNG_BYTES), "matches direct hashBytes");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: different bytes → different sourceHash", async () => {
  const dir = tmpDir();
  try {
    writeFileSync(join(dir, "a.png"), PNG_BYTES);
    const otherBytes = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBApDk1+wAAAAASUVORK5CYII=",
      "base64",
    );
    writeFileSync(join(dir, "b.png"), otherBytes);
    const a = await loadImage(join(dir, "a.png"), { ...LOAD_OPTS, cwd: dir });
    const b = await loadImage(join(dir, "b.png"), { ...LOAD_OPTS, cwd: dir });
    assert.equal(a.ok, true);
    assert.equal(b.ok, true);
    if (a.ok && b.ok) assert.notEqual(a.sourceHash, b.sourceHash, "different bytes → different hash");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("loadImage: data URL + raw base64 both return sourceHash", async () => {
  const viaUrl = await loadImage(`data:image/png;base64,${PNG_1x1_B64}`, LOAD_OPTS);
  const viaRaw = await loadImage(PNG_1x1_B64, LOAD_OPTS);
  assert.equal(viaUrl.ok, true);
  assert.equal(viaRaw.ok, true);
  if (viaUrl.ok) assert.equal(viaUrl.sourceHash, hashBytes(PNG_BYTES), "data URL hash matches bytes");
  if (viaRaw.ok) assert.equal(viaRaw.sourceHash, hashBytes(PNG_BYTES), "raw base64 hash matches bytes");
  if (viaUrl.ok && viaRaw.ok)
    assert.equal(viaUrl.sourceHash, viaRaw.sourceHash, "same bytes via different input shapes → same hash");
});

test("hashBytes: empty + known vector", () => {
  assert.equal(hashBytes(Buffer.alloc(0)), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert.equal(hashBytes(PNG_BYTES).length, 64);
});