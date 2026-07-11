import { test } from "node:test";
import assert from "node:assert/strict";
import {
  detectProtocol,
  formatFileSize,
  formatImageMetadata,
  buildImageTheme,
  makePreviewImage,
  type PreviewImage,
} from "../lib/preview.ts";

// ── formatFileSize ───────────────────────────────────────────────────────

test("formatFileSize: bytes", () => {
  assert.equal(formatFileSize(500), "500B");
  assert.equal(formatFileSize(0), "0B");
});

test("formatFileSize: kilobytes", () => {
  assert.equal(formatFileSize(1024), "1.0KB");
  assert.equal(formatFileSize(1536), "1.5KB");
});

test("formatFileSize: megabytes", () => {
  assert.equal(formatFileSize(1024 * 1024), "1.0MB");
  assert.equal(formatFileSize(2.5 * 1024 * 1024), "2.5MB");
});

// ── detectProtocol ───────────────────────────────────────────────────────

test("detectProtocol: returns a valid protocol string", () => {
  const p = detectProtocol();
  assert.ok(["kitty", "iterm2", "text fallback"].includes(p), `got: ${p}`);
});

// ── formatImageMetadata ─────────────────────────────────────────────────

test("formatImageMetadata: full metadata line", () => {
  const img: PreviewImage = {
    data: "AAAA",
    mimeType: "image/png",
    filename: "screenshot.png",
    dimensions: { widthPx: 1920, heightPx: 1080 },
    sizeBytes: 45518,
  };
  const meta = formatImageMetadata(img, "kitty");
  assert.match(meta, /screenshot\.png/);
  assert.match(meta, /1920x1080/);
  assert.match(meta, /image\/png/);
  assert.match(meta, /44\.5KB/); // 45518 bytes = 44.5KB
  assert.match(meta, /kitty/);
});

test("formatImageMetadata: null dimensions → ?", () => {
  const img: PreviewImage = {
    data: "AAAA",
    mimeType: "image/png",
    filename: "test.png",
    dimensions: null,
    sizeBytes: 100,
  };
  const meta = formatImageMetadata(img, "text fallback");
  assert.match(meta, /\?/);
  assert.match(meta, /text fallback/);
});

// ── buildImageTheme ──────────────────────────────────────────────────────

test("buildImageTheme: returns ImageTheme with fallbackColor function", () => {
  const mockFg = (color: string, text: string) => `[${color}]${text}`;
  const theme = buildImageTheme(mockFg);
  assert.equal(typeof theme.fallbackColor, "function");
  const result = theme.fallbackColor("test");
  assert.equal(result, "[dim]test");
});

// ── makePreviewImage ─────────────────────────────────────────────────────

test("makePreviewImage: from base64 data + mimeType", () => {
  // Minimal 1x1 PNG
  const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
  const img = makePreviewImage(b64, "image/png");
  assert.equal(img.mimeType, "image/png");
  assert.equal(img.filename, "(unknown)");
  assert.ok(img.dimensions !== null, "dimensions should be detected for a valid PNG");
  if (img.dimensions) {
    assert.equal(img.dimensions.widthPx, 1);
    assert.equal(img.dimensions.heightPx, 1);
  }
});

import { writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

test("makePreviewImage: with filepath → filename + sizeBytes", () => {
  const path = join(tmpdir(), "vision-preview-test.png");
  const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
  writeFileSync(path, Buffer.from(b64, "base64"));
  try {
    const img = makePreviewImage(b64, "image/png", path);
    assert.equal(img.filename, "vision-preview-test.png");
    assert.ok(img.sizeBytes > 0, "sizeBytes should be > 0 for a real file");
  } finally {
    unlinkSync(path);
  }
});

test("makePreviewImage: invalid base64 → null dimensions", () => {
  const img = makePreviewImage("not-valid-base64", "image/png");
  assert.equal(img.dimensions, null);
});