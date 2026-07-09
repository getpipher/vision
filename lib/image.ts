/**
 * Image loading + compression for the vision tool.
 *
 * Clean-room: the input surface (file path / data URL / raw base64) matches
 * what users already pass to pi-vision-tool, but the implementation is ours.
 * Compression delegates to pi's built-in `resizeImage` (Photon/WASM, runs in
 * a worker) so we ship zero native image deps. If the resizer is unavailable
 * we degrade gracefully and return the original bytes.
 */
import { readFile, stat } from "node:fs/promises";
import { existsSync, statSync } from "node:fs";
import { isAbsolute, resolve as resolvePath } from "node:path";
import { resizeImage } from "@earendil-works/pi-coding-agent";

/** Cap on source file size (64 MB) — reject up front so we never base64-encode
 *  a multi-gigabyte file. Matches pi-paster's source cap. */
export const MAX_IMAGE_BYTES = 64 * 1024 * 1024;

export interface LoadedImage {
  /** base64-encoded image data (no data: prefix). */
  data: string;
  /** MIME type, e.g. "image/png". */
  mimeType: string;
}

export type ImageLoadErrorCode =
  | "not_found"
  | "not_a_file"
  | "too_large"
  | "unsupported_format"
  | "read_error"
  | "invalid_data_url"
  | "invalid_base64";

export interface ImageLoadError {
  code: ImageLoadErrorCode;
  path?: string;
  size?: number;
  message?: string;
}

export type ImageLoadResult =
  | { ok: true; image: LoadedImage }
  | { ok: false; error: ImageLoadError };

export interface LoadOptions {
  /** Run compression (resize + re-encode) on the loaded image. */
  compress: boolean;
  /** Max long-edge dimension in pixels. */
  maxDimension: number;
  /** JPEG re-encode quality (1–100). */
  jpegQuality: number;
  /** Working directory for resolving relative paths. */
  cwd: string;
}

const SUPPORTED_MIME = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

/** Detect MIME type from magic bytes. Returns undefined for unknown formats. */
export function detectMimeType(bytes: Uint8Array): string | undefined {
  if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47 && bytes[4] === 0x0d && bytes[5] === 0x0a && bytes[6] === 0x1a && bytes[7] === 0x0a) {
    return "image/png";
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg";
  }
  if (bytes.length >= 6 && bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x38 && (bytes[4] === 0x37 || bytes[4] === 0x39) && bytes[5] === 0x61) {
    return "image/gif";
  }
  // WebP: RIFF....WEBP
  if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 && bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) {
    return "image/webp";
  }
  return undefined;
}

const DATA_URL_RE = /^data:([^;,]+)?(;base64)?,(.*)$/s;

/** Parse a `data:<mime>;base64,<data>` URL. */
function parseDataUrl(input: string): ImageLoadResult {
  const m = DATA_URL_RE.exec(input);
  if (!m) {
    return { ok: false, error: { code: "invalid_data_url", message: "malformed data URL" } };
  }
  const declaredMime = m[1];
  const isBase64 = !!m[2];
  const payload = m[3] ?? "";
  if (!isBase64) {
    return { ok: false, error: { code: "invalid_data_url", message: "only base64 data URLs are supported" } };
  }
  let bytes: Buffer;
  try {
    bytes = Buffer.from(payload, "base64");
  } catch {
    return { ok: false, error: { code: "invalid_data_url", message: "could not base64-decode data URL payload" } };
  }
  const mime = declaredMime && SUPPORTED_MIME.has(declaredMime) ? declaredMime : detectMimeType(bytes);
  if (!mime) {
    return { ok: false, error: { code: "unsupported_format", message: "could not determine image format from data URL" } };
  }
  return { ok: true, image: { data: payload, mimeType: mime } };
}

/** Is `input` plausibly a file path we should try to read (vs raw base64)? */
function looksLikeFilePath(input: string): boolean {
  if (input.startsWith("data:")) return false;
  // Absolute, or relative-with-separator, or tilde — treat as a path candidate.
  if (isAbsolute(input) || input.startsWith("./") || input.startsWith("../") || input.startsWith("~/")) {
    return true;
  }
  // Bare filename with a known image extension → also a path candidate.
  return /\.(png|jpe?g|gif|webp|bmp)$/i.test(input);
}

/** Decode a raw base64 string into bytes + detected mime. */
function decodeBase64(input: string): ImageLoadResult {
  let bytes: Buffer;
  try {
    bytes = Buffer.from(input, "base64");
  } catch {
    return { ok: false, error: { code: "invalid_base64", message: "could not base64-decode input" } };
  }
  const mime = detectMimeType(bytes);
  if (!mime) {
    return { ok: false, error: { code: "unsupported_format", message: "could not determine image format from base64 bytes" } };
  }
  return { ok: true, image: { data: bytes.toString("base64"), mimeType: mime } };
}

/**
 * Load an image from a file path, data URL, or raw base64 string. When
 * `options.compress` is true, resize + re-encode via pi's `resizeImage`
 * (Photon/WASM); if the resizer is unavailable, return the original bytes.
 */
export async function loadImage(input: string, options: LoadOptions): Promise<ImageLoadResult> {
  if (input.startsWith("data:")) {
    const parsed = parseDataUrl(input);
    if (!parsed.ok) return parsed;
    return compressIfRequested(parsed.image, options);
  }

  if (looksLikeFilePath(input)) {
    const abs = resolvePath(options.cwd, input.replace(/^~/, ""));
    if (!existsSync(abs)) {
      // A path-looking string that doesn't exist might still be raw base64
      // (rare). Fall through to base64 decode rather than hard-failing.
      const fallback = decodeBase64(input);
      if (fallback.ok) return compressIfRequested(fallback.image, options);
      return { ok: false, error: { code: "not_found", path: input } };
    }
    let st;
    try {
      st = statSync(abs);
    } catch (err) {
      return { ok: false, error: { code: "read_error", path: input, message: errorMessage(err) } };
    }
    if (!st.isFile()) {
      return { ok: false, error: { code: "not_a_file", path: input } };
    }
    if (st.size > MAX_IMAGE_BYTES) {
      return { ok: false, error: { code: "too_large", path: input, size: st.size } };
    }
    let bytes: Buffer;
    try {
      bytes = await readFile(abs);
    } catch (err) {
      return { ok: false, error: { code: "read_error", path: input, message: errorMessage(err) } };
    }
    const mime = detectMimeType(bytes);
    if (!mime) {
      return { ok: false, error: { code: "unsupported_format", path: input } };
    }
    return compressIfRequested({ data: bytes.toString("base64"), mimeType: mime }, options);
  }

  // Not a path, not a data URL → treat as raw base64.
  const decoded = decodeBase64(input);
  if (!decoded.ok) return decoded;
  return compressIfRequested(decoded.image, options);
}

async function compressIfRequested(
  image: LoadedImage,
  options: LoadOptions,
): Promise<ImageLoadResult> {
  if (!options.compress) return { ok: true, image };
  try {
    const inputBytes = Buffer.from(image.data, "base64");
    const resized = await resizeImage(inputBytes, image.mimeType, {
      maxWidth: options.maxDimension,
      maxHeight: options.maxDimension,
      jpegQuality: options.jpegQuality,
    });
    if (resized) {
      return { ok: true, image: { data: resized.data, mimeType: resized.mimeType } };
    }
  } catch {
    // resizeImage threw (e.g. Photon unavailable) → degrade to original.
  }
  return { ok: true, image };
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}