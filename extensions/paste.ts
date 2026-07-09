/**
 * @getpipher/vision — paste extension (B-lite PASS-THROUGH delivery).
 *
 * Minimal `input` hook that GUARANTEES image file paths referenced in the
 * user's message reach a multimodal primary model as native attachments —
 * regardless of whether the LLM would otherwise choose the built-in `read`
 * tool. This makes SPEC-1 T1 (0 delegation for multimodal primary) pass by
 * construction.
 *
 * Scope (v0.1.0): PASS-THROUGH delivery ONLY. No clipboard handling, no
 * `[Image-#N]` markers, no colors, no TUI preview, no editor component — all
 * of that is SPEC-3. This hook fires only when the active model is
 * multimodal; text-only models use `describe_image` (DELEGATE) instead.
 *
 * Coexistence with pi-paster: pi-paster matches its own `[#image N]`
 * placeholders (different trigger); this hook matches file paths. Transforms
 * chain across handlers, and we dedup by data hash against `event.images` so
 * the same image is never attached twice.
 */
import { existsSync, statSync } from "node:fs";
import { isAbsolute, resolve as resolvePath } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { ImageContent } from "@earendil-works/pi-ai";
import { isMultimodal } from "../lib/capability.ts";
import { loadImage } from "../lib/image.ts";

const IMAGE_EXT_RE = /\.(png|jpe?g|gif|webp|bmp)$/i;
const PATH_TOKEN_RE = /(?:\/|~\/|\.{1,2}\/)[^\s)"'<>]+\.(?:png|jpe?g|gif|webp|bmp)/gi;

/**
 * Extract candidate image file-path tokens from free text. Matches path-like
 * tokens (absolute `/…`, home `~/…`, or relative `./…`/`../…`) that end in a
 * known image extension. Bare filenames without a path separator are
 * deliberately not matched to avoid false positives on ordinary words. URLs
 * (`http://…`) can match the pattern but are filtered out later by the
 * `existsSync` check in `resolveImageFile`.
 */
export function findImagePathTokens(text: string): string[] {
  const out: string[] = [];
  PATH_TOKEN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = PATH_TOKEN_RE.exec(text)) !== null) {
    out.push(m[0]);
  }
  // Dedup, preserve order.
  return [...new Set(out)];
}

/** Resolve a token against cwd and return the absolute path if it's a real
 *  image file, else undefined. */
function resolveImageFile(token: string, cwd: string): string | undefined {
  const expanded = token.startsWith("~/") ? resolvePath(cwd, token) : token;
  const abs = isAbsolute(expanded) ? expanded : resolvePath(cwd, expanded);
  if (!existsSync(abs)) return undefined;
  try {
    if (!statSync(abs).isFile()) return undefined;
  } catch {
    return undefined;
  }
  if (!IMAGE_EXT_RE.test(abs)) return undefined;
  return abs;
}

/** Cheap FNV-1a hash over the base64 string, for dedup against existing
 *  attachments (e.g. images pi-paster already attached). */
function hashData(data: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < data.length; i++) {
    h ^= data.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

export default function pasteExtension(_pi: ExtensionAPI): void {
  _pi.on("input", async (event, ctx) => {
    // Don't re-process messages we (or another extension) injected.
    if (event.source === "extension") return { action: "continue" as const };
    // Only attach for multimodal primary models; text-only models use
    // describe_image (DELEGATE) and would waste tokens on attachments they
    // can't process.
    if (!isMultimodal(ctx.model)) return { action: "continue" as const };

    const tokens = findImagePathTokens(event.text);
    if (tokens.length === 0) return { action: "continue" as const };

    const existingHashes = new Set((event.images ?? []).map((img) => hashData(img.data)));
    const newImages: ImageContent[] = [];
    const seenPaths = new Set<string>();

    for (const token of tokens) {
      const abs = resolveImageFile(token, ctx.cwd);
      if (!abs || seenPaths.has(abs)) continue;
      seenPaths.add(abs);
      const loaded = await loadImage(abs, {
        compress: true,
        maxDimension: 1568,
        jpegQuality: 85,
        cwd: ctx.cwd,
      });
      if (!loaded.ok) continue; // skip unreadable images; the tool path errors clearly
      const hash = hashData(loaded.image.data);
      if (existingHashes.has(hash)) continue; // already attached (e.g. by pi-paster)
      existingHashes.add(hash);
      newImages.push({
        type: "image",
        data: loaded.image.data,
        mimeType: loaded.image.mimeType,
      });
    }

    if (newImages.length === 0) return { action: "continue" as const };

    return {
      action: "transform" as const,
      text: event.text,
      images: [...(event.images ?? []), ...newImages],
    };
  });
}