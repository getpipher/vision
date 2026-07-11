/**
 * @getpither/vision — paste extension (SPEC-3: graduated from B-lite).
 *
 * Input event hook that makes pasted/referenced images visible + actionable
 * end-to-end, capability-aware for both multimodal AND text-only primaries.
 *
 * **Multimodal primary** (PASS-THROUGH): attaches images as native
 * attachments (B-lite delivery guarantee) + renders `[Image-#N]` markers in
 * the text (gap #6). Zero delegation — the model sees the image natively.
 *
 * **Text-only primary** (capability-aware branching, gap #8): does NOT
 * attach images (text-only models can't process them). Renders markers +
 * branches on `textOnlyPasteMode`:
 * - `"hint"` (default): markers + a hint line nudging the model to call
 *   `describe_image`. Zero tokens — the model decides.
 * - `"auto"` (opt-in): auto-delegates each image via the v0.2.x pipeline
 *   (cache/retry/fallback) + appends descriptions. Timeout-protected (own
 *   AbortController); falls back to hint on timeout/failure.
 * - `"off"`: markers only — no attachment, no hint, no delegation.
 *
 * Coexistence with pi-paster: pi-paster matches its own `[#image N]`
 * placeholders (different trigger); this hook matches file paths. Transforms
 * chain across handlers, and we dedup by data hash against `event.images`.
 *
 * Config + cache are shared via `lib/state.ts` (set by vision.ts on
 * session_start + every mutation).
 */
import { existsSync, statSync } from "node:fs";
import { isAbsolute, resolve as resolvePath } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ImageContent } from "@earendil-works/pi-ai";
import { isMultimodal } from "../lib/capability.ts";
import { loadImage } from "../lib/image.ts";
import { renderMarkers, buildHintLine, buildDescriptionsBlock } from "../lib/marker.ts";
import { getSharedConfig, getSharedCache } from "../lib/state.ts";
import { delegateToVisionModel, type DelegateParams } from "../lib/delegate.ts";
import type { ReasoningLevel } from "../lib/config.ts";
import { createComposePreviewComponent, makePreviewImage } from "../lib/preview.ts";
import { clearSharedState } from "../lib/state.ts";

const IMAGE_EXT_RE = /\.(png|jpe?g|gif|webp|bmp)$/i;
// Matches path-like tokens (absolute /…, home ~/…, relative ./…/…/) ending in a
// known image extension. Allows \ (escaped space) which is what terminal
// drag-and-drop produces on paths with spaces (common on macOS screenshots).
// Unescaped-space paths are handled by the post-regex extension step below.
const PATH_TOKEN_RE = /(?:\/|~\/|\.{1,2}\/)(?:\\ |[^\s)"'<>])+\.(?:png|jpe?g|gif|webp|bmp)/gi;

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
 *  image file, else undefined. Unescapes \ (escaped spaces from terminal
 *  drag-and-drop) before resolving. */
function resolveImageFile(token: string, cwd: string): string | undefined {
  // Unescape \ → space (terminal drag-paste escaping)
  const unescaped = token.replace(/\\ /g, " ");
  const expanded = unescaped.startsWith("~/") ? resolvePath(cwd, unescaped) : unescaped;
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

/** A loaded image ready for attachment or delegation. */
interface LoadedImage {
  token: string;
  abs: string;
  data: string;
  mimeType: string;
  hash: string;
}

/** Load + dedup all resolvable tokens. Returns the loaded images + the
 *  set of hashes already seen (for dedup). */
async function loadAndDedup(
  tokens: string[],
  existingImages: ImageContent[],
  cwd: string,
): Promise<{ loaded: LoadedImage[]; existingHashes: Set<string> }> {
  const existingHashes = new Set(existingImages.map((img) => hashData(img.data)));
  const loaded: LoadedImage[] = [];
  const seenPaths = new Set<string>();

  for (const token of tokens) {
    const abs = resolveImageFile(token, cwd);
    if (!abs || seenPaths.has(abs)) continue;
    seenPaths.add(abs);
    const result = await loadImage(abs, {
      compress: true,
      maxDimension: 1568,
      jpegQuality: 85,
      cwd,
    });
    if (!result.ok) continue; // skip unreadable images
    const hash = hashData(result.image.data);
    if (existingHashes.has(hash)) continue; // already attached (e.g. by pi-paster)
    existingHashes.add(hash);
    loaded.push({ token, abs, data: result.image.data, mimeType: result.image.mimeType, hash });
  }

  return { loaded, existingHashes };
}

/** Build the resolved map for renderMarkers: token → index in the final
 *  images array. For multimodal, index = offset + position among loaded.
 *  For dedup'd images (hash matches existing), the marker points to the
 *  existing index. For text-only (no attachment), markers are sequential
 *  starting from the offset. */
function buildResolvedMap(
  tokens: string[],
  loaded: LoadedImage[],
  existingImages: ImageContent[],
  isMultimodalModel: boolean,
): Map<string, { index: number }> {
  const resolved = new Map<string, { index: number }>();
  const offset = existingImages.length;

  // For dedup: build a hash → existing-index lookup.
  const existingHashToIndex = new Map<string, number>();
  existingImages.forEach((img, i) => {
    existingHashToIndex.set(hashData(img.data), i);
  });

  let newImageIndex = 0;
  for (const token of tokens) {
    // Find if this token was loaded.
    const found = loaded.find((l) => l.token === token);
    if (!found) continue; // unresolvable → not in resolved map → left as-is

    // Check if it was dedup'd (hash matches an existing image).
    const existingIdx = existingHashToIndex.get(found.hash);
    if (existingIdx !== undefined) {
      resolved.set(token, { index: existingIdx });
    } else {
      // New image.
      const idx = isMultimodalModel ? offset + newImageIndex : newImageIndex;
      resolved.set(token, { index: idx });
      newImageIndex++;
    }
  }

  return resolved;
}

/** Auto-delegate a single image with timeout protection. Returns the
 *  description text or undefined on failure/timeout (caller falls back to hint). */
async function autoDelegateOne(
  ctx: ExtensionContext,
  config: NonNullable<ReturnType<typeof getSharedConfig>>,
  image: LoadedImage,
  cache: NonNullable<ReturnType<typeof getSharedCache>>,
): Promise<{ text: string; cached: boolean } | undefined> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.autoDelegateTimeoutMs);
  try {
    const params: DelegateParams = {
      image_path: image.abs,
      prompt: config.autoDelegatePrompt,
      compress: true,
      reasoning: "off" as ReasoningLevel,
    };
    const result = await delegateToVisionModel(ctx, config, params, controller.signal, cache);
    if (result.ok) {
      return { text: result.text, cached: result.details.cached };
    }
    return undefined; // failure → caller falls back to hint
  } catch {
    return undefined; // timeout or error → hint fallback
  } finally {
    clearTimeout(timer);
  }
}

/** Debounce timer for compose-time preview. */
let composeDebounceTimer: ReturnType<typeof setTimeout> | undefined;
/** Unsubscribe function for the onTerminalInput handler. */
let terminalInputUnsub: (() => void) | undefined;
/** The last set of tokens we previewed (to avoid re-rendering the same). */
let lastPreviewTokens: string = "";

/**
 * Compose-time auto-preview: while the user is typing, detect image paths in
 * the editor text and render a preview above the editor (WhatsApp style).
 * Debounced ~300ms after the last keystroke. Clears on submit or path removal.
 */
async function updateComposePreview(ctx: ExtensionContext): Promise<void> {
  const config = getSharedConfig();
  if (!config || !config.composePreview) {
    ctx.ui.setWidget("vision-compose-preview", undefined);
    return;
  }
  if (ctx.mode !== "tui") return; // widget is TUI-only

  const editorText = ctx.ui.getEditorText();
  // Don't show compose preview for slash commands — they have their own UI
  if (editorText.trim().startsWith("/")) {
    if (lastPreviewTokens) {
      ctx.ui.setWidget("vision-compose-preview", undefined);
      lastPreviewTokens = "";
    }
    return;
  }
  const tokens = findImagePathTokens(editorText);
  const tokensKey = tokens.join("\0");

  if (tokens.length === 0) {
    if (lastPreviewTokens) {
      ctx.ui.setWidget("vision-compose-preview", undefined);
      lastPreviewTokens = "";
    }
    return;
  }

  // Skip if same tokens as last time (no re-render needed)
  if (tokensKey === lastPreviewTokens) return;
  lastPreviewTokens = tokensKey;

  // Resolve + load each token (compress: false — show original quality)
  const previewImages: ReturnType<typeof makePreviewImage>[] = [];
  for (const token of tokens) {
    const unescaped = token.replace(/\\ /g, " ");
    const abs = isAbsolute(unescaped) ? unescaped : resolvePath(ctx.cwd, unescaped);
    if (!existsSync(abs)) continue;
    const result = await loadImage(abs, {
      compress: false,
      maxDimension: 1568,
      jpegQuality: 85,
      cwd: ctx.cwd,
    });
    if (!result.ok) continue;
    previewImages.push(makePreviewImage(result.image.data, result.image.mimeType, abs));
  }

  if (previewImages.length === 0) {
    ctx.ui.setWidget("vision-compose-preview", undefined);
    return;
  }

  // Set the widget above the editor
  ctx.ui.setWidget(
    "vision-compose-preview",
    (tui, theme) => createComposePreviewComponent(
      previewImages,
      (c: string, t: string) => theme.fg(c as any, t),
      config.previewMaxWidthCells,
    ),
    { placement: "aboveEditor" },
  );
}

export default function pasteExtension(_pi: ExtensionAPI): void {
  // ── Compose-time auto-preview (gap #7) ───────────────────────────────────
  _pi.on("session_start", (_event, ctx) => {
    // Register terminal input listener for compose-time preview
    if (terminalInputUnsub) terminalInputUnsub();
    if (ctx.hasUI && ctx.mode === "tui") {
      terminalInputUnsub = ctx.ui.onTerminalInput((_data: string) => {
        // Debounce: clear existing timer, set new one
        if (composeDebounceTimer) clearTimeout(composeDebounceTimer);
        composeDebounceTimer = setTimeout(() => {
          updateComposePreview(ctx).catch(() => {});
        }, 300);
        return undefined; // don't consume — let the editor process normally
      });
    }
  });

  _pi.on("session_shutdown", () => {
    if (composeDebounceTimer) {
      clearTimeout(composeDebounceTimer);
      composeDebounceTimer = undefined;
    }
    if (terminalInputUnsub) {
      terminalInputUnsub();
      terminalInputUnsub = undefined;
    }
    lastPreviewTokens = "";
  });

  _pi.on("input", async (event, ctx) => {
    // Clear compose preview on submit (only if setWidget is available — TUI mode)
    if (event.source !== "extension" && typeof ctx.ui.setWidget === "function") {
      ctx.ui.setWidget("vision-compose-preview", undefined);
      lastPreviewTokens = "";
    }
    // Don't re-process messages we (or another extension) injected.
    if (event.source === "extension") return { action: "continue" as const };

    const config = getSharedConfig();
    if (!config) return { action: "continue" as const }; // before session_start

    const tokens = findImagePathTokens(event.text);
    if (tokens.length === 0) return { action: "continue" as const };

    const multimodal = isMultimodal(ctx.model);
    const { loaded, existingHashes } = await loadAndDedup(
      tokens,
      event.images ?? [],
      ctx.cwd,
    );

    if (loaded.length === 0) return { action: "continue" as const };

    // Build the resolved map (token → index for marker numbering).
    const resolved = buildResolvedMap(tokens, loaded, event.images ?? [], multimodal);

    // Render markers in the text.
    let text = renderMarkers(event.text, tokens, resolved, config.markerStyle);

    if (multimodal) {
      // ── MULTIMODAL: attach images + markers (B-lite delivery + gap #6) ──
      const newImages: ImageContent[] = loaded.map((l) => ({
        type: "image" as const,
        data: l.data,
        mimeType: l.mimeType,
      }));
      return {
        action: "transform" as const,
        text,
        images: [...(event.images ?? []), ...newImages],
      };
    }

    // ── TEXT-ONLY: no attachment (text-only models can't process images) ──
    const mode = config.textOnlyPasteMode;

    if (mode === "off") {
      // Markers only — no attachment, no hint, no delegation.
      return { action: "transform" as const, text };
    }

    if (mode === "hint") {
      // Markers + hint line nudging the model to call describe_image.
      text = `${text}\n${buildHintLine(loaded.length)}`;
      return { action: "transform" as const, text };
    }

    // mode === "auto": auto-delegate each image + append descriptions.
    const cache = getSharedCache();
    const visionModel = config.provider && config.model ? `${config.provider}/${config.model}` : "(unconfigured)";

    if (!cache || !config.provider || !config.model) {
      // Can't delegate (no cache or unconfigured) → fall back to hint.
      text = `${text}\n${buildHintLine(loaded.length)}`;
      return { action: "transform" as const, text };
    }

    const descriptions: Array<{ token: string; index: number; text: string; cached: boolean }> = [];
    let allFailed = true;

    for (const image of loaded) {
      const result = await autoDelegateOne(ctx, config, image, cache);
      if (result) {
        const idx = resolved.get(image.token)?.index ?? 0;
        descriptions.push({ token: image.token, index: idx, text: result.text, cached: result.cached });
        allFailed = false;
      }
      // On undefined (failure/timeout) → that image gets no description.
      // If ALL fail, we fall back to hint below.
    }

    if (allFailed) {
      // All delegations failed → hint fallback for all images.
      text = `${text}\n${buildHintLine(loaded.length)}`;
      return { action: "transform" as const, text };
    }

    // Append the descriptions block.
    text = `${text}${buildDescriptionsBlock(descriptions, visionModel)}`;
    return { action: "transform" as const, text };
  });
}