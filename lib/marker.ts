/**
 * Marker rendering for the paste UX (SPEC-3 gap #6).
 *
 * Pure functions that rewrite image file-path tokens into `[Image-#N]`
 * markers styled with markdown syntax (option B-prime — no ANSI, so the
 * markers are clean for both the TUI Markdown renderer AND the model, which
 * receives the markdown syntax as readable text). Also builds the hint line
 * (text-only + "hint" mode) and the descriptions block (text-only + "auto"
 * mode).
 *
 * No I/O, no pi runtime dependency — fully unit-testable.
 */

/** The markdown style wrapped around a `[Image-#N]` marker. */
export type MarkerStyle = "code" | "bold" | "plain";

export const MARKER_STYLES: readonly MarkerStyle[] = ["code", "bold", "plain"] as const;

/** Check whether a value is a valid MarkerStyle. */
export function isMarkerStyle(value: unknown): value is MarkerStyle {
  return typeof value === "string" && (MARKER_STYLES as readonly string[]).includes(value);
}

/**
 * Wrap a `[Image-#N]` marker in the configured markdown style.
 *
 * - `"code"` → `` `[Image-#1]` `` (inline code — TUI renders with mdCode
 *   color + monospace; model sees readable inline code)
 * - `"bold"` → `**[Image-#1]**` (bold emphasis)
 * - `"plain"` → `[Image-#1]` (no wrapping — blends into body text)
 */
export function styleMarker(index: number, style: MarkerStyle): string {
  const base = `[Image-#${index}]`;
  switch (style) {
    case "code":
      return `\`${base}\``;
    case "bold":
      return `**${base}**`;
    case "plain":
    default:
      return base;
  }
}

/**
 * Render `[Image-#N]` markers into text, replacing resolved path tokens.
 *
 * `resolved` maps each successfully-loaded token to its 0-based index in the
 * final images array (pre-existing `event.images` + newly-attached). Markers
 * are 1-indexed (index 0 → `[Image-#1]`). Unresolvable tokens (not in
 * `resolved`) are left as-is — the model or a later tool call surfaces the
 * error if the path matters.
 *
 * Replacement is done in a single right-to-left pass (so earlier indices
 * don't shift when later tokens are replaced). For tokens that are
 * substrings of other tokens, the longest match at each position wins.
 */
export function renderMarkers(
  text: string,
  tokens: string[],
  resolved: Map<string, { index: number }>,
  style: MarkerStyle,
): string {
  // Collect all (start, end, marker) replacements.
  type Replacement = { start: number; end: number; marker: string };
  const replacements: Replacement[] = [];

  for (const token of tokens) {
    const info = resolved.get(token);
    if (!info) continue; // unresolvable → leave as-is
    const marker = styleMarker(info.index + 1, style);

    // Find all occurrences of this token in the text.
    let searchFrom = 0;
    while (searchFrom <= text.length) {
      const pos = text.indexOf(token, searchFrom);
      if (pos === -1) break;
      replacements.push({ start: pos, end: pos + token.length, marker });
      searchFrom = pos + token.length;
    }
  }

  if (replacements.length === 0) return text;

  // Sort by start position descending (right-to-left replacement keeps
  // indices stable). For overlapping replacements (token A is a substring
  // of token B at the same position), keep the longer one by sorting end
  // descending as a secondary key, then filtering overlaps.
  replacements.sort((a, b) => b.start - a.start || b.end - a.end);

  // Filter overlapping replacements: if a replacement is entirely within
  // a previously-accepted (to-the-right) replacement, skip it. Since we go
  // right-to-left, "previously accepted" means start >= current end.
  const accepted: Replacement[] = [];
  let lastStart = Infinity;
  for (const r of replacements) {
    if (r.end <= lastStart) {
      accepted.push(r);
      lastStart = r.start;
    }
    // else: overlaps with an accepted (longer) replacement to the right → skip
  }

  // Apply replacements right-to-left.
  let result = text;
  for (const r of accepted) {
    result = result.slice(0, r.start) + r.marker + result.slice(r.end);
  }

  return result;
}

/**
 * Build the hint line appended in text-only + "hint" mode. Plain text (no
 * markdown, no ANSI) so the model reads it cleanly.
 *
 * v0.4.0 (SPEC-4 §3.4): now lists the image **paths** so the model can
 * actually call `describe_image` (previously the hint named the tool but
 * erased the paths via markers, leaving the model unable to invoke it).
 * For N≥2 images, names the `image_paths` batch affordance so the model
 * learns the batch tool exists. Paths are listed on indented lines so they
 * are trivially extractable (regex `^  (.+)$`).
 */
export function buildHintLine(
  images: Array<{ token: string; index: number }>,
): string {
  const n = images.length;
  if (n === 0) {
    return "0 images referenced.";
  }
  const noun = n === 1 ? "image" : "images";
  const verb = n === 1 ? "analyze it" : "analyze them";
  const clause = n >= 2 ? " (single, or pass all paths to image_paths for batch analysis)" : "";
  const pathLines = images.map((img) => `  ${img.token}`).join("\n");
  return `${n} ${noun} referenced. The active model cannot process images natively — use the describe_image tool to ${verb}${clause}.
Image paths:
${pathLines}`;
}

/** A per-image result for the batch tool-result builder. */
export type BatchImageResult =
  | { ok: true; text: string; cached: boolean; fallback: boolean; fallbackModel?: string }
  | { ok: false; errorCode: string; message: string };

/**
 * Build the structured per-image tool-result text for a batch
 * `describe_image` call (SPEC-4 §3.1.1). `results` must be in input order
 * (matching `paths`); each entry is either a success (description) or a
 * failure (sentinel — a failed image becomes an `[error: …]` section, never
 * a whole-batch reject). The caller sets `isError` on the tool result only
 * if **every** image failed.
 *
 * Pure: string in → string out.
 */
export function buildBatchToolResult(
  paths: string[],
  results: BatchImageResult[],
): string {
  if (paths.length === 0) {
    return "[Batch: 0 image(s)]";
  }
  const lines: string[] = [`[Batch: ${paths.length} image(s)]`, ""];
  for (let i = 0; i < paths.length; i++) {
    const path = paths[i]!;
    const r = results[i];
    const header = `[Image ${i + 1}]`;
    if (r && r.ok) {
      const tags: string[] = [];
      if (r.cached) tags.push("cached");
      if (r.fallback) tags.push(`fallback: ${r.fallbackModel ?? "unknown"}`);
      const tagStr = tags.length > 0 ? ` (${tags.join(", ")})` : "";
      lines.push(`${header}${tagStr} ${path}`);
      lines.push(r.text);
      lines.push("");
    } else if (r && !r.ok) {
      lines.push(`${header} ${path}`);
      lines.push(`[error: ${r.errorCode} — ${r.message}]`);
      lines.push("");
    } else {
      // Defensive: result missing for this path (shouldn't happen — caller
      // passes results aligned to paths).
      lines.push(`${header} ${path}`);
      lines.push("[error: unexpected — no result for this image]");
      lines.push("");
    }
  }
  return lines.join("\n").trimEnd();
}

/**
 * Build the descriptions block appended in text-only + "auto" mode.
 * Each image's delegation result is appended as a labeled line. A footer
 * notes the vision model + how to switch to hint mode (cost awareness).
 */
export function buildDescriptionsBlock(
  descriptions: Array<{ token: string; index: number; text: string; cached: boolean }>,
  visionModel: string,
): string {
  if (descriptions.length === 0) return "";

  const lines = descriptions.map((d) => {
    const label = styleMarker(d.index + 1, "code");
    const cachedTag = d.cached ? " (cached)" : "";
    return `[${label} ${d.token}]: ${d.text}${cachedTag}`;
  });

  const footer = `[${descriptions.length} image(s) auto-described via ${visionModel}. Set textOnlyPasteMode to "hint" to delegate on-demand instead.]`;

  return `\n\n${lines.join("\n")}\n${footer}`;
}