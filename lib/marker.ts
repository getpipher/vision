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
 */
export function buildHintLine(imageCount: number): string {
  const noun = imageCount === 1 ? "image" : "images";
  return `[${imageCount} ${noun} referenced. The active model cannot process images natively — use the describe_image tool to analyze them.]`;
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