# @getpipher/vision

> Capability-aware vision + paste extension for the [pi coding agent](https://github.com/earendil-works/pi-coding-agent).

`@getpipher/vision` adds a `describe_image` tool to pi that is **aware of the
active primary model's input modalities** — so image analysis is never wasteful.

## Why

Two community pi extensions — [`pi-vision-tool`](https://github.com/xezpeleta/pi-vision-tool)
and [`pi-paster`](https://github.com/beowulf11/pi-paster) — are load-bearing but
leave real workflow friction. The biggest: `pi-vision-tool` **always delegates**
image analysis to a second vision model, even when the active primary model is
multimodal and can see images natively. That's a wasteful double-call: extra
API roundtrip, extra tokens, and a hardcoded description that's worse than the
model's own native image understanding.

`@getpipher/vision` fixes this with **capability-aware delegation**:

- **Multimodal primary model** (e.g. `minimax-m3:cloud`, `qwen3.5:cloud`) → the
  image passes through to the model natively. **Zero delegation calls, zero
  extra tokens, zero extra latency.** The `describe_image` tool is hidden from
  the model entirely, so the wasted call is structurally impossible — not just
  discouraged.
- **Text-only primary model** → `describe_image` delegates to a configured
  vision model and returns the text description (the existing, working pattern).

The capability check is automatic and silent — no config, no opt-in, no flag.
It just works under the hood.

## Install

```
pi install npm:@getpipher/vision
```

`@getpipher/vision` **replaces** `pi-vision-tool` and `pi-paster`. Uninstall
those first to avoid duplicate tools and double-attachment:

```
pi uninstall npm:pi-vision-tool
pi uninstall npm:pi-paster
```

## Configure

The vision tool needs a configured vision model for DELEGATE mode (text-only
primaries). Run `/vision` to open an interactive settings panel (pi's native
`SettingsList` — the same engine `/settings` uses): arrow keys navigate,
Enter cycles a value or opens the vision-model picker, Escape exits. Changes
apply live.

```
/vision
```

Or use the typed subcommands (power users / scripts):

```
/vision provider ollama
/vision model minimax-m3:cloud
```

The provider and model must be defined in `~/.pi/agent/models.json` and the
model should have `"input": ["text", "image"]`. Other `/vision` subcommands:

| Subcommand | Purpose |
|---|---|
| `/vision` (no arg) | Open the interactive settings panel (like `/settings`) |
| `/vision show` | Display the current config as text |
| `/vision model` (no arg) | Open a picker of vision-capable authed models |
| `/vision on` / `/vision off` | Enable / disable the tool |
| `/vision provider <name>` | Set the vision provider |
| `/vision model` (no arg) | Open a picker of vision-capable authed models |
| `/vision model <id>` | Set the vision model id (typed fallback) |
| `/vision max-dim <px>` | Max image dimension for compression (1–8000) |
| `/vision quality <1-100>` | JPEG re-encode quality |
| `/vision reasoning-effort <off\|minimal\|low\|medium\|high\|xhigh>` | Default reasoning effort for delegation |
| `/vision system-prompt [<text>\|clear]` | Set/clear a custom system prompt for the vision model (no arg → multi-line editor) |
| `/vision cache <clear\|show>` | Clear the cache or show stats (memory + disk entries) |
| `/vision fallback <provider/model>\|clear` | Set/clear a fallback vision model |
| `/vision clear` | Reset config to defaults |
| `/vision-use [provider/model]` | Switch the DELEGATE vision model inline (no arg → picker). **Hotkey: `ctrl+shift+i`** (rebindable via `keybindings.json`; on Mac, `alt`-based combos need `macos-option-as-alt=true`) |
| `/vision paste-mode [hint\|auto\|off]` | Set how pasted images are handled on a text-only primary (no arg → cycle). |
| `/vision marker-style [code\|bold\|plain]` | Set the markdown style for `[Image-#N]` markers (no arg → show current). |
| `/vision auto-prompt [<text>\|clear]` | Set/clear the generic auto-delegation prompt (no arg → multi-line editor). |
| `/vision preview <path>` | Open a full-screen TUI preview of an image (Kitty/iTerm2 graphics, text fallback on tmux). |
| `/vision batch-concurrency [<1-20>]` | Max parallel image delegations in a batch (`describe_image image_paths` + paste auto mode). 1 = serial; 20 = aggressive. Default 5. |

Config is stored at `~/.pi/agent/vision.json` (not `vision-tool.json`, so it
doesn't collide with the community package during transition).

## Resilience (v0.2.0)

DELEGATE mode (text-only primary) is resilient + cheap:

- **Caching.** Successful delegation results are cached by a content-addressed
  key (image hash + compression params + prompt + vision model + reasoning).
  A second call on the same image costs **zero** vision-model API calls.
  In-memory by default; opt into cross-session persistence with **Persist
  cache to disk** (LRU-evicted at the configured max entries). Only successes
  are cached — failures never are. `/vision cache clear` wipes both layers.
- **Retry + fallback.** On a retryable failure (HTTP 5xx, 429, network), the
  primary vision model is retried with exponential backoff (abort-aware — a
  cancelled turn stops retrying immediately). On a non-retryable error or
  exhausted retries, a configured **fallback vision model** is tried once.
  Configure both via the `/vision` panel or `/vision fallback <provider/model>`.
- **Custom system prompt.** A per-workflow framing prepended to the
  vision-model request (`/vision system-prompt <text>`, or the panel row).
- **Inline model switch.** `ctrl+shift+i` (or `/vision-use`) switches the
  DELEGATE vision model mid-session without opening the full panel.

## Paste UX (v0.3.0)

When you reference an image file path in your message, the paste hook makes it
visible and actionable — capability-aware for both multimodal and text-only
primaries.

- **[Image-#N] markers.** Image file paths in your message are replaced with
  `[Image-#N]` markers (sequential, 1-indexed). The marker style is
  configurable: `code` (inline code — default, visually distinct), `bold`, or
  `plain`. On a multimodal primary, the image is also attached as a native
  attachment (zero delegation). On a text-only primary, the marker is
  informational — the image is NOT attached (text-only models can't process
  images).
- **Text-only paste modes.** When the primary model is text-only, pasted
  images can't be processed natively. Three configurable modes:
  - **hint** (default): markers + a hint line nudging the model to call
    `describe_image`. Zero tokens — the model decides whether to delegate.
  - **auto** (opt-in): auto-delegates each image via the v0.2.x pipeline
    (cache/retry/fallback) and appends the descriptions to your message.
    Timeout-protected (own AbortController, default 30s) — falls back to hint
    on timeout or failure.
  - **off**: markers only — no attachment, no hint, no delegation.

## Image Preview (v0.3.3)

`@getpipher/vision` can render images directly in the terminal — no external
viewer needed.

**Compose-time auto-preview.** As you type a message referencing an image
path, the image renders **above the editor** (WhatsApp/Telegram compose-box
style) ~300ms after you stop typing. Clears when you remove the path or
submit the message. Enable/disable via the `/vision` panel
(**Compose preview** row) or the config field `composePreview`.

**On-demand preview.** `/vision preview <path>` opens a full panel showing
the image + metadata (filename, dimensions, MIME, file size, detected
protocol). Useful for checking an image before deciding to analyze it.

**Terminal support:**

| Terminal | Image rendering |
|---|---|
| Kitty, Ghostty, WezTerm, Warp (standalone) | ✅ Real graphics (Kitty protocol) |
| iTerm2 (standalone) | ✅ Real graphics (iTerm2 protocol) |
| tmux (any terminal) | Text fallback (`[Image: filename image/png WxH]`) |
| VSCode, Alacritty, other | Text fallback |

The text fallback still shows useful metadata (filename, dimensions, format,
file size) and confirms the image was found. Real graphics require running pi
outside tmux on a graphics-capable terminal.

## Batch + scale (v0.4.0)

`describe_image` accepts **multiple images** in a single call via
`image_paths` (alongside the single `image_path` for back-compat). For
text-only primaries (where the tool is visible), this lets the model analyze,
compare, or cross-reference several images in one tool call instead of N
serial round-trips:

```
describe_image(
  image_paths: ["/tmp/before.png", "/tmp/after.png", "/tmp/diff.png"],
  prompt: "Compare these three screenshots. What changed?"
)
```

Delegations run **in parallel**, bounded by `batchConcurrency` (default 5,
configurable 1–20 via `/vision batch-concurrency`). `1` = serial escape
hatch; `20` = aggressive (rate-limit risk is yours). Each image reuses the
v0.2.x resilience pipeline (cache/retry/fallback) independently — a cache hit
returns 0 vision-model calls, a failed image becomes an `[error: …]` section
rather than failing the whole batch, and `isError` is set only if **every**
image failed. The result is one structured, order-stable text block:

```
[Batch: 3 image(s)]

[Image 1] /tmp/before.png
<vision model description>

[Image 2] (cached) /tmp/after.png
<vision model description>

[Image 3] /tmp/diff.png
[error: not_found — image not found at /tmp/diff.png]
```

A hard cap of **50 images** per call (`MAX_BATCH_IMAGES`) defends against an
over-eager model passing an absurd array; split across calls if you need more.

**Parallel auto-delegation (paste auto mode).** When `textOnlyPasteMode` is
`"auto"` and you paste multiple image paths, delegations now run in parallel
(one batch-level timeout = the total budget, bounded by `batchConcurrency`)
instead of serially — so a 5-screenshot paste completes in ~`ceil(N/c)` ×
per-call instead of `N` × per-call.

**Hint mode now exposes paths.** In text-only + `"hint"` mode (the default),
the hint line now **lists the image paths** and names the `image_paths` batch
affordance, so the model can actually invoke `describe_image` (previously the
hint named the tool but the path markers erased the paths, leaving the model
unable to call it). Paste 2+ images and the model learns it can pass them all
to `image_paths` for batch analysis.

**Clipboard paste just works.** Pi binds `ctrl+v` (`alt+v` on Windows) to
paste the system clipboard image: it reads the clipboard, writes the bytes to
`/tmp/pi-clipboard-<uuid>.<ext>`, and inserts that path at the cursor. Our
existing path-token pipeline detects it, renders a `[Image-#N]` marker, and
attaches (multimodal) or delegates (text-only) — no separate clipboard code
path needed. Multi-image clipboard = N `ctrl+v` presses = N paths = handled
as a batch.

## How it works

Two mechanisms combine to guarantee the behavior:

1. **Conditional tool availability (mechanism A).** `describe_image` is always
   *registered*, but its visibility to the LLM is toggled with
   `pi.setActiveTools()` based on the active model's `input` capability. On a
   multimodal primary, the tool is removed from the active set — the LLM never
   sees it, so it can never waste a call. On a text-only primary, the tool is
   added back so the LLM can delegate. The `model_select` event re-syncs on
   mid-session `/model` switches.

2. **Capability-aware paste hook.** When the primary model is multimodal,
   the `input` hook detects image file paths in your message, attaches them as
   native image content, and renders `[Image-#N]` markers (mechanism B-lite,
   graduated in v0.3.0). On a text-only primary, the hook renders markers but
   does NOT attach images — instead it hints, auto-delegates, or stays silent
   based on `textOnlyPasteMode`. The hook dedups by data hash so it coexists
   cleanly with other paste extensions during transition.

## Using `describe_image`

The tool accepts a file path, data URL, or raw base64. For a **single image**:

```
describe_image(image_path: "/tmp/screenshot.png", prompt: "What's in this image?")
```

For **multiple images** (batch — parallel delegation, one structured result):

```
describe_image(
  image_paths: ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"],
  prompt: "Compare these screenshots. What changed between them?"
)
```

Parameters:

| Param | Type | Description |
|---|---|---|
| `image_path` | string? | Path to a single image, `data:` URL, or raw base64. Use for one image. |
| `image_paths` | string[]? | Multiple paths to analyze together (comparison/cross-reference). Up to 50. |
| `prompt` | string | What to analyze or answer about the image(s). For a batch, applies to each; describe what to compare. |
| `compress` | boolean? | Optimize the image(s) before delegation (default `true`) |
| `reasoning` | enum? | Reasoning effort for the delegation (`off`…`xhigh`) |

When caching or fallback is active, the tool result `details` include
`cached: true` (cache hit) and `fallback: true` (result from the fallback
model) for traceability. For a batch, `details.batch` is an array of per-image
results (index, path, ok, cached, fallback, errorCode) in input order.

For multimodal primaries you don't call `describe_image` — just reference the
image path in your message and the model sees it natively.

## Credits

Inspired by [`pi-vision-tool`](https://github.com/xezpeleta/pi-vision-tool)
(xezpeleta) and [`pi-paster`](https://github.com/beowulf11/pi-paster)
(beowulf11) — both filled this gap first and motivated this clean-room
reimplementation. Superseded by `@getpipher/vision`.

This package contains no code copied from either project. The behavior is
specified independently and implemented fresh.

## License

MIT