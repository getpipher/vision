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
primaries). Set it once via `/vision`:

```
/vision provider ollama
/vision model minimax-m3:cloud
```

The provider and model must be defined in `~/.pi/agent/models.json` and the
model should have `"input": ["text", "image"]`. Other `/vision` subcommands:

| Subcommand | Purpose |
|---|---|
| `/vision` or `/vision show` | Display the current config |
| `/vision on` / `/vision off` | Enable / disable the tool |
| `/vision provider <name>` | Set the vision provider |
| `/vision model <id>` | Set the vision model id |
| `/vision max-dim <px>` | Max image dimension for compression (1–8000) |
| `/vision quality <1-100>` | JPEG re-encode quality |
| `/vision reasoning-effort <off\|minimal\|low\|medium\|high\|xhigh>` | Default reasoning effort for delegation |
| `/vision clear` | Reset config to defaults |

Config is stored at `~/.pi/agent/vision.json` (not `vision-tool.json`, so it
doesn't collide with the community package during transition).

## How it works

Two mechanisms combine to guarantee the behavior:

1. **Conditional tool availability (mechanism A).** `describe_image` is always
   *registered*, but its visibility to the LLM is toggled with
   `pi.setActiveTools()` based on the active model's `input` capability. On a
   multimodal primary, the tool is removed from the active set — the LLM never
   sees it, so it can never waste a call. On a text-only primary, the tool is
   added back so the LLM can delegate. The `model_select` event re-syncs on
   mid-session `/model` switches.

2. **PASS-THROUGH delivery guarantee (mechanism B-lite).** When the primary
   model is multimodal, a minimal `input` hook detects image file paths
   referenced in your message and attaches them as native image content — so
   the image reaches the model regardless of whether the LLM would otherwise
   choose the built-in `read` tool. This makes "0 delegation for multimodal
   primary" pass by construction, not by hope. The hook dedups by data hash so
   it coexists cleanly with other paste extensions during transition.

## Using `describe_image`

The tool accepts a file path, data URL, or raw base64:

```
describe_image(image_path: "/tmp/screenshot.png", prompt: "What's in this image?")
```

Parameters:

| Param | Type | Description |
|---|---|---|
| `image_path` | string | File path, `data:` URL, or raw base64 |
| `prompt` | string | What to analyze or answer about the image |
| `compress` | boolean? | Optimize the image before delegation (default `true`) |
| `reasoning` | enum? | Reasoning effort for the delegation (`off`…`xhigh`) |

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