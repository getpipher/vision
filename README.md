# @getpipher/vision

> Capability-aware vision + paste extension for the [pi coding agent](https://github.com/earendil-works/pi-coding-agent).

**Status: v0.1.0 in development.** Full documentation lands with the first release.

## What it does

`@getpipher/vision` adds a `describe_image` tool to pi that is **aware of the
active primary model's input modalities**:

- **Multimodal primary model** (e.g. `minimax-m3:cloud`, `qwen3.5:cloud`) →
  the image passes through to the model natively. **Zero delegation calls,
  zero extra tokens, zero extra latency.** The `describe_image` tool is hidden
  from the model entirely.
- **Text-only primary model** → `describe_image` delegates to a configured
  vision model and returns the text description.

The capability check is automatic and silent — no config, no opt-in.

## Install

```
pi install npm:@getpipher/vision
```

`@getpipher/vision` replaces `pi-vision-tool` and `pi-paster`. Uninstall those
first to avoid duplicate tools:

```
pi uninstall npm:pi-vision-tool
pi uninstall npm:pi-paster
```

## Credits

Inspired by [`pi-vision-tool`](https://github.com/xezpeleta/pi-vision-tool)
(xezpeleta) and [`pi-paster`](https://github.com/beowulf11/pi-paster)
(beowulf11) — both filled this gap first and motivated this clean-room
reimplementation. Superseded by `@getpipher/vision`.

## License

MIT