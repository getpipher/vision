<!-- Satellite context file — extends the global hub (~/.claude/CLAUDE.md | ~/.pi/agent/AGENTS.md). Host-neutral; project-specific only. Do not duplicate hub standards here. -->

# @getpipher/vision

> Capability-aware vision + paste extension for the [pi coding agent](https://github.com/earendil-works/pi-coding-agent). npm: `@getpipher/vision`. Replaces `pi-vision-tool` + `pi-paster`.

**Org context:** getpipher is the Pi coding-agent ecosystem. No GitLab mirror for getpipher.

## What

Adds a `describe_image` tool to pi that is **aware of the active primary model's input modalities** — so image analysis is never wasteful.

- **Multimodal primary model** (e.g. `minimax-m3:cloud`, `qwen3.5:cloud`) → the image passes through to the model natively. **Zero delegation calls, zero extra tokens, zero extra latency.** The `describe_image` tool is hidden from the model entirely, so the wasted call is structurally impossible — not just discouraged.
- **Text-only primary model** → `describe_image` delegates to a configured vision model and returns the text description (the existing, working pattern).

The capability check is automatic and silent — no config, no opt-in, no flag.

## Install

```
pi install npm:@getpipher/vision
```

`@getpipher/vision` **replaces** `pi-vision-tool` and `pi-paster`. Uninstall those first to avoid duplicate tools and double-attachment.

## Structure

```
extensions/   # pi extension registering the describe_image tool + paste/paste-detection
lib/          # capability detection, delegation, marker rendering, globalThis shared state
tests/        # tsx --test suite (203 tests)
```

## Common Commands

```bash
pnpm typecheck   # tsc --noEmit
pnpm test:run     # tsx --test
```

## Notes

- jiti dual-module-instance pitfall: shared state must live on `globalThis`, not module scope (memory `pi-jiti-dual-module-instance.md`).
- Roadmap: v0.4.0 (batch + scale — 10+ images, parallel delegation) and v0.5.0 remain after v0.3.x shipped.