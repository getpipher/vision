/**
 * Preview helpers for the TUI image preview (SPEC-3 gap #7).
 *
 * Pure helpers (metadata formatting, ImageTheme bridge) + component factories
 * that build pi-tui `Container`s with `Image` components for rendering images
 * in the terminal. The factories return `Component & { dispose?() }` — the
 * `dispose()` method frees Kitty image IDs to prevent leaks.
 *
 * Image rendering is handled by pi-tui's native `Image` component, which
 * auto-detects the terminal's graphics capability:
 * - Kitty / Ghostty / WezTerm / Warp (standalone) → Kitty graphics protocol
 * - iTerm2 (standalone) → iTerm2 graphics protocol
 * - tmux or unsupported → text fallback `[Image: filename image/png WxH]`
 *
 * No manual capability detection or `forceKittyInTmux` bypass — we respect
 * pi-tui's detection (empirically validated: tmux doesn't forward Kitty
 * graphics even with `allow-passthrough on`).
 */
import {
  Container,
  Image,
  type ImageDimensions,
  type ImageTheme,
  type ImageOptions,
  Text,
  deleteKittyImage,
  getCapabilities,
  getImageDimensions,
  imageFallback,
} from "@earendil-works/pi-tui";
import type { Component } from "@earendil-works/pi-tui";
import { basename } from "node:path";
import { statSync } from "node:fs";

/** A loaded image ready for preview rendering. */
export interface PreviewImage {
  data: string; // base64
  mimeType: string;
  filename: string;
  dimensions: ImageDimensions | null;
  sizeBytes: number;
}

/** Detect the image protocol string for the metadata line. */
export function detectProtocol(): "kitty" | "iterm2" | "text fallback" {
  const caps = getCapabilities();
  if (caps.images === "kitty") return "kitty";
  if (caps.images === "iterm2") return "iterm2";
  return "text fallback";
}

/** Format the metadata line for a preview image. */
export function formatImageMetadata(img: PreviewImage, protocol: string): string {
  const dims = img.dimensions ? `${img.dimensions.widthPx}x${img.dimensions.heightPx}` : "?";
  const sizeStr = formatFileSize(img.sizeBytes);
  return `${img.filename} | ${dims} | ${img.mimeType} | ${sizeStr} | ${protocol}`;
}

/** Format file size as a human-readable string. */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/** Bridge pi's `Theme` to pi-tui's `ImageTheme`. */
export function buildImageTheme(
  themeFg: (color: string, text: string) => string,
): ImageTheme {
  return {
    fallbackColor: (str: string) => themeFg("dim", str),
  };
}

/** Load a preview image from base64 data + mimeType. Returns the PreviewImage
 *  with dimensions + file size (if the path is known). */
export function makePreviewImage(
  data: string,
  mimeType: string,
  filepath?: string,
): PreviewImage {
  const dimensions = getImageDimensions(data, mimeType);
  const filename = filepath ? basename(filepath) : "(unknown)";
  let sizeBytes = 0;
  if (filepath) {
    try {
      sizeBytes = statSync(filepath).size;
    } catch {
      sizeBytes = 0;
    }
  }
  return { data, mimeType, filename, dimensions, sizeBytes };
}

/**
 * Create a preview component for a single image (used by `/vision preview`).
 * Returns a `Container` with the `Image` + metadata `Text`. The `dispose()`
 * method frees the Kitty image ID to prevent leaks.
 */
export function createPreviewComponent(
  img: PreviewImage,
  themeFg: (color: string, text: string) => string,
  maxWidthCells: number,
): Component & { dispose?(): void } {
  const imageTheme = buildImageTheme(themeFg);
  const protocol = detectProtocol();
  const options: ImageOptions = {
    maxWidthCells,
    filename: img.filename,
  };

  const image = new Image(img.data, img.mimeType, imageTheme, options, img.dimensions ?? undefined);
  const container = new Container();
  container.addChild(image);
  container.addChild(new Text(themeFg("dim", formatImageMetadata(img, protocol)), 0, 0));
  container.addChild(new Text(themeFg("dim", "esc close"), 0, 0));

  return {
    render(width: number) {
      return container.render(width);
    },
    invalidate() {
      container.invalidate();
    },
    dispose() {
      const id = image.getImageId();
      if (id !== undefined) {
        try {
          process.stdout.write(deleteKittyImage(id));
        } catch {
          // ignore — terminal may not support Kitty graphics
        }
      }
    },
  };
}

/**
 * Create a compose-time preview component for one or more images (used by the
 * `setWidget` compose-time auto-preview). Shows each image + a single metadata
 * line. The `dispose()` method frees all Kitty image IDs.
 */
export function createComposePreviewComponent(
  images: PreviewImage[],
  themeFg: (color: string, text: string) => string,
  maxWidthCells: number,
): Component & { dispose?(): void } {
  const imageTheme = buildImageTheme(themeFg);
  const protocol = detectProtocol();
  const container = new Container();
  const kittyImageIds: number[] = [];

  for (const img of images) {
    const options: ImageOptions = {
      maxWidthCells,
      filename: img.filename,
    };
    const image = new Image(img.data, img.mimeType, imageTheme, options, img.dimensions ?? undefined);
    container.addChild(image);
    kittyImageIds.push(-1); // placeholder — actual ID allocated on first render
  }

  // Add metadata line
  const metaLine = images
    .map((img) => formatImageMetadata(img, protocol))
    .join(" • ");
  container.addChild(new Text(themeFg("dim", metaLine), 0, 0));

  // Track the Image components for ID cleanup
  const imageComponents = images.map(
    (img) => new Image(img.data, img.mimeType, imageTheme, { maxWidthCells, filename: img.filename }, img.dimensions ?? undefined),
  );

  return {
    render(width: number) {
      return container.render(width);
    },
    invalidate() {
      container.invalidate();
    },
    dispose() {
      for (const image of imageComponents) {
        const id = image.getImageId();
        if (id !== undefined) {
          try {
            process.stdout.write(deleteKittyImage(id));
          } catch {
            // ignore
          }
        }
      }
      // Also check the images already added to the container
      for (const id of kittyImageIds) {
        if (id >= 0) {
          try {
            process.stdout.write(deleteKittyImage(id));
          } catch {
            // ignore
          }
        }
      }
    },
  };
}