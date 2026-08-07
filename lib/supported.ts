/**
 * Supported vision-model API types (ADR-0001).
 *
 * `callVisionModel` dispatches its request/response shape on the model's
 * declared `api` type. This module is the single source of truth for which
 * types the extension can delegate to today — pickers, auto-detect, and the
 * delegate path all consult it so users never see a model that can't run.
 *
 * When a new API type is implemented (e.g. `anthropic-messages`), add it
 * here + add a dispatch branch in `lib/delegate.ts`; the picker/auto-detect
 * filters relax in one place.
 */
import type { Api, Model } from "@earendil-works/pi-ai";

/** API types the delegate path can speak (ADR-0001). */
export const SUPPORTED_VISION_APIS: ReadonlySet<string> = new Set([
  "openai-completions",
  "openai-responses",
]);

/** Whether a model's declared API type is delegatable. */
export function isSupportedVisionApi(api: Api | string): boolean {
  return SUPPORTED_VISION_APIS.has(api);
}

/** Whether a model is usable as a vision model: image-capable + supported
 *  API type (the picker/auto-detect filter, ADR-0001). */
export function isUsableVisionModel(model: Model<Api>): boolean {
  return (model.input?.includes("image") ?? false) && isSupportedVisionApi(model.api);
}
