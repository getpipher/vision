// src/index.ts — public stable surface for @getpipher/vision.
// Re-exports the capability/delegate/config pure functions so non-extension
// consumers (e.g. @getpipher/armory-fleet) can delegate without an ExtensionContext.
export { isMultimodal, TOOL_NAME } from "../lib/capability.ts";
export { loadConfig, configFilePath, type VisionConfig } from "../lib/config.ts";
export {
  delegateToVisionModel,
  createVisionDelegator,
  type DelegateParams,
  type DelegateResult,
  type DelegateSuccess,
  type DelegateFailure,
  type VisionDelegator,
  type VisionDelegatorDeps,
  type ModelRegistryLike,
} from "../lib/delegate.ts";