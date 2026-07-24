// src/index.d.ts — typed declaration mirroring src/index.ts (dual-condition: types→.d.ts, default→.ts).
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