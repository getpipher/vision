/**
 * Audit log read/write helpers (SPEC-5 §3.1, PLAN-5 §1.2/§1.3/step 1).
 *
 * The audit log is a persisted, append-only JSONL record of where each image
 * went during vision-model delegation. One line per delegation event (single
 * or per-image in a batch — each per-image `delegateToVisionModel` call
 * writes its own entry). The log answers "where did my image bytes go?"
 * (provider / model / cached / fallback / ok / error / latency) **without
 * storing the image bytes or the full prompt**.
 *
 * Location: `~/.pi/agent/vision-audit.log` by default (resolved from
 * `config.auditLogPath` or the default `<agentDir>/vision-audit.log`). The
 * path resolution lives in `resolveAuditPath` (pure); the actual
 * `getAgentDir()` call is made by the caller (`lib/delegate.ts`) so this
 * module stays pure + unit-testable without env manipulation.
 *
 * All helpers are best-effort: a log failure (disk full, permissions) is
 * swallowed + warned to stderr — the delegation result is the primary
 * outcome; the audit log is secondary (SPEC-5 §9.11).
 *
 * **Concurrency (PLAN-5 §1.2):** `appendAuditEntry` uses `appendFileSync`
 * (O_APPEND). Node.js is single-threaded for JS execution, so synchronous
 * calls don't interleave; POSIX guarantees writes ≤ 4096 bytes to an
 * O_APPEND file are atomic. A single audit entry is ~300–500 bytes — safe
 * under parallel batch delegations (no locking needed, T68).
 *
 * **Privacy stance (SPEC-5 §3.1):** never logs image bytes or the full
 * prompt. `source_hash` is a one-way content fingerprint; `image_path` is
 * truncated for data:URL/base64 (see `truncateImagePathForLog`).
 *
 * Pure: no pi runtime, no shared state (stateless — no `globalThis` needed,
 * unlike `lib/state.ts`).
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, truncateSync } from "node:fs";
import { dirname, join } from "node:path";

/** One delegation event, logged as a single JSONL line. */
export interface AuditEntry {
  /** ISO 8601 timestamp of the delegation event. */
  ts: string;
  /** The provider id the image was sent to (or attempted). On a fallback
   *  success this is the *configured primary* provider (the attempted route),
   *  not the fallback provider — `fallback` + `fallback_model` disambiguate. */
  provider: string;
  /** The model id that actually responded (`result.details.model`, the
   *  `"provider/model"` string). On a fallback success, the fallback model. */
  model: string;
  /** The image_path the user passed (file path / data:URL / base64 —
   *  truncated for data:URL + base64 via `truncateImagePathForLog`; file
   *  paths logged in full since the user already has them). */
  image_path: string;
  /** SHA-256 hex of the original image bytes (the content-addressed
   *  fingerprint — lets the user correlate repeat queries on the same
   *  image without us storing the bytes). Always logged. */
  source_hash: string;
  /** true if the result came from the cache (0 network calls). */
  cached: boolean;
  /** true if the result came from the fallback vision model. */
  fallback: boolean;
  /** The fallback model id, if fallback was used (else undefined). */
  fallback_model: string | undefined;
  /** true if the delegation succeeded (description returned), false on any error. */
  ok: boolean;
  /** Error code on failure (e.g. "local_only", "vision_call_error", "aborted",
   *  "model_not_found"). undefined on success. */
  error_code: string | undefined;
  /** Round-trip latency in ms (the vision-model call time, or 0 for a cache
   *  hit / local-only refusal). For a fallback, the fallback call's latency. */
  latency_ms: number;
  /** true if local-only mode was active (the entry was a cache hit or a
   *  refused cache miss). Makes local-only behavior greppable. */
  local_only: boolean;
}

/** The default audit log filename inside the agent dir. */
export const AUDIT_LOG_FILENAME = "vision-audit.log";

/**
 * Resolve the audit log path: explicit config path (if non-empty), or the
 * default `<agentDir>/vision-audit.log`. Pure — takes `agentDir` as a param
 * (the caller does the `getAgentDir()` call) so this is unit-testable without
 * env manipulation.
 */
export function resolveAuditPath(configPath: string | undefined, agentDir: string): string {
  const trimmed = configPath?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : join(agentDir, AUDIT_LOG_FILENAME);
}

/**
 * Truncate an image_path for logging: file paths full; data:URL + long base64
 * truncated to first 64 chars + a `…(N bytes|chars)` suffix. Pure.
 *
 * Conservative guard: anything with a `/` or `(` is treated as a path (full).
 * A long base64 that happens to contain `/` (the 63rd base64 char) is a
 * deliberate false-negative on truncation — better to over-log a long
 * base64 than to truncate a real path.
 */
export function truncateImagePathForLog(imagePath: string): string {
  if (imagePath.startsWith("data:")) {
    const bytes = Buffer.byteLength(imagePath, "utf8");
    return `${imagePath.slice(0, 64)}…(${bytes} bytes)`;
  }
  // Long base64 (no path separators, > 200 chars) → truncate.
  if (imagePath.length > 200 && !/[/(]/.test(imagePath)) {
    return `${imagePath.slice(0, 64)}…(${imagePath.length} chars)`;
  }
  return imagePath;
}

/**
 * Append an audit entry as one JSONL line. Best-effort: never throws — a
 * log failure (disk full, permissions, unwritable parent) is swallowed + a
 * warning is written to stderr. Creates the parent dir (recursive) if
 * missing. Synchronous (`appendFileSync`) — safe under concurrent batch
 * delegations (Node single-threaded + O_APPEND atomic ≤ 4096B; PLAN-5 §1.2).
 */
export function appendAuditEntry(path: string, entry: AuditEntry): void {
  try {
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf8");
  } catch (err) {
    // Best-effort: a log failure must never break the delegation.
    // eslint-disable-next-line no-console
    console.warn(`[vision] audit log write failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/** Truncate the audit log to 0 entries. Best-effort (no throw on missing file). */
export function clearAuditLog(path: string): void {
  try {
    if (existsSync(path)) truncateSync(path, 0);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(`[vision] audit log clear failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * Read the last `n` entries (tail), newest-last. Skips unparseable lines
 * (defensive against log corruption — a corrupt line never throws). Returns
 * `[]` if the file is missing.
 */
export function tailAuditLog(path: string, n: number): AuditEntry[] {
  if (!existsSync(path)) return [];
  try {
    const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim().length > 0);
    const tail = lines.slice(-n);
    const out: AuditEntry[] = [];
    for (const line of tail) {
      try {
        out.push(JSON.parse(line) as AuditEntry);
      } catch {
        // skip corrupt line — defensive
      }
    }
    return out;
  } catch {
    return [];
  }
}

/** Count entries (non-empty lines) in the audit log. Returns 0 if missing. */
export function countAuditLog(path: string): number {
  if (!existsSync(path)) return 0;
  try {
    return readFileSync(path, "utf8").split("\n").filter((l) => l.trim().length > 0).length;
  } catch {
    return 0;
  }
}