/**
 * Retry + fallback resilience for vision-model delegation (SPEC-2 gap #4).
 *
 * `withRetry` wraps a call function with exponential-backoff retry on
 * retryable errors (HTTP 5xx / 429 / network). Non-retryable client errors
 * (4xx) + empty-content responses throw immediately so the caller can fall
 * back to a different model. Aborts (`ctx.signal`) are respected — no retry,
 * no fallback — via an abort-aware `sleep`.
 *
 * Pure + unit-testable: `sleepFn` is injectable for fake-clock tests, and the
 * error classifier is a pure function over the thrown value. No pi runtime
 * dependency.
 */

/** Sentinel for an abort (user cancelled the turn). Stops retry + skips
 *  fallback in the delegate pipeline. */
export class AbortError extends Error {
  constructor(message = "Aborted") {
    super(message);
    this.name = "AbortError";
  }
}

export type ErrorClass = "retryable" | "client" | "abort" | "no_content";

/**
 * Classify a thrown error to decide retry behavior.
 *
 * - `retryable`: HTTP 5xx, 429 (rate limit), network errors (fetch throws
 *   `TypeError`, ECONNRESET, ETIMEDOUT, …) → retry with backoff.
 * - `client`: HTTP 4xx except 429 (bad request, auth, not found) → no retry
 *   (a different model may still succeed → caller falls back).
 * - `no_content`: 2xx but empty response → no retry (same model likely yields
 *   the same empty) → caller falls back.
 * - `abort`: `ctx.signal` aborted → no retry, no fallback (respect the cancel).
 * - unknown → `client` (safe default: don't amplify an unknown failure).
 */
export function classifyError(err: unknown): ErrorClass {
  if (err instanceof AbortError) return "abort";
  if (err instanceof Error && err.name === "AbortError") return "abort"; // native/DOMException
  const msg = err instanceof Error ? err.message : String(err);
  if (msg.includes("returned no content")) return "no_content";
  const m = /returned (\d{3}):/.exec(msg);
  if (m) {
    const status = Number(m[1]);
    if (status === 429 || status >= 500) return "retryable";
    if (status >= 400 && status < 500) return "client";
  }
  if (err instanceof TypeError) return "retryable"; // fetch network failure
  if (/fetch failed|network|ECONNRESET|ETIMEDOUT|ECONNREFUSED|socket hang up|ENOTFOUND/i.test(msg)) {
    return "retryable";
  }
  return "client";
}

/**
 * Abort-aware sleep. Resolves after `ms` unless `signal` aborts first, in
 * which case it rejects with `AbortError`. If `signal` is already aborted,
 * rejects immediately.
 */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new AbortError());
  return new Promise<void>((resolve, reject) => {
    if (!signal) {
      setTimeout(resolve, ms);
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new AbortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export interface RetryOptions {
  /** Number of retries after the first failure (total attempts = attempts + 1). */
  attempts: number;
  /** Base backoff in ms; delay = min(backoffMs * 2^attempt, 8000). */
  backoffMs: number;
  /** Agent abort signal — stops retry mid-backoff. */
  signal?: AbortSignal;
  /** Injectable sleep for fake-clock tests (default: real `sleep`). */
  sleepFn?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

/**
 * Call `callFn` with retry + exponential backoff. `callFn(attempt)` receives
 * the 0-based attempt index.
 *
 * - retryable error + attempts remaining → `await sleepFn(delay, signal)` then retry.
 * - retryable error + last attempt → throw the error (caller falls back).
 * - client / no_content → throw immediately (caller falls back).
 * - abort → throw `AbortError` (caller: no fallback).
 *
 * If `signal` aborts during a backoff sleep, `sleepFn` rejects with
 * `AbortError`, which propagates out (no further attempts).
 */
export async function withRetry<T>(
  callFn: (attempt: number) => Promise<T>,
  opts: RetryOptions,
): Promise<T> {
  const sleepFn = opts.sleepFn ?? sleep;
  const totalAttempts = Math.max(1, Math.round(opts.attempts) + 1);
  let lastErr: unknown;
  for (let attempt = 0; attempt < totalAttempts; attempt++) {
    if (opts.signal?.aborted) throw new AbortError();
    try {
      return await callFn(attempt);
    } catch (err) {
      lastErr = err;
      const cls = classifyError(err);
      if (cls === "abort") {
        throw err instanceof AbortError ? err : new AbortError();
      }
      const isLast = attempt === totalAttempts - 1;
      if (cls === "client" || cls === "no_content" || isLast) {
        throw err; // non-retryable, or out of retries → caller decides fallback
      }
      // retryable + attempts remaining → backoff then retry
      const delay = Math.min(opts.backoffMs * 2 ** attempt, 8000);
      await sleepFn(delay, opts.signal);
    }
  }
  // Unreachable: the loop always returns or throws. Defensive.
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}