import { test } from "node:test";
import assert from "node:assert/strict";
import { AbortError, classifyError, sleep, withRetry, type ErrorClass } from "../lib/resilience.ts";

function httpError(status: number, body = "err"): Error {
  return new Error(`Vision model returned ${status}: ${body}`);
}

test("classifyError: 500/502/503 → retryable", () => {
  assert.equal(classifyError(httpError(500)), "retryable");
  assert.equal(classifyError(httpError(502)), "retryable");
  assert.equal(classifyError(httpError(503)), "retryable");
});

test("classifyError: 429 → retryable (rate limit)", () => {
  assert.equal(classifyError(httpError(429)), "retryable");
});

test("classifyError: 400/401/403/404 → client (no retry)", () => {
  assert.equal(classifyError(httpError(400)), "client");
  assert.equal(classifyError(httpError(401)), "client");
  assert.equal(classifyError(httpError(403)), "client");
  assert.equal(classifyError(httpError(404)), "client");
});

test("classifyError: 'returned no content' → no_content", () => {
  assert.equal(classifyError(new Error("Vision model returned no content in the response")), "no_content");
});

test("classifyError: AbortError → abort (custom + native name)", () => {
  assert.equal(classifyError(new AbortError()), "abort");
  const native = new Error("aborted");
  native.name = "AbortError";
  assert.equal(classifyError(native), "abort");
});

test("classifyError: TypeError (fetch network failure) → retryable", () => {
  assert.equal(classifyError(new TypeError("fetch failed")), "retryable");
});

test("classifyError: network-like messages → retryable", () => {
  assert.equal(classifyError(new Error("ECONNRESET")), "retryable");
  assert.equal(classifyError(new Error("socket hang up")), "retryable");
  assert.equal(classifyError(new Error("ETIMEDOUT")), "retryable");
});

test("classifyError: unknown Error → client (safe default, no retry)", () => {
  assert.equal(classifyError(new Error("something weird")), "client");
  assert.equal(classifyError("a string"), "client");
});

test("withRetry: succeeds on first attempt (no retry)", async () => {
  let calls = 0;
  const r = await withRetry(async () => { calls++; return "ok"; }, { attempts: 3, backoffMs: 10 });
  assert.equal(r, "ok");
  assert.equal(calls, 1);
});

test("withRetry: retries on 500 then succeeds on 3rd attempt (attempts=2)", async () => {
  const delays: number[] = [];
  let calls = 0;
  const r = await withRetry(
    async () => {
      calls++;
      if (calls < 3) throw httpError(500);
      return "ok";
    },
    { attempts: 2, backoffMs: 100, sleepFn: (ms) => { delays.push(ms); return Promise.resolve(); } },
  );
  assert.equal(r, "ok");
  assert.equal(calls, 3, "total attempts = attempts+1");
  assert.equal(delays.length, 2, "slept twice (between 3 attempts)");
  assert.equal(delays[0], 100, "backoff attempt 0 = backoffMs");
  assert.equal(delays[1], 200, "backoff attempt 1 = 2*backoffMs");
});

test("withRetry: no retry on 400 (client) — throws immediately, one call", async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw httpError(400); }, { attempts: 3, backoffMs: 10, sleepFn: () => Promise.resolve() }),
    /returned 400/,
  );
  assert.equal(calls, 1, "client errors are not retried");
});

test("withRetry: no retry on no_content — throws immediately", async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw new Error("Vision model returned no content in the response"); }, { attempts: 3, backoffMs: 10, sleepFn: () => Promise.resolve() }),
    /no content/,
  );
  assert.equal(calls, 1);
});

test("withRetry: retryable exhausts retries → throws last error", async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw httpError(503); }, { attempts: 2, backoffMs: 10, sleepFn: () => Promise.resolve() }),
    /returned 503/,
  );
  assert.equal(calls, 3, "3 total attempts then throw");
});

test("withRetry: abort → AbortError, no further attempts", async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; throw new AbortError(); }, { attempts: 3, backoffMs: 10, sleepFn: () => Promise.resolve() }),
    (err: unknown) => err instanceof AbortError,
  );
  assert.equal(calls, 1, "abort stops immediately");
});

test("withRetry: signal already aborted before first attempt → AbortError, zero calls", async () => {
  const ac = new AbortController();
  ac.abort();
  let calls = 0;
  await assert.rejects(
    withRetry(async () => { calls++; return "ok"; }, { attempts: 3, backoffMs: 10, signal: ac.signal, sleepFn: () => Promise.resolve() }),
    (err: unknown) => err instanceof AbortError,
  );
  assert.equal(calls, 0);
});

test("withRetry: signal aborts during backoff sleep → AbortError, no further attempts", async () => {
  const ac = new AbortController();
  let calls = 0;
  let sleepCalls = 0;
  await assert.rejects(
    withRetry(
      async () => { calls++; throw httpError(500); },
      {
        attempts: 5, backoffMs: 10, signal: ac.signal,
        sleepFn: () => { sleepCalls++; ac.abort(); return Promise.reject(new AbortError()); },
      },
    ),
    (err: unknown) => err instanceof AbortError,
  );
  assert.equal(calls, 1, "only the first attempt ran");
  assert.equal(sleepCalls, 1, "sleep aborted + propagated");
});

test("withRetry: backoff caps at 8000ms", async () => {
  const delays: number[] = [];
  let calls = 0;
  await assert.rejects(
    withRetry(
      async () => { calls++; throw httpError(500); },
      { attempts: 10, backoffMs: 5000, sleepFn: (ms) => { delays.push(ms); return Promise.resolve(); } },
    ),
    /returned 500/,
  );
  // delays: 5000, 10000→capped 8000, 8000, ...
  assert.equal(delays[0], 5000);
  assert.equal(delays[1], 8000, "capped at 8000");
  assert.ok(delays.every((d) => d <= 8000), "no delay exceeds cap");
});

test("sleep: resolves after ms (no signal)", async () => {
  const start = Date.now();
  await sleep(30);
  assert.ok(Date.now() - start >= 25, "slept ~30ms");
});

test("sleep: rejects immediately if signal already aborted", async () => {
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(sleep(100, ac.signal), (err: unknown) => err instanceof AbortError);
});

test("sleep: rejects with AbortError if signal aborts during sleep", async () => {
  const ac = new AbortController();
  const p = sleep(1000, ac.signal);
  ac.abort();
  await assert.rejects(p, (err: unknown) => err instanceof AbortError);
});

test("ErrorClass type is exported (compile-time)", () => {
  const c: ErrorClass = "retryable";
  assert.equal(c, "retryable");
});