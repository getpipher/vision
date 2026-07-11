import { test } from "node:test";
import assert from "node:assert/strict";
import { mapWithConcurrency } from "../lib/batch.ts";

/** Small delay helper for concurrency-timing assertions. */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * A concurrency-tracking fn wrapper. Records the high-water mark of
 * in-flight calls so tests can assert the bound is respected.
 */
function makeTracker<T, R>(
  fn: (item: T, index: number) => Promise<R>,
): {
  fn: (item: T, index: number) => Promise<R>;
  inFlight: () => number;
  maxObserved: () => number;
} {
  let inFlight = 0;
  let maxObserved = 0;
  return {
    fn: async (item: T, index: number) => {
      inFlight++;
      maxObserved = Math.max(maxObserved, inFlight);
      try {
        return await fn(item, index);
      } finally {
        inFlight--;
      }
    },
    inFlight: () => inFlight,
    maxObserved: () => maxObserved,
  };
}

test("mapWithConcurrency: empty input → empty output", async () => {
  const result = await mapWithConcurrency([], 5, async (x) => x);
  assert.deepEqual(result, []);
});

test("mapWithConcurrency: single item → single result", async () => {
  const result = await mapWithConcurrency([42], 5, async (x) => x * 2);
  assert.deepEqual(result, [84]);
});

test("mapWithConcurrency: concurrency=1 → fully serial (maxObserved == 1)", async () => {
  const tracker = makeTracker(async (x: number) => {
    await delay(15);
    return x * 2;
  });
  const result = await mapWithConcurrency([1, 2, 3, 4], 1, tracker.fn);
  assert.deepEqual(result, [2, 4, 6, 8]);
  assert.equal(tracker.maxObserved(), 1, "serial → never more than 1 in flight");
});

test("mapWithConcurrency: concurrency=3, 5 items → maxObserved == 3, order preserved", async () => {
  // Vary latencies so completion order differs from input order.
  const tracker = makeTracker(async (x: number, _i: number) => {
    await delay(x % 2 === 0 ? 40 : 10); // evens slow, odds fast
    return x;
  });
  const result = await mapWithConcurrency([10, 11, 12, 13, 14], 3, tracker.fn);
  assert.deepEqual(result, [10, 11, 12, 13, 14]); // input order, not completion
  assert.equal(tracker.maxObserved(), 3, "bound == 3");
});

test("mapWithConcurrency: concurrency >= n → clamped to n (all in flight)", async () => {
  const tracker = makeTracker(async (x: number) => {
    await delay(20);
    return x * 2;
  });
  const result = await mapWithConcurrency([1, 2, 3], 10, tracker.fn);
  assert.deepEqual(result, [2, 4, 6]);
  assert.equal(tracker.maxObserved(), 3, "clamped to n=3");
});

test("mapWithConcurrency: concurrency=0 → treated as 1 (serial)", async () => {
  const tracker = makeTracker(async (x: number) => {
    await delay(10);
    return x;
  });
  const result = await mapWithConcurrency([1, 2, 3], 0, tracker.fn);
  assert.deepEqual(result, [1, 2, 3]);
  assert.equal(tracker.maxObserved(), 1);
});

test("mapWithConcurrency: negative concurrency → treated as 1 (serial)", async () => {
  const tracker = makeTracker(async (x: number) => {
    await delay(10);
    return x;
  });
  const result = await mapWithConcurrency([1, 2], -5, tracker.fn);
  assert.deepEqual(result, [1, 2]);
  assert.equal(tracker.maxObserved(), 1);
});

test("mapWithConcurrency: fractional concurrency → rounded", async () => {
  const tracker = makeTracker(async (x: number) => {
    await delay(15);
    return x;
  });
  // 2.9 rounds to 3 → 4 items with c=3
  const result = await mapWithConcurrency([1, 2, 3, 4], 2.9, tracker.fn);
  assert.deepEqual(result, [1, 2, 3, 4]);
  assert.ok(tracker.maxObserved() <= 3, "rounded concurrency respected");
});

test("mapWithConcurrency: rejecting fn → rejects the whole batch", async () => {
  const tracker = makeTracker(async (x: number) => {
    await delay(10);
    if (x === 2) throw new Error("boom");
    return x;
  });
  await assert.rejects(
    () => mapWithConcurrency([1, 2, 3], 3, tracker.fn),
    /boom/,
  );
});

test("mapWithConcurrency: wrapped fn (catch → sentinel) → resolves with sentinels in order", async () => {
  // The batch-tool pattern: fn never rejects, returns a sentinel on failure.
  const wrapped = async (x: number): Promise<{ ok: true; v: number } | { ok: false; e: string }> => {
    await delay(10);
    if (x === 2) return { ok: false, e: "boom" };
    return { ok: true, v: x };
  };
  const result = await mapWithConcurrency([1, 2, 3], 3, wrapped);
  assert.deepEqual(result, [
    { ok: true, v: 1 },
    { ok: false, e: "boom" },
    { ok: true, v: 3 },
  ]);
});

test("mapWithConcurrency: preserves input order under mixed latencies (concurrency=2)", async () => {
  const tracker = makeTracker(async (x: number, _i: number) => {
    // Late items are fast, early items are slow → completion order reversed.
    await delay(50 - x * 5);
    return x;
  });
  const result = await mapWithConcurrency([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2, tracker.fn);
  assert.deepEqual(result, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.equal(tracker.maxObserved(), 2, "bound == 2 across 10 items");
});

test("mapWithConcurrency: index is passed to fn", async () => {
  const indices: number[] = [];
  await mapWithConcurrency(["a", "b", "c"], 3, async (item, i) => {
    indices.push(i);
    return item;
  });
  indices.sort((a, b) => a - b);
  assert.deepEqual(indices, [0, 1, 2]);
});