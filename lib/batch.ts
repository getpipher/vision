/**
 * Bounded-concurrency map (SPEC-4 §4.1, PLAN-4 §1.2).
 *
 * Run `fn` over `items` with at most `concurrency` in-flight at a time.
 * Returns results in the **input order** (not completion order).
 *
 * - `concurrency <= 0` → treated as 1 (serial; safe default).
 * - `concurrency >= items.length` → all in flight at once (effectively
 *   `Promise.all`).
 * - If `fn` rejects for one item, that rejection propagates (the whole
 *   batch rejects). Callers that want per-item resilience wrap `fn` to
 *   catch + return a sentinel (the batch tool does exactly this — a failed
 *   image becomes an `[error: …]` section, not a reject).
 *
 * Pure: no I/O, no timers of its own, no pi runtime. The worker-cursor
 * pattern is safe because Node.js is single-threaded for JS execution —
 * `cursor++` is atomic (no `await` between the read and the write), so two
 * workers can never claim the same index.
 */

/**
 * Map `items` through `fn` with at most `concurrency` concurrent calls.
 * Results are returned in input order.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const n = items.length;
  if (n === 0) return [];

  // Clamp + round. Negative or zero → 1 (serial). Above n → n.
  const raw = Number.isFinite(concurrency) ? concurrency : 1;
  const c = Math.max(1, Math.min(Math.round(raw) || 1, n));

  const results: R[] = new Array(n);
  let cursor = 0;

  async function worker(): Promise<void> {
    while (true) {
      const i = cursor++; // claim next index (single-threaded → no race)
      if (i >= n) return;
      results[i] = await fn(items[i]!, i); // each worker writes its own slot
    }
  }

  // Spawn `c` workers and wait for all to drain the cursor.
  const workers: Promise<void>[] = [];
  for (let w = 0; w < c; w++) workers.push(worker());
  await Promise.all(workers);
  return results;
}