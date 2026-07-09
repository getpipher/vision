import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, utimesSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cacheKey, type CacheEntry, VisionCache } from "../lib/cache.ts";

function tmpCacheDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-cache-"));
}

function makeEntry(text: string, model = "ollama/minimax-m3:cloud"): CacheEntry {
  return {
    text,
    details: {
      model,
      image_path: "/tmp/x.png",
      prompt: "describe",
      compressed: true,
      reasoning: "off",
      cached: false,
      fallback: false,
    },
    storedAt: Date.now(),
  };
}

test("cacheKey: deterministic for identical inputs", () => {
  const a = cacheKey("hash1", true, 1568, 85, "describe", "ollama/m", "off");
  const b = cacheKey("hash1", true, 1568, 85, "describe", "ollama/m", "off");
  assert.equal(a, b);
  assert.match(a, /^[0-9a-f]{64}$/);
});

test("cacheKey: differs when any component changes", () => {
  const base = cacheKey("h", true, 1568, 85, "p", "m", "off");
  assert.notEqual(cacheKey("h2", true, 1568, 85, "p", "m", "off"), base);
  assert.notEqual(cacheKey("h", false, 1568, 85, "p", "m", "off"), base);
  assert.notEqual(cacheKey("h", true, 1024, 85, "p", "m", "off"), base);
  assert.notEqual(cacheKey("h", true, 1568, 90, "p", "m", "off"), base);
  assert.notEqual(cacheKey("h", true, 1568, 85, "p2", "m", "off"), base);
  assert.notEqual(cacheKey("h", true, 1568, 85, "p", "m2", "off"), base);
  assert.notEqual(cacheKey("h", true, 1568, 85, "p", "m", "high"), base);
});

test("cacheKey: \\0 separator prevents concatenation ambiguity", () => {
  // ("a","bc") vs ("ab","c") must NOT collide when joined with \0
  const a = cacheKey("a", true, 1, 1, "bc", "m", "off");
  const b = cacheKey("ab", true, 1, 1, "c", "m", "off");
  assert.notEqual(a, b, "field-boundary separator must prevent collisions");
});

test("VisionCache memory-only: set + get hit", () => {
  const cache = new VisionCache(undefined, 256);
  const k = cacheKey("h", true, 1568, 85, "p", "m", "off");
  assert.equal(cache.get(k), undefined);
  cache.set(k, makeEntry("desc"));
  const hit = cache.get(k);
  assert.equal(hit?.text, "desc");
  assert.equal(cache.persisted, false);
});

test("VisionCache memory-only: unknown key → miss", () => {
  const cache = new VisionCache(undefined, 256);
  assert.equal(cache.get("nope"), undefined);
});

test("VisionCache disk: hit restores after memory wipe (promotion)", () => {
  const dir = tmpCacheDir();
  try {
    const cache = new VisionCache(dir, 256);
    const k = cacheKey("h", true, 1568, 85, "p", "m", "off");
    cache.set(k, makeEntry("disk-desc"));
    assert.ok(existsSync(join(dir, `${k}.json`)), "entry persisted to disk");

    // Simulate a session restart: new cache instance (memory empty), same dir.
    const restarted = new VisionCache(dir, 256);
    const hit = restarted.get(k);
    assert.equal(hit?.text, "disk-desc", "disk hit after memory wipe");
    assert.equal(restarted.get(k)?.text, "disk-desc", "promoted to memory (2nd get is memory hit)");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("VisionCache disk: LRU evicts oldest at maxEntries+1", () => {
  const dir = tmpCacheDir();
  try {
    const cache = new VisionCache(dir, 3);
    const keys = ["k1", "k2", "k3", "k4"];
    const now = Date.now() / 1000;
    keys.forEach((k, i) => {
      cache.set(k, makeEntry(`desc-${i}`));
      // Set distinct mtimes so LRU ordering is deterministic.
      utimesSync(join(dir, `${k}.json`), now + i, now + i);
    });
    // 4 entries inserted, max 3 → oldest (k1) evicted.
    assert.equal(existsSync(join(dir, "k1.json")), false, "oldest evicted");
    assert.equal(existsSync(join(dir, "k4.json")), true, "newest retained");
    const stats = cache.stats();
    assert.equal(stats.diskEntries, 3);
    assert.equal(stats.maxEntries, 3);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("VisionCache clear(): wipes memory + disk", () => {
  const dir = tmpCacheDir();
  try {
    const cache = new VisionCache(dir, 256);
    const k = cacheKey("h", true, 1568, 85, "p", "m", "off");
    cache.set(k, makeEntry("desc"));
    assert.ok(existsSync(join(dir, `${k}.json`)));
    cache.clear();
    assert.equal(cache.get(k), undefined, "memory cleared");
    assert.equal(existsSync(join(dir, `${k}.json`)), false, "disk cleared");
    assert.equal(cache.stats().diskEntries, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("VisionCache stats(): counts memory + disk", () => {
  const dir = tmpCacheDir();
  try {
    const cache = new VisionCache(dir, 256);
    cache.set("a", makeEntry("1"));
    cache.set("b", makeEntry("2"));
    const s = cache.stats();
    assert.equal(s.memoryEntries, 2);
    assert.equal(s.diskEntries, 2);
    assert.equal(s.persisted, true);
    assert.equal(s.maxEntries, 256);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("VisionCache disk: corrupt file → miss + removed", () => {
  const dir = tmpCacheDir();
  try {
    const cache = new VisionCache(dir, 256);
    const k = "badkey";
    writeFileSync(join(dir, `${k}.json`), "{ not valid json");
    assert.equal(cache.get(k), undefined, "corrupt file → miss");
    assert.equal(existsSync(join(dir, `${k}.json`)), false, "corrupt file removed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("VisionCache disk: write failure degrades to memory-only (no throw)", () => {
  // A non-existent dir path → write throws internally → swallowed; memory still hits.
  const cache = new VisionCache(join(tmpdir(), "vision-cache-nonexistent-xyz"), 256);
  const k = "memkey";
  assert.doesNotThrow(() => cache.set(k, makeEntry("mem-only")));
  assert.equal(cache.get(k)?.text, "mem-only", "memory hit even when disk write failed");
});