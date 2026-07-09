/**
 * Content-addressed cache for vision-model delegation results.
 *
 * A second `describe_image` call on the same image (same prompt, same vision
 * model, same compression params, same reasoning) returns the cached
 * description WITHOUT calling the vision model — zero tokens, zero latency
 * (SPEC-2 gap #2).
 *
 * Cache key = sha256(sourceHash + compress + maxDimension + jpegQuality +
 * prompt + modelId + reasoning). Keying on the ORIGINAL-byte hash (not the
 * compressed bytes) makes hits stable regardless of compression
 * nondeterminism (worker vs in-process fallback — see PLAN-2 §1.1).
 *
 * Two layers:
 * 1. In-memory `Map` (session-scoped, always active when `cacheEnabled`).
 * 2. Optional persisted disk cache (`<dir>/<key>.json`, LRU-evicted by file
 *    mtime) — active when a `dir` is provided (`cachePersist: true`).
 *
 * Only successful results are cached (failures are never cached — a transient
 * error must not poison the cache). Writes are atomic (tmp + rename). No
 * cross-session lockfile (benign races only — see PLAN-2 §1.2).
 */
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { DelegateSuccess } from "./delegate.ts";

/** A cached delegation result. `storedAt` is informational (LRU uses file
 *  mtime, which is robust to clock skew across sessions). */
export interface CacheEntry {
  text: string;
  details: DelegateSuccess["details"];
  storedAt: number;
}

export interface CacheStats {
  memoryEntries: number;
  diskEntries: number;
  maxEntries: number;
  persisted: boolean;
}

/**
 * Compute the content-addressed cache key for a delegation call. Deterministic
 * + collision-safe (sha256 over the full tuple with `\0` separators so no two
 * distinct tuples can collide via concatenation ambiguity).
 */
export function cacheKey(
  sourceHash: string,
  compress: boolean,
  maxDimension: number,
  jpegQuality: number,
  prompt: string,
  modelId: string,
  reasoning: string,
): string {
  const tuple = [sourceHash, compress, maxDimension, jpegQuality, prompt, modelId, reasoning].join("\0");
  return createHash("sha256").update(tuple).digest("hex");
}

/**
 * Vision description cache. Memory-first, disk-optional. Pure I/O — no pi
 * runtime dependency — so it unit-tests with a tmp dir or memory-only.
 */
export class VisionCache {
  private readonly memory = new Map<string, CacheEntry>();
  private readonly dir?: string;
  private readonly maxEntries: number;

  constructor(dir?: string, maxEntries = 256) {
    this.dir = dir;
    this.maxEntries = Math.max(1, Math.round(maxEntries));
  }

  get persisted(): boolean {
    return this.dir !== undefined;
  }

  /** Look up a cached entry. Memory first; on miss, disk (promoting a hit
   *  into memory). A corrupt disk file is treated as a miss + removed. */
  get(key: string): CacheEntry | undefined {
    const mem = this.memory.get(key);
    if (mem) return mem;
    if (this.dir) {
      const file = this.fileFor(key);
      if (existsSync(file)) {
        try {
          const entry = JSON.parse(readFileSync(file, "utf8")) as CacheEntry;
          if (typeof entry.text === "string" && entry.details) {
            this.memory.set(key, entry); // promote disk hit → memory
            return entry;
          }
        } catch {
          // corrupt JSON → remove + miss
        }
        try {
          rmSync(file, { force: true });
        } catch {
          // best-effort cleanup
        }
      }
    }
    return undefined;
  }

  /** Store a successful result. Memory always; disk (atomic tmp+rename) when
   *  persisted. A disk write failure never fails the call (memory still has
   *  the entry for the session). */
  set(key: string, entry: CacheEntry): void {
    this.memory.set(key, entry);
    if (this.dir) {
      const file = this.fileFor(key);
      const tmp = `${file}.tmp`;
      try {
        writeFileSync(tmp, JSON.stringify(entry), "utf8");
        renameSync(tmp, file);
      } catch {
        // disk failure → memory-only degradation; don't throw
      }
      this.evictIfNeeded();
    }
  }

  /** Wipe both layers. */
  clear(): void {
    this.memory.clear();
    if (this.dir && existsSync(this.dir)) {
      try {
        for (const f of readdirSync(this.dir)) {
          if (f.endsWith(".json")) rmSync(join(this.dir, f), { force: true });
        }
      } catch {
        // best-effort
      }
    }
  }

  stats(): CacheStats {
    let diskEntries = 0;
    if (this.dir && existsSync(this.dir)) {
      try {
        diskEntries = readdirSync(this.dir).filter((f) => f.endsWith(".json")).length;
      } catch {
        // best-effort
      }
    }
    return {
      memoryEntries: this.memory.size,
      diskEntries,
      maxEntries: this.maxEntries,
      persisted: this.dir !== undefined,
    };
  }

  private fileFor(key: string): string {
    return join(this.dir!, `${key}.json`);
  }

  /** LRU eviction by file mtime: if disk entries exceed `maxEntries`, delete
   *  the oldest until under cap. Benign across concurrent sessions (worst
   *  case: a redundant eviction). */
  private evictIfNeeded(): void {
    if (!this.dir || !existsSync(this.dir)) return;
    let files: { name: string; mtime: number }[] = [];
    try {
      files = readdirSync(this.dir)
        .filter((f) => f.endsWith(".json"))
        .map((name) => {
          let mtime = 0;
          try {
            mtime = statSync(join(this.dir!, name)).mtimeMs;
          } catch {
            // unreadable file → mtime 0 (evicted first)
          }
          return { name, mtime };
        });
    } catch {
      return;
    }
    if (files.length <= this.maxEntries) return;
    files.sort((a, b) => a.mtime - b.mtime); // oldest first
    const toEvict = files.length - this.maxEntries;
    for (let i = 0; i < toEvict; i++) {
      const victim = files[i];
      if (!victim) continue;
      try {
        rmSync(join(this.dir!, victim.name), { force: true });
      } catch {
        // best-effort
      }
    }
  }
}