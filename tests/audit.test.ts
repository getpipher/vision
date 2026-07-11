/**
 * Unit tests for `lib/audit.ts` (SPEC-5 §3.1 / PLAN-5 §1.2, §1.3, step 1).
 *
 * The audit log is a persisted, append-only JSONL record of where each image
 * went (provider / model / cached / fallback / ok / error / latency). These
 * tests cover the pure read/write helpers: path resolution, image-path
 * truncation, append (with parent-dir creation + best-effort failure),
 * clear, tail (with corruption defense), count, + the concurrency guarantee
 * (parallel appends don't interleave or corrupt — T68).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync, statSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  appendAuditEntry,
  clearAuditLog,
  countAuditLog,
  resolveAuditPath,
  tailAuditLog,
  truncateImagePathForLog,
  type AuditEntry,
} from "../lib/audit.ts";

function tmpLogDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-audit-"));
}

function makeEntry(overrides: Partial<AuditEntry> = {}): AuditEntry {
  return {
    ts: "2026-07-11T12:00:00.000Z",
    provider: "Ollama",
    model: "minimax-m3:cloud",
    image_path: "/tmp/a.png",
    source_hash: "abc123",
    cached: false,
    fallback: false,
    fallback_model: undefined,
    ok: true,
    error_code: undefined,
    latency_ms: 42,
    local_only: false,
    ...overrides,
  };
}

// ── resolveAuditPath ────────────────────────────────────────────────────
test("resolveAuditPath: undefined config → default <agentDir>/vision-audit.log", () => {
  assert.equal(resolveAuditPath(undefined, "/x/agent"), "/x/agent/vision-audit.log");
  assert.equal(resolveAuditPath("", "/x/agent"), "/x/agent/vision-audit.log");
  assert.equal(resolveAuditPath("   ", "/x/agent"), "/x/agent/vision-audit.log");
});

test("resolveAuditPath: explicit config path wins (trimmed)", () => {
  assert.equal(resolveAuditPath("/custom/path.log", "/x/agent"), "/custom/path.log");
  assert.equal(resolveAuditPath("  /custom/path.log  ", "/x/agent"), "/custom/path.log");
});

// ── truncateImagePathForLog ─────────────────────────────────────────────
test("truncateImagePathForLog: file path full (has slash)", () => {
  assert.equal(truncateImagePathForLog("/tmp/a.png"), "/tmp/a.png");
  assert.equal(truncateImagePathForLog("~/x/screenshot.jpeg"), "~/x/screenshot.jpeg");
  assert.equal(truncateImagePathForLog("./relative/img.png"), "./relative/img.png");
});

test("truncateImagePathForLog: data: URL truncated to first 64 chars + …(N bytes)", () => {
  const url = "data:image/png;base64," + "A".repeat(500);
  const out = truncateImagePathForLog(url);
  assert.ok(out.startsWith(url.slice(0, 64)), "starts with first 64 chars");
  assert.ok(out.endsWith(" bytes)"), "ends with bytes suffix");
  assert.ok(out.includes("…"), "has ellipsis");
  // The byte count should reflect the full data URL length.
  assert.ok(out.includes(`${Buffer.byteLength(url, "utf8")} bytes`), "byte count accurate");
});

test("truncateImagePathForLog: long base64 (no slash, >200 chars) truncated", () => {
  const long = "B".repeat(300);
  const out = truncateImagePathForLog(long);
  assert.ok(out.startsWith(long.slice(0, 64)), "starts with first 64 chars");
  assert.ok(out.endsWith(" chars)"), "ends with chars suffix");
  assert.ok(out.includes("…"), "has ellipsis");
  assert.ok(out.includes("300 chars"), "char count accurate");
});

test("truncateImagePathForLog: short base64 (<200 chars) full", () => {
  const short = "C".repeat(68);
  assert.equal(truncateImagePathForLog(short), short);
});

test("truncateImagePathForLog: long path WITH slash stays full (conservative guard)", () => {
  const longPath = "/" + "d".repeat(300) + "/file.png";
  assert.equal(truncateImagePathForLog(longPath), longPath);
});

test("truncateImagePathForLog: base64 with a slash char → not truncated (conservative)", () => {
  // base64 alphabet includes '/' (the 63rd value). A long base64 that happens
  // to contain '/' is treated as a path (full). This is a deliberate
  // conservative false-negative: better to over-log than truncate a real path.
  const longWithSlash = "D".repeat(150) + "/" + "D".repeat(150);
  assert.equal(truncateImagePathForLog(longWithSlash), longWithSlash);
});

// ── appendAuditEntry ────────────────────────────────────────────────────
test("appendAuditEntry: creates file + parent dir, writes one JSONL line", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "nested", "deep", "audit.log");
    appendAuditEntry(path, makeEntry());
    assert.ok(existsSync(path), "file created");
    const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim().length > 0);
    assert.equal(lines.length, 1, "one line");
    const parsed = JSON.parse(lines[0]!) as AuditEntry;
    assert.equal(parsed.provider, "Ollama");
    assert.equal(parsed.model, "minimax-m3:cloud");
    assert.equal(parsed.source_hash, "abc123");
    assert.equal(parsed.ok, true);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("appendAuditEntry: two appends → two distinct lines (append, not overwrite)", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    appendAuditEntry(path, makeEntry({ source_hash: "h1" }));
    appendAuditEntry(path, makeEntry({ source_hash: "h2" }));
    const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim().length > 0);
    assert.equal(lines.length, 2, "two lines");
    assert.ok(lines[0]!.includes("h1"));
    assert.ok(lines[1]!.includes("h2"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("appendAuditEntry: best-effort — a write failure is swallowed, never throws", () => {
  const dir = tmpLogDir();
  try {
    // Point at a path whose parent is a file (mkdirSync recursive fails; the
    // append is swallowed). Use a real unwritable setup.
    const blocker = join(dir, "blocker");
    writeFileSync(blocker, "x"); // a file, not a dir
    const path = join(blocker, "audit.log"); // parent is a file → mkdir throws
    // Must not throw.
    assert.doesNotThrow(() => appendAuditEntry(path, makeEntry()));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── clearAuditLog ────────────────────────────────────────────────────────
test("clearAuditLog: truncates an existing file to 0 entries", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    appendAuditEntry(path, makeEntry({ source_hash: "h1" }));
    appendAuditEntry(path, makeEntry({ source_hash: "h2" }));
    assert.equal(countAuditLog(path), 2);
    clearAuditLog(path);
    assert.equal(countAuditLog(path), 0);
    // File still exists (truncated, not deleted).
    assert.ok(existsSync(path));
    assert.equal(statSync(path).size, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("clearAuditLog: no-op on missing file (no throw)", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "never-existed.log");
    assert.doesNotThrow(() => clearAuditLog(path));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── tailAuditLog ────────────────────────────────────────────────────────
test("tailAuditLog: returns last N entries, newest-last", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    for (let i = 0; i < 12; i++) {
      appendAuditEntry(path, makeEntry({ source_hash: `h${i}` }));
    }
    const tail = tailAuditLog(path, 10);
    assert.equal(tail.length, 10, "capped at N");
    // newest-last: the last entry is the most-recently-appended.
    assert.equal(tail[9]!.source_hash, "h11");
    assert.equal(tail[0]!.source_hash, "h2");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("tailAuditLog: fewer entries than N → returns all", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    appendAuditEntry(path, makeEntry({ source_hash: "only" }));
    const tail = tailAuditLog(path, 10);
    assert.equal(tail.length, 1);
    assert.equal(tail[0]!.source_hash, "only");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("tailAuditLog: skips corrupt lines (defensive — no throw)", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    appendAuditEntry(path, makeEntry({ source_hash: "good1" }));
    // Inject a corrupt line manually.
    writeFileSync(path, "this is not json\n", { flag: "a" });
    appendAuditEntry(path, makeEntry({ source_hash: "good2" }));
    const tail = tailAuditLog(path, 10);
    assert.equal(tail.length, 2, "corrupt line skipped, 2 valid returned");
    assert.equal(tail[0]!.source_hash, "good1");
    assert.equal(tail[1]!.source_hash, "good2");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("tailAuditLog: missing file → [] (no throw)", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "never-existed.log");
    assert.deepEqual(tailAuditLog(path, 10), []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── countAuditLog ────────────────────────────────────────────────────────
test("countAuditLog: counts non-empty lines", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    for (let i = 0; i < 5; i++) appendAuditEntry(path, makeEntry({ source_hash: `h${i}` }));
    assert.equal(countAuditLog(path), 5);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("countAuditLog: missing file → 0", () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "never-existed.log");
    assert.equal(countAuditLog(path), 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── T68: concurrency (★ PLAN-5 §1.2) ───────────────────────────────────
test("T68: 10 parallel appendAuditEntry → 10 distinct lines, no interleaving/corruption", async () => {
  const dir = tmpLogDir();
  try {
    const path = join(dir, "audit.log");
    // Fire 10 appends "concurrently" (synchronous calls in a Promise.all of
    // immediately-resolved microtasks). appendFileSync is synchronous so
    // these serialize in JS, but the test asserts the contract: N appends →
    // N parseable distinct lines, no corruption.
    const entries = Array.from({ length: 10 }, (_, i) =>
      makeEntry({ source_hash: `parallel-${i}`, latency_ms: i }),
    );
    // Use Promise.all over synchronous fns wrapped in Promise.resolve to
    // exercise the concurrency path the real batch uses (the batch awaits
    // delegateToVisionModel which awaits fetch; the audit write itself is sync
    // inside that async chain).
    await Promise.all(entries.map((e) => Promise.resolve(appendAuditEntry(path, e))));
    assert.equal(countAuditLog(path), 10, "10 entries");
    const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim().length > 0);
    assert.equal(lines.length, 10);
    // Every line parses + has a unique source_hash.
    const hashes = new Set<string>();
    for (const line of lines) {
      const parsed = JSON.parse(line) as AuditEntry;
      assert.ok(parsed.source_hash.startsWith("parallel-"));
      assert.ok(!hashes.has(parsed.source_hash), "no duplicate lines");
      hashes.add(parsed.source_hash);
    }
    assert.equal(hashes.size, 10, "all 10 distinct");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});