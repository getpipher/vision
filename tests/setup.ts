// Agent-dir redirect for the test suite — imported as the first statement of
// every tests/*.test.ts file, so it takes effect however the suite is started:
// `pnpm test`, a bare `tsx --test`, or a single file. (A meta-test in
// delegate.test.ts enforces that every test file still does so.)
//
// Why: config, cache and the audit log all resolve through getAgentDir(), which
// reads PI_CODING_AGENT_DIR at call time and otherwise falls back to the real
// ~/.pi/agent. Any test that delegates with the default config (auditLog is on)
// would append fixture rows — fake providers, images that never existed — to the
// developer's production vision-audit.log.
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";

/** The real pi config root — nothing the suite writes may land inside it. */
export const REAL_PI_DIR = join(homedir(), ".pi");

/** Mirror of the tilde handling getAgentDir() applies to the env var, so the
 *  value we hand on is the one getAgentDir() will return. */
function expand(dir: string): string {
  if (dir === "~") return homedir();
  if (dir.startsWith("~/")) return join(homedir(), dir.slice(2));
  return resolve(dir);
}

/** True if `dir` is ~/.pi itself or anything below it. Separator-aware: a
 *  sibling such as ~/.pi-sandbox is a legitimate test dir, not the real one. */
export function isInsideRealPiDir(dir: string): boolean {
  const abs = expand(dir);
  return abs === REAL_PI_DIR || abs.startsWith(REAL_PI_DIR + sep);
}

// Respect a dir supplied from outside (CI, a wrapper script) — but never one
// that points back into ~/.pi, which is exactly the pollution this file exists
// to prevent. Only the dir we created ourselves is ours to delete.
let ownDir: string | undefined;
const inherited = process.env.PI_CODING_AGENT_DIR;

if (inherited && !isInsideRealPiDir(inherited)) {
  process.env.PI_CODING_AGENT_DIR = expand(inherited);
  mkdirSync(process.env.PI_CODING_AGENT_DIR, { recursive: true });
} else {
  if (inherited) {
    console.warn(
      `[tests/setup] ignoring PI_CODING_AGENT_DIR=${inherited} — it resolves inside ${REAL_PI_DIR}; using a temp dir instead`,
    );
  }
  ownDir = mkdtempSync(join(tmpdir(), "vision-test-agent-"));
  process.env.PI_CODING_AGENT_DIR = ownDir;
}

function cleanup(): void {
  if (ownDir) rmSync(ownDir, { recursive: true, force: true });
}

process.on("exit", cleanup);

// `exit` never fires when we are terminated by a signal — a Ctrl-C on the suite
// reaches every per-file child process and would otherwise leak one temp dir
// each. rmSync({ force: true }) is idempotent, so the extra `exit` pass that
// process.exit() triggers is harmless.
for (const [signal, code] of [
  ["SIGINT", 130],
  ["SIGTERM", 143],
  ["SIGHUP", 129],
] as const) {
  process.on(signal, () => {
    cleanup();
    process.exit(code);
  });
}
