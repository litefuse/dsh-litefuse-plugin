/**
 * The integration's own local log file.
 *
 * A booted dsh profile composes no logger plugin, so `ctx.logger` output is not
 * visible to the person running the agent. Every Litefuse integration therefore
 * keeps a local log of its own — the file to look at when traces do not appear —
 * and the trace spec requires exactly that: a failure here writes a line and
 * returns, never disturbing the agent.
 *
 * @module dsh-litefuse/log
 */

import { appendFileSync, mkdirSync, renameSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

/** Size at which the log rotates to `<file>.1`, so it cannot grow without bound. */
const ROTATE_BYTES = 4 * 1024 * 1024

/** Levels the integration writes; `debug` lines are opt-in. */
export interface LitefuseLog {
  /** Record one normal lifecycle line. */
  info(line: string): void
  /** Record one problem the operator should see. */
  warn(line: string): void
  /** Record one verbose line, dropped unless debug logging is on. */
  debug(line: string): void
  /** Absolute path of the file being written. */
  readonly file: string
}

/**
 * Resolve the harness home the same way dsh does, so the log sits beside the
 * sessions and settings it describes.
 * @returns the absolute harness-home directory.
 */
export function harnessHome(): string {
  const configured = process.env['DSH_HOME']
  return configured !== undefined && configured.length > 0 ? configured : join(homedir(), '.dsh')
}

/**
 * Open the integration's log.
 * @param file - absolute path to write; the parent directory is created.
 * @param debug - whether verbose lines are kept.
 * @returns a log whose every method is failure-contained.
 */
export function createLog(file: string, debug: boolean): LitefuseLog {
  const write = (level: string, line: string): void => {
    try {
      rotate(file)
      mkdirSync(dirname(file), { recursive: true })
      appendFileSync(file, `${new Date().toISOString()} [${level}] ${line}\n`)
    } catch {
      // The log is best-effort by definition: a read-only or full disk must not
      // turn observability into an agent failure, and there is no second place
      // to report it to.
    }
  }
  return {
    file,
    info: line => write('info', line),
    warn: line => write('warn', line),
    debug: line => {
      if (debug) write('debug', line)
    },
  }
}

/** Move an oversized log aside, keeping exactly one previous generation. */
function rotate(file: string): void {
  try {
    if (statSync(file).size < ROTATE_BYTES) return
  } catch {
    // No file yet, which is the common case on the first write.
    return
  }
  renameSync(file, `${file}.1`)
}
