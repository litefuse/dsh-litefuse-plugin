import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'

/**
 * The harness's generated catalog of every event type this build declares.
 *
 * Loaded by file path because the package does not export it. That coupling is
 * deliberate rather than careless: if the file moves, this test fails, which is
 * exactly what a guard against harness drift should do — the alternative is
 * pinning our own copy of the list, which cannot notice the additions this
 * exists to catch.
 */
async function knownEventTypes(): Promise<ReadonlySet<string>> {
  const manifest = createRequire(import.meta.url).resolve('@deepseek-ai/dsh-session/package.json')
  const catalog = new URL('lib/types/known-event-types.js', pathToFileURL(manifest))
  const loaded = await import(catalog.href) as { KNOWN_SESSION_EVENT_TYPES: ReadonlySet<string> }
  return loaded.KNOWN_SESSION_EVENT_TYPES
}

/**
 * The event vocabulary is merge-extensible, so exhaustiveness cannot be checked
 * by the compiler: a plugin adds types by augmenting `SessionEventMap`, and
 * `record()` deliberately ignores what it does not model. That makes "did we
 * miss something the harness now emits?" a question no type error will ever
 * answer — a new event type simply flows into the default branch and is lost
 * without a sound.
 *
 * This pins the answer instead. Every type the harness declares is either
 * modelled or listed below with a reason, so a harness upgrade that adds one
 * fails here and forces a decision rather than a silent gap.
 */

/** Event types `record()` folds into the trace, read from the switch itself. */
const HANDLED = new Set(
  [...readFileSync(new URL('../src/trace.ts', import.meta.url), 'utf8')
    .matchAll(/case '([a-z][a-z0-9/-]*)':/g)]
    .map(match => match[1]!)
    .filter(type => type.includes('/')),
)

/** Types deliberately left out, each with the reason it carries no observation. */
const IGNORED = new Map([
  ['step/end', 'a generation closes on its assistant/message; the step marker adds nothing'],
  ['session/end-seed', 'a seed boundary, not agent work — the descriptor scan reads it off the log instead'],
  ['agent/inbox/spliced', 'inbox bookkeeping, not model or tool work'],
  ['agent-preset/selected', 'composition metadata, fixed for the session'],
  ['todo/write', 'UI list state, replaced wholesale and never a step'],
  ['session/title', 'a derived label for the session, not part of any turn'],
  ['session/title-llm-request', 'a side-channel model call that logs no usage; nothing to account'],
  ['session/title-llm-response', 'the other half of the same side-channel call'],
  ['web/deepseek-search-llm-request', 'a side-channel model call that logs no usage'],
  ['sandbox/mode', 'session policy; candidate trace metadata, not an observation'],
  ['approval/policy', 'session policy'],
  ['approval/asked', 'an approval prompt is not model or tool work'],
  ['approval/decided', 'an approval outcome is not model or tool work'],
  ['permission/preset', 'session policy'],
  ['plan/mode', 'session policy'],
  ['goal/change', 'session state, not a step'],
  ['schedule/change', 'session state, not a step'],
  ['feedback/record', 'user feedback, carried by the harness rather than the trace'],
  ['command/run', 'a slash command is not a model call'],
  ['command/done', 'a slash command is not a model call'],
  ['hook/invoked', 'hook execution is host work, not agent work'],
  ['hook/result', 'hook execution is host work, not agent work'],
  ['llm/retry', 'a retry is folded into the generation it belongs to'],
  ['llm/retry-started', 'a retry is folded into the generation it belongs to'],
  ['compaction/start', 'compaction is recorded once, at its end'],
  ['compaction/summary', 'compaction is recorded once, at its end'],
  ['compaction/prune', 'compaction is recorded once, at its end'],
  ['tool-workflow/run-start', 'workflow structure, not a model call or tool execution'],
  ['tool-workflow/run-end', 'workflow structure, not a model call or tool execution'],
  ['tool-workflow/agent-start', 'workflow structure, not a model call or tool execution'],
  ['tool-workflow/agent-end', 'workflow structure, not a model call or tool execution'],
])

describe('event vocabulary', () => {
  it('classifies every event type the harness declares', async () => {
    const unclassified = [...await knownEventTypes()]
      .filter(type => !HANDLED.has(type) && !IGNORED.has(type))
      .sort()
    expect(unclassified, 'new harness event types — model them or add a reason to IGNORED').toEqual([])
  })

  it('reads a real handled set rather than an empty one', () => {
    // Guards the regex above: if the switch is refactored into a shape it no
    // longer matches, HANDLED silently empties and the check above passes by
    // classifying nothing.
    expect(HANDLED.size).toBeGreaterThanOrEqual(12)
    expect(HANDLED).toContain('assistant/message')
    expect(HANDLED).toContain('tool/code-dispatch')
  })

  it('keeps IGNORED free of types that are actually handled', () => {
    expect([...IGNORED.keys()].filter(type => HANDLED.has(type))).toEqual([])
  })
})
