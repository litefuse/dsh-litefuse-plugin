import { describe, expect, it } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { TraceAssembler, type AssemblerOptions } from '../src/trace.js'
import type { LitefuseSpan } from '../src/otlp.js'
import { attribute, call, createContext, createSession, metadataOf, TurnWriter } from './fixture.js'

const OPTIONS: AssemblerOptions = {
  agentName: 'DeepSeek Harness',
  environment: 'test',
  userId: 'tester',
  tags: ['dsh'],
  maxValueChars: 10_000,
  requestInput: 'none',
  delegationTools: ['subagent', 'subagent_fork'],
}

/** Mount an assembler on the session firehose and collect everything it emits. */
async function observe(options: Partial<AssemblerOptions> = {}): Promise<{ ctx: Context; spans: LitefuseSpan[] }> {
  const ctx = await createContext()
  const spans: LitefuseSpan[] = []
  const assembler = new TraceAssembler({ ...OPTIONS, ...options }, span => spans.push(span))
  ctx.on('session/event', (session, event) => assembler.record(session, event))
  return { ctx, spans }
}

/** Run one child session's whole delegated run. */
function runChild(ctx: Context, parentId: ReturnType<typeof createSession>['id'], answer: string): void {
  const child = new TurnWriter(createSession(ctx, parentId), 1)
  child.start().prompt('summarize the notes file').header()
  const step = child.openStep()
  child.assistant(step, [{ type: 'tool-call', id: call('child-1'), name: 'read', arguments: '{"file_path":"/tmp/notes.txt"}' }])
  const seq = child.toolCall(step, 'child-1', 'read', { file_path: '/tmp/notes.txt' })
  child.toolResult(step, 'child-1', seq, 'note contents').closeStep(step)
  const answerStep = child.openStep()
  child.assistant(answerStep, [{ type: 'text', text: answer }], { inputTokens: 30, outputTokens: 5 })
  child.closeStep(answerStep).end()
}

describe('subagent subtree', () => {
  it('mounts a delegated run as a container under the delegation tool span', async () => {
    const { ctx, spans } = await observe()
    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate the summary').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{"description":"summarize"}' }])
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'summarize', prompt: 'summarize the notes file' })

    runChild(ctx, parent.session.id, 'the notes cover the release plan')

    parent.toolResult(step, 'p1', seq, 'the notes cover the release plan').closeStep(step)
    const answerStep = parent.openStep()
    parent.assistant(answerStep, [{ type: 'text', text: 'done' }])
    parent.closeStep(answerStep).end()

    expect(spans.map(span => span.name)).toEqual([
      'plan (1 tool) #1',
      'plan (1 tool) #1',
      'tool: read (notes.txt) #2',
      'subagent response',
      'subagent',
      'tool (1 subagent) #2',
      'response',
      'DeepSeek Harness — Turn 1',
    ])

    const container = spans[4]!
    const delegation = spans[5]!
    const root = spans[7]!
    expect(attribute(container, 'langfuse.observation.type')).toBe('agent')
    expect(container.parentSpanId).toBe(delegation.spanId)
    expect(delegation.parentSpanId).toBe(root.spanId)
    // One trace: the child's usage rolls into the parent's total cost.
    for (const span of spans) expect(span.traceId).toBe(root.traceId)
  })

  it('restarts container numbering at #1 and keeps the child steps inside it', async () => {
    const { ctx, spans } = await observe()
    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }])
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'summarize' })
    runChild(ctx, parent.session.id, 'child answer')
    parent.toolResult(step, 'p1', seq, 'child answer').closeStep(step).end()

    const childPlan = spans[1]!
    const childTool = spans[2]!
    const container = spans[4]!
    expect(metadataOf(childPlan)['agent_step_index']).toBe(1)
    expect(metadataOf(childTool)['agent_step_index']).toBe(2)
    expect(metadataOf(childTool)['agent_plan_step']).toBe(1)
    for (const span of [childPlan, childTool, spans[3]!]) {
      expect(span.parentSpanId).toBe(container.spanId)
    }
    expect(metadataOf(container)['agent_subagent']).toBe(true)
    expect(attribute(container, 'langfuse.observation.output')).toBe('child answer')
  })

  it('gives a container no trace header, because the trace belongs to the parent', async () => {
    const { ctx, spans } = await observe()
    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }])
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'summarize' })
    runChild(ctx, parent.session.id, 'child answer')
    parent.toolResult(step, 'p1', seq, 'child answer').closeStep(step).end()

    for (const span of spans.slice(1, 5)) {
      expect(attribute(span, 'langfuse.trace.name')).toBeUndefined()
      expect(attribute(span, 'session.id')).toBeUndefined()
      expect(attribute(span, 'langfuse.trace.output')).toBeUndefined()
      expect(attribute(span, 'langfuse.environment')).toBe('test')
    }
  })

  it('collects several children of one delegation under that one tool span', async () => {
    const { ctx, spans } = await observe()
    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate twice').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }])
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'fan out' })
    runChild(ctx, parent.session.id, 'first')
    runChild(ctx, parent.session.id, 'second')
    parent.toolResult(step, 'p1', seq, 'both done').closeStep(step).end()

    const delegation = spans.find(span => span.name.startsWith('tool ('))!
    expect(delegation.name).toBe('tool (2 subagents) #2')
    const containers = spans.filter(span => span.name === 'subagent')
    expect(containers).toHaveLength(2)
    for (const container of containers) expect(container.parentSpanId).toBe(delegation.spanId)
  })

  it('binds to the delegation call even while an ordinary tool runs beside it', async () => {
    const { ctx, spans } = await observe()
    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('parallel work').header()
    const step = parent.openStep()
    parent.assistant(step, [
      { type: 'tool-call', id: call('bash-1'), name: 'bash', arguments: '{"command":"sleep 1"}' },
      { type: 'tool-call', id: call('sub-1'), name: 'subagent', arguments: '{}' },
    ])
    const bashSeq = parent.toolCall(step, 'bash-1', 'bash', { command: 'sleep 1' })
    const subSeq = parent.toolCall(step, 'sub-1', 'subagent', { description: 'analyze' })
    runChild(ctx, parent.session.id, 'analysis')
    parent.toolResult(step, 'bash-1', bashSeq, 'slept')
    parent.toolResult(step, 'sub-1', subSeq, 'analysis').closeStep(step).end()

    const bash = spans.find(span => span.name.startsWith('tool: bash'))!
    const delegation = spans.find(span => span.name.startsWith('tool ('))!
    expect(bash.name).toBe('tool: bash (sleep) #2')
    expect(delegation.name).toBe('tool (1 subagent) #3')
    expect(spans.find(span => span.name === 'subagent')!.parentSpanId).toBe(delegation.spanId)
  })

  it('gives an unbound child session its own trace', async () => {
    const { ctx, spans } = await observe()
    const parent = createSession(ctx)
    // No delegation call is in flight: a background child owns its own trace.
    runChild(ctx, parent.id, 'independent')

    const roots = spans.filter(span => attribute(span, 'langfuse.observation.type') === 'agent')
    expect(roots.map(span => span.name)).toEqual(['DeepSeek Harness — Turn 1'])
    expect(metadataOf(roots[0]!)['agent_parent_session_id']).toBe(String(parent.id))
    expect(attribute(roots[0]!, 'session.id')).not.toBe(String(parent.id))
  })

  it('does not warn when the child session is disposed before the parent records the result', async () => {
    const ctx = await createContext()
    const spans: LitefuseSpan[] = []
    const assembler = new TraceAssembler(OPTIONS, span => spans.push(span))
    ctx.on('session/event', (session, event) => assembler.record(session, event))

    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }])
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'summarize' })
    const child = createSession(ctx, parent.session.id)
    const childWriter = new TurnWriter(child, 1)
    childWriter.start().prompt('work').header()
    const childStep = childWriter.openStep()
    childWriter.assistant(childStep, [{ type: 'text', text: 'child answer' }], { inputTokens: 30, outputTokens: 5 })
    childWriter.closeStep(childStep).end()
    // The harness disposes a delegated run BEFORE the parent appends its
    // tool/result; a successful child must not be reported as interrupted.
    assembler.closeSession(String(child.id), Date.now())
    parent.toolResult(step, 'p1', seq, 'child answer').closeStep(step).end()

    const container = spans.find(span => span.name === 'subagent')!
    expect(attribute(container, 'langfuse.observation.level')).toBeUndefined()
    expect(attribute(container, 'langfuse.observation.status_message')).toBeUndefined()
    expect(attribute(container, 'langfuse.observation.output')).toBe('child answer')
    expect(container.parentSpanId).toBe(spans.find(span => span.name.startsWith('tool ('))!.spanId)
  })

  it('still reports a failed delegated run as failed', async () => {
    const ctx = await createContext()
    const spans: LitefuseSpan[] = []
    const assembler = new TraceAssembler(OPTIONS, span => spans.push(span))
    ctx.on('session/event', (session, event) => assembler.record(session, event))

    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }])
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'summarize' })
    const child = createSession(ctx, parent.session.id)
    const childWriter = new TurnWriter(child, 1)
    childWriter.start().prompt('work').header()
    childWriter.end({ kind: 'error', error: { message: 'boom', code: 'UNKNOWN' } } as never)
    assembler.closeSession(String(child.id), Date.now())
    parent.toolResult(step, 'p1', seq, 'subagent run failed', true).closeStep(step).end()

    const container = spans.find(span => span.name === 'subagent')!
    expect(attribute(container, 'langfuse.observation.level')).toBe('ERROR')
  })

  it('rolls the child tokens into the root total without double counting them', async () => {
    const { ctx, spans } = await observe()
    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }],
      { inputTokens: 100, outputTokens: 10 })
    const seq = parent.toolCall(step, 'p1', 'subagent', { description: 'summarize' })
    // The child spends 30 + 20 input and 5 + 3 output across its two calls.
    runChild(ctx, parent.session.id, 'child answer')
    parent.toolResult(step, 'p1', seq, 'child answer').closeStep(step).end()

    const container = spans.find(span => span.name === 'subagent')!
    const root = spans[spans.length - 1]!
    expect(metadataOf(container)['agent_input_tokens']).toBe(30)
    expect(metadataOf(root)['agent_input_tokens']).toBe(130)
    expect(metadataOf(root)['agent_output_tokens']).toBe(15)
    expect(metadataOf(root)['agent_total_tokens']).toBe(145)
    // Two of the three model calls reported accounting; the child's plan step
    // carried none, so it contributes nothing rather than a zero.
    expect(metadataOf(root)['agent_accounted_generations']).toBe(2)
    // The rollup lives in metadata only; usage_details stays on generations so
    // Litefuse's own cost aggregation cannot count the same tokens twice.
    expect(root.attributes['langfuse.observation.usage_details']).toBeUndefined()
    expect(container.attributes['langfuse.observation.usage_details']).toBeUndefined()
  })

  it('closes a container left open when the parent turn ends without a result', async () => {
    const ctx = await createContext()
    const spans: LitefuseSpan[] = []
    const assembler = new TraceAssembler(OPTIONS, span => spans.push(span))
    ctx.on('session/event', (session, event) => assembler.record(session, event))

    const parent = new TurnWriter(createSession(ctx), 1)
    parent.start().prompt('delegate').header()
    const step = parent.openStep()
    parent.assistant(step, [{ type: 'tool-call', id: call('p1'), name: 'subagent', arguments: '{}' }])
    parent.toolCall(step, 'p1', 'subagent', { description: 'never returns' })
    const child = new TurnWriter(createSession(ctx, parent.session.id), 1)
    child.start().prompt('work').header()

    parent.end({ kind: 'aborted', reason: { kind: 'user' } } as never)

    expect(spans.map(span => span.name)).toContain('subagent')
    const container = spans.find(span => span.name === 'subagent')!
    expect(attribute(container, 'langfuse.observation.level')).toBe('WARNING')
    expect(attribute(container, 'langfuse.observation.status_message'))
      .toBe('delegation ended before the subagent returned')
  })
})
