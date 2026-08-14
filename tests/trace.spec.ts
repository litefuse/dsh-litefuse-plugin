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
  requestInput: 'full',
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

/** Drive one two-step turn: a bash call, then the final answer. */
function runToolTurn(writer: TurnWriter): void {
  writer.start().prompt('list the files').header()
  const first = writer.openStep()
  writer.assistant(first, [
    { type: 'reasoning', text: 'I should look at the directory.' },
    { type: 'tool-call', id: call('call-1'), name: 'bash', arguments: '{"command":"ls -la /tmp"}' },
  ], { inputTokens: 100, outputTokens: 20, cacheReadTokens: 900, cacheWriteTokens: 12, reasoningTokens: 7 })
  const callSeq = writer.toolCall(first, 'call-1', 'bash', { command: 'cd /tmp && ls -la' })
  writer.toolResult(first, 'call-1', callSeq, 'total 0').closeStep(first)
  const second = writer.openStep()
  writer.assistant(second, [{ type: 'text', text: 'The directory is empty.' }], { inputTokens: 140, outputTokens: 8 })
  writer.closeStep(second).end()
}

describe('trace assembly', () => {
  it('builds one trace per turn with a flat agent root', async () => {
    const { ctx, spans } = await observe()
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    expect(spans.map(span => span.name)).toEqual([
      'plan (1 tool) #1',
      'tool: bash (ls) #2',
      'response',
      'DeepSeek Harness — Turn 1',
    ])
    const root = spans[3]!
    expect(attribute(root, 'langfuse.observation.type')).toBe('agent')
    expect(root.parentSpanId).toBeUndefined()
    for (const span of spans.slice(0, 3)) {
      expect(span.parentSpanId).toBe(root.spanId)
      expect(span.traceId).toBe(root.traceId)
    }
  })

  it('carries the trace header on every span so the trace appears before the turn ends', async () => {
    const { ctx, spans } = await observe()
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    for (const span of spans) {
      expect(attribute(span, 'langfuse.trace.name')).toBe('DeepSeek Harness — Turn 1')
      expect(attribute(span, 'langfuse.trace.input')).toBe('list the files')
      expect(attribute(span, 'user.id')).toBe('tester')
      expect(attribute(span, 'langfuse.environment')).toBe('test')
      expect(span.attributes['langfuse.trace.tags']).toEqual(['dsh', 'model:deepseek-chat'])
    }
    // Only the root closes the trace, so only it carries the final answer.
    expect(spans.filter(span => attribute(span, 'langfuse.trace.output') !== undefined)).toHaveLength(1)
    expect(attribute(spans[3]!, 'langfuse.trace.output')).toBe('The directory is empty.')
  })

  it('splits reasoning out of the completion total, as the Litefuse classifier expects', async () => {
    const { ctx, spans } = await observe()
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    const plan = spans[0]!
    // `output` carries completion MINUS reasoning, and the two sum back to the
    // 20 the adapter reported. Litefuse adds up every key containing `output`,
    // and its own ingestion processor normalizes provider payloads the same
    // way, so this split is what makes the displayed figure the billed one.
    expect(JSON.parse(attribute(plan, 'langfuse.observation.usage_details')!)).toEqual({
      input: 100,
      output: 13,
      output_reasoning_tokens: 7,
      cache_read_input_tokens: 900,
      cache_creation_input_tokens: 12,
    })
    expect(attribute(plan, 'langfuse.observation.model.name')).toBe('deepseek-chat')
    expect(metadataOf(plan)['agent_reasoning_tokens']).toBe(7)
  })

  it('numbers generations and tools in one shared sequence and links a tool to its plan', async () => {
    const { ctx, spans } = await observe()
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    expect(metadataOf(spans[0]!)['agent_step_index']).toBe(1)
    const tool = metadataOf(spans[1]!)
    expect(tool['agent_step_index']).toBe(2)
    expect(tool['agent_plan_step']).toBe(1)
    expect(tool['agent_tool_call_id']).toBe('call-1')
    expect(tool['agent_tool_name']).toBe('bash')
    // `response` carries no #N in its name but keeps its own step index.
    expect(metadataOf(spans[2]!)['agent_step_index']).toBe(3)
  })

  it('records real latency and time to first token', async () => {
    const { ctx, spans } = await observe()
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    for (const span of spans) {
      expect(span.endTimeMillis).toBeGreaterThanOrEqual(span.startTimeMillis)
    }
    expect(metadataOf(spans[0]!)['agent_time_to_first_token_ms']).toBeTypeOf('number')
    expect(metadataOf(spans[1]!)['agent_duration_ms']).toBeTypeOf('number')
  })

  it('names a step by its primary action, not by the model or the extras it carried', async () => {
    const { ctx, spans } = await observe()
    const writer = new TurnWriter(createSession(ctx), 1)
    writer.start().prompt('think about it').header()
    const thinking = writer.openStep()
    writer.assistant(thinking, [{ type: 'reasoning', text: 'hmm' }]).closeStep(thinking)
    const planning = writer.openStep()
    writer.assistant(planning, [
      { type: 'text', text: 'let me check two things' },
      { type: 'tool-call', id: call('a'), name: 'read', arguments: '{"file_path":"/tmp/a/b/notes.txt"}' },
      { type: 'tool-call', id: call('b'), name: 'grep', arguments: '{"pattern":"TODO"}' },
    ])
    const seqA = writer.toolCall(planning, 'a', 'read', { file_path: '/tmp/a/b/notes.txt' })
    const seqB = writer.toolCall(planning, 'b', 'grep', { pattern: 'TODO' })
    writer.toolResult(planning, 'a', seqA, 'contents').toolResult(planning, 'b', seqB, 'no matches')
    writer.closeStep(planning).end()

    expect(spans.map(span => span.name)).toEqual([
      'think #1',
      'plan (2 tools) #2',
      'tool: read (notes.txt) #3',
      'tool: grep (TODO) #4',
      'DeepSeek Harness — Turn 1',
    ])
    // Both tools name the same plan, which is what makes the pairing queryable.
    expect(metadataOf(spans[2]!)['agent_plan_step']).toBe(2)
    expect(metadataOf(spans[3]!)['agent_plan_step']).toBe(2)
  })

  it('marks a failed tool ERROR and a turn with no answer WARNING', async () => {
    const { ctx, spans } = await observe()
    const writer = new TurnWriter(createSession(ctx), 1)
    writer.start().prompt('break it').header()
    const step = writer.openStep()
    writer.assistant(step, [{ type: 'tool-call', id: call('c'), name: 'bash', arguments: '{"command":"false"}' }])
    const seq = writer.toolCall(step, 'c', 'bash', { command: 'false' })
    writer.toolResult(step, 'c', seq, 'exit status 1', true)
    writer.closeStep(step).end()

    expect(spans.map(span => span.name)).toEqual([
      'plan (1 tool) #1',
      'tool: bash (false) #2',
      'DeepSeek Harness — Turn 1',
    ])
    expect(attribute(spans[1]!, 'langfuse.observation.level')).toBe('ERROR')
    expect(attribute(spans[1]!, 'langfuse.observation.status_message')).toBe('exit status 1')
    expect(metadataOf(spans[1]!)['agent_is_error']).toBe(true)
    expect(attribute(spans[2]!, 'langfuse.observation.level')).toBe('WARNING')
    expect(attribute(spans[2]!, 'langfuse.observation.status_message'))
      .toBe('turn ended without a final text response')
  })

  it('reports a failed turn as an ERROR root', async () => {
    const { ctx, spans } = await observe()
    const writer = new TurnWriter(createSession(ctx), 1)
    writer.start().prompt('fail').header().end({ kind: 'error', error: { message: 'boom', code: 'UNKNOWN' } } as never)

    expect(attribute(spans[0]!, 'langfuse.observation.level')).toBe('ERROR')
    expect(metadataOf(spans[0]!)['agent_end_reason']).toBe('error')
  })

  it('closes an interrupted step as a warning instead of losing it', async () => {
    const { ctx, spans } = await observe()
    const writer = new TurnWriter(createSession(ctx), 1)
    writer.start().prompt('cancel me').header()
    const step = writer.openStep()
    writer.assistant(step, [{ type: 'tool-call', id: call('d'), name: 'bash', arguments: '{"command":"sleep 60"}' }])
    writer.toolCall(step, 'd', 'bash', { command: 'sleep 60' })
    writer.end({ kind: 'aborted', reason: { kind: 'user' } } as never)

    expect(spans.map(span => span.name)).toEqual([
      'plan (1 tool) #1',
      'tool: bash (sleep) #2',
      'DeepSeek Harness — Turn 1',
    ])
    expect(attribute(spans[1]!, 'langfuse.observation.status_message')).toBe('turn ended before the tool completed')
  })

  it('keeps turn numbering from the session log so a resumed conversation continues', async () => {
    const { ctx, spans } = await observe()
    const session = createSession(ctx)
    runToolTurn(new TurnWriter(session, 7))
    runToolTurn(new TurnWriter(session, 8))

    const roots = spans.filter(span => attribute(span, 'langfuse.observation.type') === 'agent')
    expect(roots.map(span => span.name)).toEqual(['DeepSeek Harness — Turn 7', 'DeepSeek Harness — Turn 8'])
    expect(roots[0]!.traceId).not.toBe(roots[1]!.traceId)
    expect(attribute(roots[0]!, 'session.id')).toBe(attribute(roots[1]!, 'session.id'))
  })

  it('sends the whole request as the generation input, minus the reply itself', async () => {
    const { ctx, spans } = await observe()
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    const input = JSON.parse(attribute(spans[0]!, 'langfuse.observation.input')!) as {
      system: string
      tools: string[]
      messages: { role: string }[]
    }
    expect(input.system).toBe('You are a coding agent.')
    expect(input.tools).toEqual(['bash'])
    expect(input.messages.map(message => message.role)).toEqual(['user'])
  })

  it('records only the new messages when the request-input scope is delta', async () => {
    const { ctx, spans } = await observe({ requestInput: 'delta' })
    runToolTurn(new TurnWriter(createSession(ctx), 1))

    expect(JSON.parse(attribute(spans[0]!, 'langfuse.observation.input')!)).toHaveLength(1)
    // The second call sees only the tool result appended since the first.
    const secondCall = JSON.parse(attribute(spans[2]!, 'langfuse.observation.input')!) as { role: string }[]
    expect(secondCall).toHaveLength(1)
    expect(metadataOf(spans[2]!)['agent_input_scope']).toBe('delta')
  })

  it('truncates an oversized value and records what was cut', async () => {
    const { ctx, spans } = await observe({ maxValueChars: 32 })
    const writer = new TurnWriter(createSession(ctx), 1)
    writer.start().prompt('long').header()
    const step = writer.openStep()
    writer.assistant(step, [{ type: 'text', text: 'x'.repeat(200) }])
    writer.closeStep(step).end()

    const generation = spans[0]!
    expect(attribute(generation, 'langfuse.observation.output')).toHaveLength(33)
    expect(metadataOf(generation)['agent_output_truncated']).toBe(true)
    expect(metadataOf(generation)['agent_output_orig_len']).toBe(200)
  })

  it('ignores events for a turn it never saw open', async () => {
    const { ctx, spans } = await observe()
    const session = createSession(ctx)
    // No turn/start: the plugin loaded mid-turn and does not invent a root.
    session.append('step/start', { turn: 1, step: 1 })
    session.append('tool/call', { turn: 1, step: 1, callId: call('x'), name: 'bash', arguments: '{}' })

    expect(spans).toEqual([])
  })
})
