/**
 * Post one synthetic verification trace to a Litefuse project.
 *
 * It drives the real `@deepseek-ai/dsh-session` Session through a scripted turn
 * — a tool call plus a delegated subagent run — and ships the result through
 * the same assembler and exporter the plugin uses. Run it to confirm that a
 * Litefuse deployment accepts this integration's wire format without waiting
 * for a model round trip.
 *
 *   node scripts/send-verification-trace.mjs
 *
 * Reads LITEFUSE_PUBLIC_KEY / LITEFUSE_SECRET_KEY / LITEFUSE_BASE_URL from the
 * environment. The trace is tagged `environment: development` and named
 * `dsh-litefuse-plugin-verify — Turn 1` so it cannot be mistaken for real agent
 * traffic in a dashboard.
 */

import { Context } from '@deepseek-ai/cordis'
import SessionStore from '@deepseek-ai/dsh-session'
import { CallId, MessageId } from '@deepseek-ai/dsh-llm'
import { TraceAssembler } from '../lib/trace.js'
import { LitefuseSpanExporter } from '../lib/otlp.js'

const baseUrl = process.env.LITEFUSE_BASE_URL ?? 'https://litefuse.cloud'
const publicKey = process.env.LITEFUSE_PUBLIC_KEY ?? process.env.LANGFUSE_PUBLIC_KEY
const secretKey = process.env.LITEFUSE_SECRET_KEY ?? process.env.LANGFUSE_SECRET_KEY
if (!publicKey || !secretKey) {
  console.error('set LITEFUSE_PUBLIC_KEY and LITEFUSE_SECRET_KEY first')
  process.exit(1)
}

const APPEND = { surfaceOp: 'append' }
const spans = []
const exporter = new LitefuseSpanExporter({
  baseUrl,
  credentials: async () => ({ publicKey, secretKey }),
  exportDelayMillis: 50,
  requestTimeoutMillis: 15_000,
  onFailure: message => console.error('FAIL', message),
  onSuccess: message => console.log('ok  ', message),
}, {
  resource: { 'service.name': 'deepseek-harness', 'service.version': 'verify' },
  scopeName: 'langfuse-sdk-dsh-litefuse-plugin',
  scopeVersion: 'verify',
})

const assembler = new TraceAssembler({
  agentName: 'dsh-litefuse-plugin-verify',
  environment: 'development',
  userId: 'verification',
  tags: ['dsh', 'verification'],
  maxValueChars: 100_000,
  requestInput: 'full',
  delegationTools: ['subagent', 'subagent_fork'],
}, (span) => {
  spans.push(span)
  exporter.enqueue(span)
})

const ctx = new Context()
ctx.plugin(SessionStore)
await new Promise(resolve => setTimeout(resolve, 30))
ctx.on('session/event', (session, event) => assembler.record(session, event))

/** Append one model call's assembled message. */
const assistant = (session, turn, step, content, usage) => session.append('assistant/message', {
  turn,
  step,
  message: {
    id: MessageId(`a-${turn}-${step}-${Math.random().toString(16).slice(2)}`),
    role: 'assistant',
    content,
    source: { kind: 'model', provider: 'deepseek', model: 'deepseek-chat' },
  },
  ...usage ? { usage } : {},
}, APPEND)

/** Append one tool result, closing its span. */
const toolResult = (session, turn, step, callId, callSeq, text) => session.append('tool/result', {
  turn,
  step,
  message: {
    id: MessageId(`r-${callId}`),
    role: 'user',
    content: [{ type: 'tool-result', toolCallId: CallId(callId), content: [{ type: 'text', text }] }],
    source: { kind: 'tool', callId: CallId(callId) },
  },
}, { ...APPEND, sourceEventSeqs: [callSeq] })

/** Append the request header that carries model identity. */
const header = session => session.append('request/header', {
  header: {
    config: { provider: 'deepseek', model: 'deepseek-chat', temperature: 0.2, maxTokens: 4096 },
    system: 'You are a coding agent.',
    tools: [{ name: 'bash', description: 'Run a command.', parameters: {} }],
  },
  reason: 'initial',
})

const wait = ms => new Promise(resolve => setTimeout(resolve, ms))

const parent = ctx.sessions.create(undefined, { meta: { cwd: process.cwd() } })
parent.append('turn/start', { turn: 1 })
parent.append('user/message', {
  id: MessageId('u-1'),
  role: 'user',
  content: [{ type: 'text', text: 'check the build and delegate the summary' }],
  source: { kind: 'user' },
}, APPEND)
header(parent)

parent.append('step/start', { turn: 1, step: 1 })
parent.append('assistant/chunk', { turn: 1, step: 1, chunk: { type: 'block-start', index: 0, blockType: 'text' } })
await wait(120)
assistant(parent, 1, 1, [
  { type: 'reasoning', text: 'First look at the build output.' },
  { type: 'tool-call', id: CallId('c1'), name: 'bash', arguments: '{"command":"pnpm run build"}' },
], { inputTokens: 1200, outputTokens: 96, cacheReadTokens: 18_000, reasoningTokens: 40 })
const c1 = parent.append('tool/call', { turn: 1, step: 1, callId: CallId('c1'), name: 'bash', arguments: '{"command":"pnpm run build"}' })
await wait(80)
toolResult(parent, 1, 1, 'c1', c1.seq, 'build succeeded in 4.1s')
parent.append('step/end', { turn: 1, step: 1 })

parent.append('step/start', { turn: 1, step: 2 })
parent.append('assistant/chunk', { turn: 1, step: 2, chunk: { type: 'block-start', index: 0, blockType: 'text' } })
await wait(90)
assistant(parent, 1, 2, [
  { type: 'tool-call', id: CallId('c2'), name: 'subagent', arguments: '{"description":"summarize the build"}' },
], { inputTokens: 1400, outputTokens: 40 })
const c2 = parent.append('tool/call', { turn: 1, step: 2, callId: CallId('c2'), name: 'subagent', arguments: '{"description":"summarize the build"}' })

const child = ctx.sessions.create(undefined, {
  meta: { cwd: process.cwd(), parentSession: parent.id, origin: 'subagent' },
})
child.append('turn/start', { turn: 1 })
child.append('user/message', {
  id: MessageId('u-child'),
  role: 'user',
  content: [{ type: 'text', text: 'summarize the build output' }],
  source: { kind: 'user' },
}, APPEND)
header(child)
child.append('step/start', { turn: 1, step: 1 })
await wait(70)
assistant(child, 1, 1, [
  { type: 'tool-call', id: CallId('k1'), name: 'read', arguments: '{"file_path":"/tmp/build.log"}' },
], { inputTokens: 300, outputTokens: 20 })
const k1 = child.append('tool/call', { turn: 1, step: 1, callId: CallId('k1'), name: 'read', arguments: '{"file_path":"/tmp/build.log"}' })
await wait(40)
toolResult(child, 1, 1, 'k1', k1.seq, '… 240 lines …')
child.append('step/end', { turn: 1, step: 1 })
child.append('step/start', { turn: 1, step: 2 })
await wait(60)
assistant(child, 1, 2, [{ type: 'text', text: 'The build is clean; two warnings about unused imports.' }], { inputTokens: 420, outputTokens: 22 })
child.append('step/end', { turn: 1, step: 2 })
child.append('turn/end', { turn: 1, reason: { kind: 'completed' } })

await wait(60)
toolResult(parent, 1, 2, 'c2', c2.seq, 'The build is clean; two warnings about unused imports.')
parent.append('step/end', { turn: 1, step: 2 })

parent.append('step/start', { turn: 1, step: 3 })
parent.append('assistant/chunk', { turn: 1, step: 3, chunk: { type: 'block-start', index: 0, blockType: 'text' } })
await wait(110)
assistant(parent, 1, 3, [{ type: 'text', text: 'Build is green. Two unused-import warnings are worth cleaning up.' }], { inputTokens: 1600, outputTokens: 30 })
parent.append('step/end', { turn: 1, step: 3 })
parent.append('turn/end', { turn: 1, reason: { kind: 'completed' } })

await exporter.shutdown(20_000)

const root = spans.find(span => span.parentSpanId === undefined)
console.log(`\ntrace ${root.traceId}  (${spans.length} spans)`)
for (const span of spans) {
  const type = span.attributes['langfuse.observation.type']
  console.log(`  ${String(type).padEnd(10)} ${span.name}`)
}
console.log(`\nOpen ${baseUrl} -> your project -> Traces, environment "development".`)
process.exit(0)
