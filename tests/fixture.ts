/**
 * Test fixture: a real harness session store driven through scripted turns.
 *
 * The tests deliberately use the shipped `@deepseek-ai/dsh-session` Session
 * rather than a stand-in, so the assembler is exercised against the real
 * append validation, surface rules, and derived-history projection it reads in
 * production.
 */

import { Context } from '@deepseek-ai/cordis'
import SessionStore from '@deepseek-ai/dsh-session'
import type { Session, SessionId } from '@deepseek-ai/dsh-session'
import { CallId, MessageId } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, TokenUsage } from '@deepseek-ai/dsh-llm'
import type { LitefuseSpan } from '../src/otlp.js'

/** Brand a plain string as a tool call id, for readable test data. */
export const call = (id: string): CallId => CallId(id)

/** Append option every model-visible event needs. */
const APPEND = { surfaceOp: 'append' } as const

/**
 * Build a settled context with the real session store mounted.
 * @returns the context, ready for `ctx.sessions`.
 */
export async function createContext(): Promise<Context> {
  const ctx = new Context()
  ctx.plugin(SessionStore)
  await new Promise(resolve => setTimeout(resolve, 20))
  return ctx
}

/** Read every observation-metadata field off one emitted span, prefix stripped. */
export function metadataOf(span: LitefuseSpan): Record<string, unknown> {
  return metadataUnder(span, 'langfuse.observation.metadata')
}

/** Read every trace-metadata field off one emitted span, prefix stripped. */
export function traceMetadataOf(span: LitefuseSpan): Record<string, unknown> {
  return metadataUnder(span, 'langfuse.trace.metadata')
}

/** Collect the per-key metadata attributes under one namespace. */
function metadataUnder(span: LitefuseSpan, namespace: string): Record<string, unknown> {
  const fields: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(span.attributes)) {
    if (key.startsWith(`${namespace}.`)) fields[key.slice(namespace.length + 1)] = value
  }
  return fields
}

/** Read one Langfuse attribute as a string. */
export function attribute(span: LitefuseSpan, key: string): string | undefined {
  const value = span.attributes[key]
  return typeof value === 'string' ? value : undefined
}

/** Scripted writer over one live session; every method appends real events. */
export class TurnWriter {
  private step = 0

  /**
   * @param session - the live session to append to.
   * @param turn - the turn number these appends belong to.
   */
  constructor(readonly session: Session, readonly turn: number) {}

  /** Open the turn. */
  start(): this {
    this.session.append('turn/start', { turn: this.turn })
    return this
  }

  /** Append the human prompt that becomes the trace input. */
  prompt(text: string): this {
    this.session.append('user/message', {
      id: MessageId(`user-${this.turn}`),
      role: 'user',
      content: [{ type: 'text', text }],
      source: { kind: 'user' },
    }, APPEND)
    return this
  }

  /** Log the request header that supplies model identity and sampling knobs. */
  header(model = 'deepseek-chat', provider = 'deepseek'): this {
    this.session.append('request/header', {
      header: {
        config: { provider, model, temperature: 0.2, maxTokens: 4096 },
        system: 'You are a coding agent.',
        tools: [{ name: 'bash', description: 'Run a command.', parameters: {} }],
      },
      reason: 'initial',
    })
    return this
  }

  /** Open one model call and stream its first chunk. */
  openStep(withChunk = true): number {
    const step = ++this.step
    this.session.append('step/start', { turn: this.turn, step })
    if (withChunk) {
      this.session.append('assistant/chunk', {
        turn: this.turn,
        step,
        chunk: { type: 'block-start', index: 0, blockType: 'text' },
      })
    }
    return step
  }

  /** Close one model call with the blocks it produced. */
  assistant(step: number, content: ContentBlock[], usage?: TokenUsage): this {
    this.session.append('assistant/message', {
      turn: this.turn,
      step,
      message: {
        id: MessageId(`assistant-${this.turn}-${step}`),
        role: 'assistant',
        content,
        source: { kind: 'model', provider: 'deepseek', model: 'deepseek-chat' },
      },
      ...usage === undefined ? {} : { usage },
    }, APPEND)
    return this
  }

  /** Record a tool call entering dispatch. */
  toolCall(step: number, callId: string, name: string, args: unknown): number {
    const event = this.session.append('tool/call', {
      turn: this.turn,
      step,
      callId: CallId(callId),
      name,
      arguments: JSON.stringify(args),
    })
    return event.seq
  }

  /** Record a tool result, closing its span. */
  toolResult(step: number, callId: string, callSeq: number, text: string, isError = false): this {
    this.session.append('tool/result', {
      turn: this.turn,
      step,
      message: {
        id: MessageId(`tool-${callId}`),
        role: 'user',
        content: [{
          type: 'tool-result',
          toolCallId: CallId(callId),
          content: [{ type: 'text', text }],
          ...isError ? { isError: true } : {},
        }],
        source: { kind: 'tool', callId: CallId(callId) },
      },
    }, { ...APPEND, sourceEventSeqs: [callSeq] })
    return this
  }

  /** Close the loop step. */
  closeStep(step: number): this {
    this.session.append('step/end', { turn: this.turn, step })
    return this
  }

  /** Close the turn with the given reason. */
  end(reason: { kind: string } = { kind: 'completed' }): this {
    this.session.append('turn/end', { turn: this.turn, reason } as never)
    return this
  }
}

/**
 * Create a session, optionally as a subagent child of another.
 * @param ctx - the context owning the session store.
 * @param parent - the parent session id recorded in the child's header.
 * @returns the announced live session.
 */
export function createSession(ctx: Context, parent?: SessionId): Session {
  return ctx.sessions.create(undefined, parent === undefined
    ? { meta: { cwd: '/workspace' } }
    : { meta: { cwd: '/workspace', parentSession: parent, origin: 'subagent' } })
}
