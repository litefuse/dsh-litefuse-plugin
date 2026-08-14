/**
 * The trace assembler: DeepSeek Harness session events in, Litefuse spans out.
 *
 * One user turn becomes one trace whose root is an `agent` observation; every
 * model call and every tool execution mounts flat under it, sharing one
 * container-local step counter. A subagent's session mounts as a nested
 * `agent` container under the delegation tool span, with its own numbering
 * restarted at #1 — the one place the tree gains depth, because it is the one
 * place the execution really is nested.
 *
 * Spans are written exactly once, when the observation ends: OTel spans are
 * immutable, so an in-flight step is deliberately not visible until it closes.
 * Trace-level attributes ride on every span of a turn, which is what makes the
 * trace appear as soon as the first observation completes rather than at turn
 * end.
 *
 * @module dsh-litefuse/trace
 */

import type { Message, TokenUsage } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-compaction'
import {
  assistantOutput,
  reasoningChars,
  serializeValue,
  toolCalls,
  usageDetails,
  visibleText,
  type SerializedValue,
} from './content.js'
import { generationObservationName, toolObservationName, traceName } from './naming.js'
import { newSpanId, newTraceId, type LitefuseSpan, type SpanAttributeValue } from './otlp.js'

/** How much of a model request one generation observation carries as its input. */
export type RequestInputScope = 'full' | 'delta' | 'none'

/** Deployment-chosen assembly behavior. */
export interface AssemblerOptions {
  /** Display name in the trace title, e.g. `DeepSeek Harness — Turn 3`. */
  readonly agentName: string
  /** Litefuse tracing environment for every span. */
  readonly environment: string
  /** Trace-level `user.id`. */
  readonly userId: string
  /** Extra trace tags added beside the agent tag and `model:<name>`. */
  readonly tags: readonly string[]
  /** Character budget for one `input`/`output` attribute. */
  readonly maxValueChars: number
  /** How much of the request a generation records. */
  readonly requestInput: RequestInputScope
  /** Tool names treated as delegation when binding a child session to a call. */
  readonly delegationTools: readonly string[]
  /** Optional agent version recorded as the trace release. */
  readonly release?: string
}

/** One model call in flight, from its step opening to its assembled message. */
interface GenerationState {
  spanId: string
  startMillis: number
  stepIndex: number
  /** First streamed chunk, i.e. time to first token. */
  completionStartMillis?: number
}

/** One tool execution in flight, from its recorded call to its result. */
interface ToolState {
  spanId: string
  startMillis: number
  stepIndex: number
  /** Step index of the generation that requested this call. */
  planStep: number | undefined
  name: string
  /** Parsed arguments, or the raw string when the model produced invalid JSON. */
  args: unknown
  /** Subagent sessions that mounted their container under this call. */
  children: SessionState[]
}

/**
 * One numbering and parenting scope: a turn's root `agent` span, or a
 * subagent container mounted under a delegation tool span. Both own a step
 * counter, an input, and the observations still open inside them.
 */
interface Scope {
  kind: 'turn' | 'container'
  traceId: string
  /** This scope's own span id — the parent of every observation inside it. */
  spanId: string
  startMillis: number
  /** Last observed activity, used as the scope's end when it closes. */
  endMillis: number
  /** The session's own turn number; a container keeps the child's latest. */
  turn: number
  stepIndex: number
  apiCalls: number
  toolCalls: number
  input: string
  /** Whether {@link input} came from a direct human prompt rather than injected context. */
  inputFromUser: boolean
  /** Final assistant text, i.e. the trace or container output. */
  output: string
  /** Open generations by loop step number. */
  generations: Map<number, GenerationState>
  /** Step index of the generation that requested each loop step's tool calls. */
  planSteps: Map<number, number>
  /** Open tool executions by call id. */
  tools: Map<string, ToolState>
  /** Messages appended since the last generation closed, for `delta` request input. */
  pendingInput: Message[]
  /** Running token totals for this scope and every container beneath it. */
  usage: ScopeUsage
  /** Delegation tool span this container mounts under; absent on a turn scope. */
  toolSpanId?: string
}

/**
 * Tokens accumulated across a scope, including the containers nested under it.
 *
 * These totals ride the `agent` span's METADATA, never its `usage_details`:
 * Litefuse derives a trace's cost by summing observations, so a container that
 * also declared its children's tokens would double the bill. The metadata form
 * answers "how big was this turn" without touching that arithmetic.
 */
interface ScopeUsage {
  input: number
  output: number
  cacheRead: number
  cacheWrite: number
  reasoning: number
  /** Model calls whose accounting is included above. */
  counted: number
}

/** Per-session assembly state, alive as long as the session is. */
interface SessionState {
  id: string
  parentSessionId: string | undefined
  cwd: string | undefined
  provider: string | undefined
  model: string | undefined
  contextWindow: number | undefined
  modelParameters: Record<string, unknown> | undefined
  scope: Scope | undefined
  /**
   * Set once the session leaves the store while its container is still open.
   * A delegated run is disposed BEFORE the parent records the delegation's
   * `tool/result`, so the child's departure is the normal end of a successful
   * run, not an interruption — the parent closes the container.
   */
  detached: boolean
  /** Level the container closes with, from the child's own last `turn/end`. */
  containerLevel: ObservationLevel | undefined
  /** Status message paired with {@link containerLevel}. */
  containerStatus: string | undefined
}

/** Litefuse observation levels; `DEFAULT` is left implicit by omitting the attribute. */
type ObservationLevel = 'WARNING' | 'ERROR'

/** Flat attribute map for one span before OTLP encoding. */
type Attributes = Record<string, SpanAttributeValue | undefined>

/** Sink the assembler writes finished spans to. */
export type SpanSink = (span: LitefuseSpan) => void

/**
 * Rebuilds Litefuse traces from a session's committed event stream.
 *
 * Every method is synchronous and self-contained: the assembler is driven from
 * the session firehose, so it may never await, and a session it has not seen
 * before is adopted on its first event rather than requiring a creation hook.
 */
export class TraceAssembler {
  private readonly sessions = new Map<string, SessionState>()

  /**
   * @param options - deployment-chosen assembly behavior.
   * @param emit - receives every finished span, in completion order.
   */
  constructor(
    private readonly options: AssemblerOptions,
    private readonly emit: SpanSink,
  ) {}

  /**
   * Fold one committed session event into the trace under assembly.
   * @param session - the session whose log grew.
   * @param event - the appended event, exactly as recorded.
   */
  record(session: Session, event: SessionEvent): void {
    const state = this.stateOf(session)
    switch (event.type) {
      case 'turn/start':
        this.onTurnStart(state, event.data.turn, event.time)
        return
      case 'user/message':
        this.onUserMessage(state, session, event)
        return
      case 'request/header': {
        const { config } = event.data.header
        state.provider = config.provider
        state.model = config.model
        state.modelParameters = modelParametersOf({ ...config })
        return
      }
      case 'request/context':
        state.provider = event.data.provider
        state.model = event.data.model
        state.contextWindow = event.data.contextWindow
        return
      case 'step/start':
        this.onStepStart(state, event.data.step, event.time)
        return
      case 'assistant/chunk':
        this.onChunk(state, event.data.step, event.time)
        return
      case 'assistant/message':
        this.onAssistantMessage(state, session, event.data.step, event.data.message.content, event.data.usage, event.time)
        return
      case 'tool/call':
        this.onToolCall(state, event.data.step, event.data.callId, event.data.name, event.data.arguments, event.time)
        return
      case 'tool/result':
        this.onToolResult(state, session, event)
        return
      case 'turn/end':
        this.onTurnEnd(state, event.data.reason, event.time)
        return
      case 'compaction/end':
        this.onCompaction(state, event.data.error, event.time)
        return
      default:
        // Merge-extensible vocabulary: an event type this integration does not
        // model — including one added by a plugin it never heard of — carries
        // no observation of its own.
        return
    }
  }

  /**
   * Close everything still open for one session, e.g. at its disposal or at
   * plugin teardown. Open observations close as warnings: an unfinished step is
   * a faithful record of a turn that was interrupted, not a collection error.
   * @param sessionId - the session leaving the store.
   * @param endMillis - the moment to close open observations at.
   */
  closeSession(sessionId: string, endMillis: number): void {
    const state = this.sessions.get(sessionId)
    if (state === undefined) return
    this.sessions.delete(sessionId)
    if (state.scope === undefined) return
    if (state.scope.kind === 'container') {
      // A delegated run is disposed BEFORE its parent records the delegation's
      // `tool/result`, so leaving the store is how a SUCCESSFUL child ends.
      // The parent owns this container and closes it with the child's own
      // outcome; closing it here would report every delegation as interrupted.
      state.detached = true
      return
    }
    this.closeScope(state, state.scope, endMillis, 'WARNING', 'session ended before the turn completed')
  }

  /**
   * Close every session still under assembly, in insertion order.
   * @param endMillis - the moment to close open observations at.
   */
  closeAll(endMillis: number): void {
    for (const id of [...this.sessions.keys()]) this.closeSession(id, endMillis)
  }

  /** Adopt a session on first sight, recording the header facts spans need. */
  private stateOf(session: Session): SessionState {
    const id = String(session.id)
    const existing = this.sessions.get(id)
    if (existing !== undefined) return existing
    const state: SessionState = {
      id,
      parentSessionId: session.header.parentSession === undefined ? undefined : String(session.header.parentSession),
      cwd: session.header.cwd,
      provider: undefined,
      model: undefined,
      contextWindow: undefined,
      modelParameters: undefined,
      scope: undefined,
      detached: false,
      containerLevel: undefined,
      containerStatus: undefined,
    }
    this.sessions.set(id, state)
    return state
  }

  /**
   * Open the turn's scope. A session already inside a subagent container keeps
   * that container across its turns — the container belongs to the delegation,
   * not to one turn — and only records the newer turn number.
   */
  private onTurnStart(state: SessionState, turn: number, timeMillis: number): void {
    if (state.scope?.kind === 'container') {
      state.scope.turn = turn
      state.scope.endMillis = timeMillis
      return
    }
    if (state.scope !== undefined) {
      // Defensive: a turn opened without its predecessor closing means the
      // previous turn/end never reached this process.
      this.closeScope(state, state.scope, timeMillis, 'WARNING', 'turn superseded before it ended')
    }
    state.scope = this.bindContainer(state, turn, timeMillis) ?? {
      kind: 'turn',
      traceId: newTraceId(),
      spanId: newSpanId(),
      startMillis: timeMillis,
      endMillis: timeMillis,
      turn,
      stepIndex: 0,
      apiCalls: 0,
      toolCalls: 0,
      input: '',
      inputFromUser: false,
      output: '',
      generations: new Map(),
      planSteps: new Map(),
      tools: new Map(),
      pendingInput: [],
      usage: emptyUsage(),
    }
  }

  /**
   * Mount a subagent session's run under the parent's in-flight delegation
   * call, joining the parent's trace. Preference goes to a call whose tool name
   * is configured as delegation; otherwise the most recently started in-flight
   * call wins, because that is the call whose execution created this session.
   * @returns the container scope, or `undefined` when this session is not a bound child.
   */
  private bindContainer(state: SessionState, turn: number, timeMillis: number): Scope | undefined {
    if (state.parentSessionId === undefined) return undefined
    const parent = this.sessions.get(state.parentSessionId)
    const parentScope = parent?.scope
    if (parentScope === undefined) return undefined
    const inFlight = [...parentScope.tools.values()]
    if (inFlight.length === 0) return undefined
    const delegations = inFlight.filter(tool => this.options.delegationTools.includes(tool.name))
    const pool = delegations.length > 0 ? delegations : inFlight
    let chosen = pool[0]
    /* v8 ignore next -- pool is non-empty, so the seed is always defined */
    if (chosen === undefined) return undefined
    for (const tool of pool) {
      if (tool.startMillis >= chosen.startMillis) chosen = tool
    }
    chosen.children.push(state)
    return {
      kind: 'container',
      traceId: parentScope.traceId,
      spanId: newSpanId(),
      toolSpanId: chosen.spanId,
      startMillis: timeMillis,
      endMillis: timeMillis,
      turn,
      stepIndex: 0,
      apiCalls: 0,
      toolCalls: 0,
      input: '',
      inputFromUser: false,
      output: '',
      generations: new Map(),
      planSteps: new Map(),
      tools: new Map(),
      pendingInput: [],
      usage: emptyUsage(),
    }
  }

  /**
   * Record the scope's input. A direct human prompt outranks injected context:
   * a turn that begins with a file-change notice still shows the prompt the
   * person actually wrote once it arrives.
   */
  private onUserMessage(state: SessionState, session: Session, event: Extract<SessionEvent, { type: 'user/message' }>): void {
    const scope = state.scope
    if (scope === undefined) return
    scope.endMillis = event.time
    this.stageInput(scope, session, event)
    const fromUser = event.data.source.kind === 'user'
    if (!fromUser && scope.input.length > 0) return
    if (scope.inputFromUser && !fromUser) return
    const text = visibleText(event.data.content)
    if (text.length === 0) return
    scope.input = text
    scope.inputFromUser ||= fromUser
  }

  /** Open one model call's generation, taking the next step number in this scope. */
  private onStepStart(state: SessionState, step: number, timeMillis: number): void {
    const scope = state.scope
    if (scope === undefined) return
    scope.endMillis = timeMillis
    scope.apiCalls += 1
    const stepIndex = ++scope.stepIndex
    scope.generations.set(step, { spanId: newSpanId(), startMillis: timeMillis, stepIndex })
    scope.planSteps.set(step, stepIndex)
  }

  /** Stamp time-to-first-token from the step's first streamed chunk. */
  private onChunk(state: SessionState, step: number, timeMillis: number): void {
    const generation = state.scope?.generations.get(step)
    if (generation === undefined || generation.completionStartMillis !== undefined) return
    generation.completionStartMillis = timeMillis
  }

  /** Close one model call: name it by what the model did, attach usage, ship it. */
  private onAssistantMessage(
    state: SessionState,
    session: Session,
    step: number,
    content: readonly Message['content'][number][],
    usage: TokenUsage | undefined,
    timeMillis: number,
  ): void {
    const scope = state.scope
    if (scope === undefined) return
    const generation = scope.generations.get(step)
    if (generation === undefined) return
    scope.generations.delete(step)
    scope.endMillis = timeMillis
    const calls = toolCalls(content)
    const text = visibleText(content)
    const thinkingChars = reasoningChars(content)
    const name = generationObservationName(
      { toolCallCount: calls.length, hasText: text.length > 0, hasReasoning: thinkingChars > 0 },
      generation.stepIndex,
      scope.kind === 'container',
    )
    if (calls.length === 0 && text.length > 0) scope.output = text
    addUsage(scope.usage, usage)
    const input = this.requestInput(scope, session)
    const output = serializeValue(assistantOutput(content), this.options.maxValueChars)
    scope.pendingInput = []
    this.emit({
      traceId: scope.traceId,
      spanId: generation.spanId,
      parentSpanId: scope.spanId,
      name,
      startTimeMillis: generation.startMillis,
      endTimeMillis: timeMillis,
      attributes: {
        ...this.commonAttributes(state, scope),
        'langfuse.observation.type': 'generation',
        'langfuse.observation.model.name': state.model,
        // `model.parameters`, matching `model.name`: the underscore spelling
        // some integrations use matches no server constant and is dropped whole.
        'langfuse.observation.model.parameters': jsonOrUndefined(state.modelParameters),
        // A bare ISO string: the server treats a JSON-quoted timestamp as a
        // legacy quirk it parses defensively, not as the expected form.
        'langfuse.observation.completion_start_time': generation.completionStartMillis === undefined
          ? undefined
          : new Date(generation.completionStartMillis).toISOString(),
        'langfuse.observation.input': input?.text,
        'langfuse.observation.output': output.text.length === 0 ? undefined : output.text,
        'langfuse.observation.usage_details': jsonOrUndefined(usageDetails(usage)),
        ...metadata(OBSERVATION_METADATA, {
          turn_number: scope.turn,
          step_index: generation.stepIndex,
          provider: state.provider,
          api_duration_ms: timeMillis - generation.startMillis,
          time_to_first_token_ms: generation.completionStartMillis === undefined
            ? undefined
            : generation.completionStartMillis - generation.startMillis,
          tool_call_count: calls.length === 0 ? undefined : calls.length,
          thinking_chars: thinkingChars === 0 ? undefined : thinkingChars,
          reasoning_tokens: usage?.reasoningTokens,
          context_window: state.contextWindow,
          input_scope: input === undefined ? undefined : this.options.requestInput,
          input_truncated: input?.truncated === true ? true : undefined,
          input_orig_len: input?.truncated === true ? input.originalLength : undefined,
          output_truncated: output.truncated ? true : undefined,
          output_orig_len: output.truncated ? output.originalLength : undefined,
        }),
      },
    })
  }

  /** Open one tool execution, taking the next step number in this scope. */
  private onToolCall(
    state: SessionState,
    step: number,
    callId: string,
    name: string,
    rawArguments: string,
    timeMillis: number,
  ): void {
    const scope = state.scope
    if (scope === undefined) return
    scope.endMillis = timeMillis
    scope.toolCalls += 1
    scope.tools.set(callId, {
      spanId: newSpanId(),
      startMillis: timeMillis,
      stepIndex: ++scope.stepIndex,
      planStep: scope.planSteps.get(step),
      name,
      args: parseArguments(rawArguments),
      children: [],
    })
  }

  /** Close one tool execution, first closing any subagent container it hosted. */
  private onToolResult(state: SessionState, session: Session, event: Extract<SessionEvent, { type: 'tool/result' }>): void {
    const scope = state.scope
    if (scope === undefined) return
    scope.endMillis = event.time
    this.stageInput(scope, session, event)
    const block = event.data.message.content[0]
    const callId = String(block.toolCallId)
    const tool = scope.tools.get(callId)
    if (tool === undefined) return
    scope.tools.delete(callId)
    for (const child of tool.children) {
      const container = child.scope
      if (container?.kind !== 'container') continue
      mergeUsage(scope.usage, container.usage)
      this.closeScope(child, container, container.endMillis, child.containerLevel, child.containerStatus)
    }
    const isError = block.isError === true
    const output = serializeValue(visibleText(block.content), this.options.maxValueChars)
    const input = serializeValue(tool.args, this.options.maxValueChars)
    this.emit({
      traceId: scope.traceId,
      spanId: tool.spanId,
      parentSpanId: scope.spanId,
      name: toolObservationName(tool.name, tool.args, tool.stepIndex, tool.children.length),
      startTimeMillis: tool.startMillis,
      endTimeMillis: event.time,
      attributes: {
        ...this.commonAttributes(state, scope),
        'langfuse.observation.type': 'tool',
        'langfuse.observation.input': input.text,
        'langfuse.observation.output': output.text.length === 0 ? undefined : output.text,
        'langfuse.observation.level': isError ? 'ERROR' : undefined,
        'langfuse.observation.status_message': isError ? output.text.slice(0, 500) : undefined,
        ...metadata(OBSERVATION_METADATA, {
          tool_name: tool.name,
          tool_call_id: callId,
          step_index: tool.stepIndex,
          plan_step: tool.planStep,
          turn_number: scope.turn,
          duration_ms: event.time - tool.startMillis,
          is_error: isError ? true : undefined,
          error_code: event.data.error?.code,
          error_name: event.data.error?.name,
          subagent_count: tool.children.length === 0 ? undefined : tool.children.length,
          input_truncated: input.truncated ? true : undefined,
          input_orig_len: input.truncated ? input.originalLength : undefined,
          output_truncated: output.truncated ? true : undefined,
          output_orig_len: output.truncated ? output.originalLength : undefined,
        }),
      },
    })
  }

  /** Close a turn scope; a subagent container outlives its child's turns. */
  private onTurnEnd(state: SessionState, reason: { kind: string }, timeMillis: number): void {
    const scope = state.scope
    if (scope === undefined) return
    scope.endMillis = timeMillis
    if (scope.kind === 'container') {
      // The container closes later, with its parent's tool result; remember the
      // child's own verdict so a failed delegation is still reported as one.
      const [level, status] = outcomeOf(reason, scope.output)
      state.containerLevel = level
      state.containerStatus = status
      return
    }
    const failed = reason.kind === 'error'
    const level: ObservationLevel | undefined = failed
      ? 'ERROR'
      : scope.output.length === 0 ? 'WARNING' : undefined
    const statusMessage = failed
      ? `turn ended with ${reason.kind}`
      : scope.output.length === 0 ? 'turn ended without a final text response' : undefined
    this.closeScope(state, scope, timeMillis, level, statusMessage, reason.kind)
  }

  /** Record a context compaction as a timeline event: it explains the next call's token drop. */
  private onCompaction(state: SessionState, error: string | undefined, timeMillis: number): void {
    const scope = state.scope
    if (scope === undefined) return
    scope.endMillis = timeMillis
    this.emit({
      traceId: scope.traceId,
      spanId: newSpanId(),
      parentSpanId: scope.spanId,
      name: 'context compaction',
      startTimeMillis: timeMillis,
      endTimeMillis: timeMillis,
      attributes: {
        ...this.commonAttributes(state, scope),
        'langfuse.observation.type': 'event',
        'langfuse.observation.level': error === undefined ? undefined : 'ERROR',
        'langfuse.observation.status_message': error,
        ...metadata(OBSERVATION_METADATA, { turn_number: scope.turn }),
      },
    })
  }

  /**
   * Emit one scope's own span and release it. Observations still open inside
   * close first, as warnings, so the tree never references a span that was
   * never written.
   */
  private closeScope(
    state: SessionState,
    scope: Scope,
    endMillis: number,
    level: ObservationLevel | undefined,
    statusMessage: string | undefined,
    endReason?: string,
  ): void {
    if (state.scope === scope) state.scope = undefined
    const closeAt = Math.max(endMillis, scope.endMillis)
    for (const [step, generation] of scope.generations) {
      scope.generations.delete(step)
      this.emit({
        traceId: scope.traceId,
        spanId: generation.spanId,
        parentSpanId: scope.spanId,
        name: `generation #${generation.stepIndex}`,
        startTimeMillis: generation.startMillis,
        endTimeMillis: closeAt,
        attributes: {
          ...this.commonAttributes(state, scope),
          'langfuse.observation.type': 'generation',
          'langfuse.observation.model.name': state.model,
          'langfuse.observation.level': 'WARNING',
          'langfuse.observation.status_message': 'turn ended before the model call completed',
          ...metadata(OBSERVATION_METADATA, { turn_number: scope.turn, step_index: generation.stepIndex }),
        },
      })
    }
    for (const [callId, tool] of scope.tools) {
      scope.tools.delete(callId)
      // A delegation that never reported a result still owns live containers;
      // close them here so no span references a parent that was never written.
      for (const child of tool.children) {
        const container = child.scope
        if (container?.kind !== 'container') continue
        mergeUsage(scope.usage, container.usage)
        this.closeScope(child, container, closeAt, 'WARNING', 'delegation ended before the subagent returned')
      }
      this.emit({
        traceId: scope.traceId,
        spanId: tool.spanId,
        parentSpanId: scope.spanId,
        name: toolObservationName(tool.name, tool.args, tool.stepIndex, tool.children.length),
        startTimeMillis: tool.startMillis,
        endTimeMillis: closeAt,
        attributes: {
          ...this.commonAttributes(state, scope),
          'langfuse.observation.type': 'tool',
          'langfuse.observation.input': serializeValue(tool.args, this.options.maxValueChars).text,
          'langfuse.observation.level': 'WARNING',
          'langfuse.observation.status_message': 'turn ended before the tool completed',
          ...metadata(OBSERVATION_METADATA, {
            tool_name: tool.name,
            tool_call_id: callId,
            step_index: tool.stepIndex,
            plan_step: tool.planStep,
            turn_number: scope.turn,
          }),
        },
      })
    }
    const isContainer = scope.kind === 'container'
    const summary: Record<string, unknown> = {
      turn_number: scope.turn,
      session_id: state.id,
      parent_session_id: state.parentSessionId,
      cwd: state.cwd,
      provider: state.provider,
      model: state.model,
      api_calls: scope.apiCalls,
      tool_calls: scope.toolCalls,
      steps: scope.stepIndex,
      duration_ms: closeAt - scope.startMillis,
      context_window: state.contextWindow,
      end_reason: endReason,
      subagent: isContainer ? true : undefined,
      // Token totals for this scope AND every container beneath it. They live
      // in metadata, never in `usage_details`: Litefuse sums observations to
      // price a trace, so declaring them again here would double the bill.
      accounted_generations: scope.usage.counted === 0 ? undefined : scope.usage.counted,
      input_tokens: scope.usage.counted === 0 ? undefined : scope.usage.input,
      output_tokens: scope.usage.counted === 0 ? undefined : scope.usage.output,
      cache_read_tokens: scope.usage.cacheRead === 0 ? undefined : scope.usage.cacheRead,
      cache_write_tokens: scope.usage.cacheWrite === 0 ? undefined : scope.usage.cacheWrite,
      reasoning_tokens: scope.usage.reasoning === 0 ? undefined : scope.usage.reasoning,
      total_tokens: scope.usage.counted === 0
        ? undefined
        : scope.usage.input + scope.usage.output + scope.usage.cacheRead + scope.usage.cacheWrite,
    }
    const output = serializeValue(scope.output, this.options.maxValueChars)
    this.emit({
      traceId: scope.traceId,
      spanId: scope.spanId,
      ...isContainer && scope.toolSpanId !== undefined ? { parentSpanId: scope.toolSpanId } : {},
      name: isContainer ? 'subagent' : traceName(this.options.agentName, scope.turn),
      startTimeMillis: scope.startMillis,
      endTimeMillis: closeAt,
      attributes: {
        ...this.commonAttributes(state, scope),
        'langfuse.observation.type': 'agent',
        'langfuse.observation.input': scope.input.length === 0 ? undefined : scope.input,
        'langfuse.observation.output': output.text.length === 0 ? undefined : output.text,
        'langfuse.observation.level': level,
        'langfuse.observation.status_message': statusMessage,
        ...metadata(OBSERVATION_METADATA, summary),
        // A container owns no trace-level fields: the trace is the parent's.
        'langfuse.trace.output': isContainer || output.text.length === 0 ? undefined : output.text,
        ...isContainer ? {} : metadata(TRACE_METADATA, summary),
      },
    })
  }

  /**
   * Attributes every span of a scope carries: the tracing environment plus,
   * outside a subagent container, the trace header. Repeating the header on
   * each span is what makes a trace queryable as soon as its first observation
   * completes, since the root span is not written until the turn ends.
   */
  private commonAttributes(state: SessionState, scope: Scope): Attributes {
    if (scope.kind === 'container') {
      return { 'langfuse.environment': this.options.environment }
    }
    const tags = [...this.options.tags]
    if (state.model !== undefined) tags.push(`model:${state.model}`)
    return {
      'langfuse.environment': this.options.environment,
      'langfuse.trace.name': traceName(this.options.agentName, scope.turn),
      'langfuse.trace.input': scope.input.length === 0 ? undefined : scope.input,
      'langfuse.trace.tags': tags,
      'langfuse.release': this.options.release,
      'session.id': state.id,
      'user.id': this.options.userId,
    }
  }

  /** Stage one derived model message for `delta` request input. */
  private stageInput(scope: Scope, session: Session, event: SessionEvent): void {
    if (this.options.requestInput !== 'delta') return
    const message = session.deriveEventMessage(event)
    if (message !== null) scope.pendingInput.push(message)
  }

  /**
   * Build the generation's `input`: the whole request the adapter received, the
   * messages added since the previous call, or nothing.
   */
  private requestInput(scope: Scope, session: Session): SerializedValue | undefined {
    switch (this.options.requestInput) {
      case 'none':
        return undefined
      case 'delta':
        return scope.pendingInput.length === 0
          ? undefined
          : serializeValue(scope.pendingInput, this.options.maxValueChars)
      default: {
        const header = session.requestHeader()
        const history = session.deriveMessages()
        // The assembled message for this very step is already committed, and it
        // is the response — not part of what was sent.
        if (history[history.length - 1]?.role === 'assistant') history.pop()
        return serializeValue({
          ...header?.system === undefined ? {} : { system: header.system },
          ...header?.tools === undefined ? {} : { tools: header.tools.map(tool => tool.name) },
          messages: history,
        }, this.options.maxValueChars)
      }
    }
  }
}

/** Parse the model's raw argument JSON, keeping invalid JSON as its own text. */
function parseArguments(raw: string): unknown {
  if (raw.length === 0) return {}
  try {
    return JSON.parse(raw)
  } catch {
    // The model produced something that is not JSON; the raw text is still the
    // most faithful record of what it asked for.
    return raw
  }
}

/**
 * Report every knob the logged call configuration carries, minus the two
 * fields that are identity rather than parameters. Enumerating a fixed list
 * instead would silently drop whatever a provider plugin adds to the config,
 * and would report nothing at all for a deployment that leaves the common
 * sampling fields at their provider defaults.
 * @param config - the logged call configuration for the next request.
 * @returns the parameter map, or `undefined` when only identity was recorded.
 */
function modelParametersOf(config: Record<string, unknown>): Record<string, unknown> | undefined {
  const parameters: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(config)) {
    if (key === 'provider' || key === 'model' || value === undefined) continue
    parameters[key] = value
  }
  return Object.keys(parameters).length === 0 ? undefined : parameters
}

/**
 * Map one turn's end reason onto the level its `agent` span carries. A turn
 * that produced no final text is a faithful record of an interrupted run, not
 * a collection error, so it warns rather than fails.
 * @param reason - the durable `turn/end` reason.
 * @param output - the final assistant text, empty when the turn produced none.
 * @returns the level and status message, both `undefined` on a clean turn.
 */
function outcomeOf(
  reason: { kind: string },
  output: string,
): [ObservationLevel | undefined, string | undefined] {
  if (reason.kind === 'error') return ['ERROR', `turn ended with ${reason.kind}`]
  if (output.length === 0) return ['WARNING', 'turn ended without a final text response']
  return [undefined, undefined]
}

/** A scope's zeroed token accumulator. */
function emptyUsage(): ScopeUsage {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, reasoning: 0, counted: 0 }
}

/** Add one model call's accounting into a scope's running totals. */
function addUsage(target: ScopeUsage, usage: TokenUsage | undefined): void {
  if (usage === undefined) return
  target.input += usage.inputTokens
  target.output += usage.outputTokens
  target.cacheRead += usage.cacheReadTokens ?? 0
  target.cacheWrite += usage.cacheWriteTokens ?? 0
  target.reasoning += usage.reasoningTokens ?? 0
  target.counted += 1
}

/** Fold a closed container's totals into the scope that hosts it. */
function mergeUsage(target: ScopeUsage, source: ScopeUsage): void {
  target.input += source.input
  target.output += source.output
  target.cacheRead += source.cacheRead
  target.cacheWrite += source.cacheWrite
  target.reasoning += source.reasoning
  target.counted += source.counted
}

/**
 * Emit one metadata map BOTH ways: per-key attributes under the namespace, and
 * the whole map as one JSON string on the namespace itself.
 *
 * The uniform `agent_` prefix — never a per-agent namespace — is what lets a
 * single Litefuse dashboard query span every agent integration, and absent
 * fields are dropped rather than padded with nulls.
 *
 * Both forms are sent because the server has two readers. Its ingestion
 * processor merges them (`{...topLevelMetadata, ...langfuseMetadata}`), so
 * duplicate keys carrying identical values are harmless. The trace-detail UI
 * loads observation metadata through a separate path that appears to read only
 * the serialized form: sending per-key alone left that panel empty while the
 * public API returned every field. Until the two readers agree, sending both is
 * the only form that populates both.
 * @param namespace - `langfuse.observation.metadata` or `langfuse.trace.metadata`.
 * @param fields - metadata entries without their prefix.
 * @returns attributes ready to spread onto a span.
 */
function metadata(namespace: string, fields: Record<string, unknown>): Attributes {
  const attributes: Attributes = {}
  const serialized: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined) continue
    const prefixed = `agent_${key}`
    attributes[`${namespace}.${prefixed}`] = value as SpanAttributeValue
    serialized[prefixed] = value
  }
  if (Object.keys(serialized).length === 0) return attributes
  attributes[namespace] = JSON.stringify(serialized)
  return attributes
}

/** Observation-level metadata namespace. */
const OBSERVATION_METADATA = 'langfuse.observation.metadata'

/** Trace-level metadata namespace. */
const TRACE_METADATA = 'langfuse.trace.metadata'

/** Serialize a structured attribute, dropping it when there is nothing to say. */
function jsonOrUndefined(value: unknown): string | undefined {
  return value === undefined ? undefined : JSON.stringify(value)
}
