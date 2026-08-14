/**
 * OTLP/HTTP JSON span encoding and the Litefuse trace transport.
 *
 * Litefuse ingests OpenTelemetry spans at `POST <baseUrl>/api/public/otel/v1/traces`
 * as OTLP/HTTP JSON authorized with HTTP Basic over the project's public and
 * secret key. This module owns the wire form — attribute typing, nanosecond
 * timestamps, trace and span id minting — and the send path: spans coalesce in
 * a delay-bounded buffer, each span is written exactly once (OTel spans are
 * immutable, so an observation ships when it ends and is never resent), and a
 * failed post is logged and dropped so reporting can never block the agent.
 *
 * The transport is hand-written rather than composed from the OTel JS SDK
 * because every span here is reconstructed from a durable session log with its
 * own past start and end timestamps and its own parent, which the SDK's
 * tracer-and-context API cannot express for spans whose parent outlives them in
 * a different session.
 *
 * @module @deepseek-ai/dsh-session-telemetry-litefuse/otlp
 */

import { randomBytes } from 'node:crypto'

/** One attribute value the OTLP JSON encoder can carry. */
export type SpanAttributeValue = string | number | boolean | readonly string[]

/** A span ready for the wire: absolute millisecond bounds and flat attributes. */
export interface LitefuseSpan {
  /** 32-hex trace id shared by every span of one user turn. */
  traceId: string
  /** 16-hex id unique to this span. */
  spanId: string
  /** Enclosing span's id; absent only on a trace's root span. */
  parentSpanId?: string
  /** Observation name, per the Litefuse agent-trace spec's naming rules. */
  name: string
  /** Wall-clock start in epoch milliseconds. */
  startTimeMillis: number
  /** Wall-clock end in epoch milliseconds. */
  endTimeMillis: number
  /** Langfuse observation and trace attributes; `undefined` entries are dropped (sparse metadata). */
  attributes: Readonly<Record<string, SpanAttributeValue | undefined>>
}

/** OTLP JSON `AnyValue` for the attribute types this package emits. */
type OtlpAnyValue =
  | { stringValue: string }
  | { intValue: number }
  | { doubleValue: number }
  | { boolValue: boolean }
  | { arrayValue: { values: { stringValue: string }[] } }

/** One OTLP JSON `KeyValue` entry. */
interface OtlpKeyValue {
  key: string
  value: OtlpAnyValue
}

/** Path appended to the configured base URL to reach the Litefuse OTLP trace ingest. */
export const LITEFUSE_TRACES_PATH = '/api/public/otel/v1/traces'

/**
 * Ingestion protocol this exporter speaks, declared per request.
 *
 * Litefuse gates its trace ingest on client capability and rejects anything
 * older with HTTP 400. The gate accepts either a first-party SDK new enough to
 * emit complete spans inline, or this header as the documented opt-in for a
 * custom OTel exporter. The claim is accurate here: every span is written once,
 * when the observation ends, with no create-then-update split.
 */
const INGESTION_VERSION = '4'

/** Span kind `INTERNAL`; every observation is in-process work of this harness. */
const SPAN_KIND_INTERNAL = 1

/**
 * Mint a fresh 32-hex OTLP trace id.
 * @returns the lowercase hex id of 16 random bytes.
 */
export function newTraceId(): string {
  return randomBytes(16).toString('hex')
}

/**
 * Mint a fresh 16-hex OTLP span id.
 * @returns the lowercase hex id of 8 random bytes.
 */
export function newSpanId(): string {
  return randomBytes(8).toString('hex')
}

/**
 * Render epoch milliseconds as the OTLP nanosecond timestamp string.
 * @param millis - epoch milliseconds, as carried by every session event.
 * @returns the nanosecond value as a decimal string.
 */
export function nanoTimestamp(millis: number): string {
  return `${Math.round(millis)}000000`
}

/**
 * Encode one attribute map into OTLP `KeyValue` entries, dropping absent
 * fields so the exported metadata stays sparse (no null padding).
 * @param attributes - flat attribute map; `undefined` values are omitted.
 * @returns the encoded entries in insertion order.
 */
function encodeAttributes(attributes: Readonly<Record<string, SpanAttributeValue | undefined>>): OtlpKeyValue[] {
  const encoded: OtlpKeyValue[] = []
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined) continue
    encoded.push({ key, value: encodeValue(value) })
  }
  return encoded
}

/**
 * Encode one attribute value into its OTLP `AnyValue` form.
 * @param value - the attribute value to encode.
 * @returns the matching `AnyValue` variant.
 */
function encodeValue(value: SpanAttributeValue): OtlpAnyValue {
  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map(entry => ({ stringValue: entry })) } }
  }
  switch (typeof value) {
    case 'number':
      return Number.isInteger(value) ? { intValue: value } : { doubleValue: value }
    case 'boolean':
      return { boolValue: value }
    default:
      return { stringValue: value as string }
  }
}

/** Identity of the exporting instrumentation, carried once per export batch. */
export interface ExporterIdentity {
  /** Resource attributes describing the emitting application (`service.name`, `service.version`, …). */
  readonly resource: Readonly<Record<string, SpanAttributeValue | undefined>>
  /** Instrumentation scope name, i.e. this package. */
  readonly scopeName: string
  /** Instrumentation scope version, i.e. this package's version. */
  readonly scopeVersion: string
}

/** Everything the exporter needs that a deployment chooses. */
export interface ExporterOptions {
  /** Litefuse base URL; the trace path is appended to it. */
  readonly baseUrl: string
  /** Resolve the project key pair for one send; `undefined` skips the send. */
  readonly credentials: () => Promise<{ publicKey: string; secretKey: string } | undefined>
  /** How long a buffered span may wait for company before the batch posts. */
  readonly exportDelayMillis: number
  /** Per-request deadline for one OTLP post. */
  readonly requestTimeoutMillis: number
  /** Report a send failure; the exporter never throws to its caller. */
  readonly onFailure: (message: string) => void
  /** Report a delivered batch, for verbose logging. */
  readonly onSuccess: (message: string) => void
}

/**
 * Buffers ended spans and posts them to one Litefuse project.
 *
 * Sends are fire-and-forget: {@link enqueue} never awaits, a rejected or
 * non-2xx post is reported through `onFailure` and dropped, and an unreachable
 * target costs nothing but the request deadline. {@link flush} exists so a turn
 * boundary and disposal can bound how long a finished trace waits.
 */
export class LitefuseSpanExporter {
  private readonly endpoint: string
  private readonly buffer: LitefuseSpan[] = []
  private readonly inflight = new Set<Promise<void>>()
  private timer: ReturnType<typeof setTimeout> | undefined
  private closed = false

  /**
   * @param options - deployment-chosen endpoint, credentials, and bounds.
   * @param identity - resource and instrumentation-scope identity for every batch.
   */
  constructor(
    private readonly options: ExporterOptions,
    private readonly identity: ExporterIdentity,
  ) {
    this.endpoint = `${options.baseUrl.replace(/\/+$/, '')}${LITEFUSE_TRACES_PATH}`
  }

  /**
   * Buffer one ended span for export. Spans that end in the same burst share a
   * post; the delay bound keeps a lone span from waiting for one.
   * @param span - the ended span; the exporter owns it after this call.
   */
  enqueue(span: LitefuseSpan): void {
    if (this.closed) return
    this.buffer.push(span)
    if (this.timer !== undefined) return
    this.timer = setTimeout(() => {
      this.timer = undefined
      this.post()
    }, this.options.exportDelayMillis)
    // A pending export batch must never keep the process alive on its own:
    // application teardown drains through shutdown(), not through this timer.
    this.timer.unref?.()
  }

  /**
   * Post whatever is buffered now and wait for every in-flight post, bounded by
   * `timeoutMillis`. Abandoning the wait cannot cancel a request already on the
   * wire; those spans may still arrive after the wait returns.
   * @param timeoutMillis - outer bound for the drain.
   * @returns resolves when the drain completes or the bound expires.
   */
  async flush(timeoutMillis: number): Promise<void> {
    if (this.timer !== undefined) {
      clearTimeout(this.timer)
      this.timer = undefined
    }
    this.post()
    if (this.inflight.size === 0) return
    let timer: ReturnType<typeof setTimeout> | undefined
    const deadline = new Promise<void>((resolve) => {
      timer = setTimeout(resolve, timeoutMillis)
      timer.unref?.()
    })
    try {
      await Promise.race([Promise.allSettled([...this.inflight]).then(() => undefined), deadline])
    } finally {
      /* v8 ignore else -- the Promise executor assigns timer synchronously before this race starts. */
      if (timer !== undefined) clearTimeout(timer)
    }
  }

  /**
   * Drain the exporter and refuse further spans. Later {@link enqueue} calls are
   * dropped, so a span produced during teardown cannot resurrect the timer.
   * @param timeoutMillis - outer bound for the final drain.
   * @returns resolves when the drain completes or the bound expires.
   */
  async shutdown(timeoutMillis: number): Promise<void> {
    await this.flush(timeoutMillis)
    this.closed = true
  }

  /** Take the buffer and start one post for it; a resolved-credential miss drops the batch. */
  private post(): void {
    if (this.buffer.length === 0) return
    const spans = this.buffer.splice(0, this.buffer.length)
    const body = this.encode(spans)
    const request = this.send(body, spans.length)
    this.inflight.add(request)
    void request.finally(() => this.inflight.delete(request))
  }

  /** Build the OTLP/HTTP JSON payload for one batch. */
  private encode(spans: readonly LitefuseSpan[]): string {
    return JSON.stringify({
      resourceSpans: [{
        resource: { attributes: encodeAttributes(this.identity.resource) },
        scopeSpans: [{
          scope: { name: this.identity.scopeName, version: this.identity.scopeVersion },
          spans: spans.map(span => ({
            traceId: span.traceId,
            spanId: span.spanId,
            ...span.parentSpanId === undefined ? {} : { parentSpanId: span.parentSpanId },
            name: span.name,
            kind: SPAN_KIND_INTERNAL,
            startTimeUnixNano: nanoTimestamp(span.startTimeMillis),
            endTimeUnixNano: nanoTimestamp(span.endTimeMillis),
            attributes: encodeAttributes(span.attributes),
          })),
        }],
      }],
    })
  }

  /**
   * Resolve credentials, post one batch, and contain every failure. The
   * credential read happens per batch, so a rotated key reaches the next
   * export without reloading this plugin.
   */
  private async send(body: string, spanCount: number): Promise<void> {
    try {
      const keys = await this.options.credentials()
      if (keys === undefined) {
        this.options.onFailure(`dropped ${spanCount} span(s): Litefuse credentials are not configured`)
        return
      }
      const authorization = Buffer.from(`${keys.publicKey}:${keys.secretKey}`).toString('base64')
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'authorization': `Basic ${authorization}`,
          'content-type': 'application/json',
          'x-langfuse-ingestion-version': INGESTION_VERSION,
        },
        body,
        signal: AbortSignal.timeout(this.options.requestTimeoutMillis),
      })
      if (response.ok) {
        this.options.onSuccess(`sent ${spanCount} span(s) -> ${this.endpoint} HTTP ${response.status}`)
        return
      }
      const detail = await response.text().catch(() => '')
      this.options.onFailure(`export failed: HTTP ${response.status} ${detail.slice(0, 300)}`)
    } catch (error) {
      this.options.onFailure(`export failed: ${String(error)}`)
    }
  }
}
