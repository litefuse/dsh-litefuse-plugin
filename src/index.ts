/**
 * Litefuse observability for DeepSeek Harness.
 *
 * A Cordis plugin that subscribes to the session firehose and rebuilds each
 * user turn as one Litefuse trace: an `agent` root, one `generation` per model
 * call, one `tool` per tool execution, and a nested container for every
 * subagent the turn delegated to. It touches nothing else in the harness — no
 * service is registered, no request is altered, and every failure is contained
 * and logged, so an unreachable Litefuse project costs the agent nothing.
 *
 * Credentials are read as references, never stored in configuration: the
 * public and secret key come from the harness credential store when one is
 * mounted, otherwise from the process environment that `~/.dsh/.env` feeds.
 *
 * @module dsh-litefuse-plugin
 */

import { createRequire } from 'node:module'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'
import type {} from '@deepseek-ai/dsh-session'
import { createLog, harnessHome, type LitefuseLog } from './log.js'
import { LitefuseSpanExporter, type LitefuseSpan } from './otlp.js'
import { OBSERVATION_METADATA, TraceAssembler, type RequestInputScope } from './trace.js'

const { version } = createRequire(import.meta.url)('../package.json') as { version: string }

/** Cordis plugin name. */
export const name = 'dsh-litefuse-plugin'

/**
 * Instrumentation scope reported with every batch: this package, named
 * accurately.
 *
 * The ingest copies a span's whole raw attribute map into each observation's
 * metadata as `attributes` unless the scope name begins with `langfuse-sdk`.
 * Claiming that prefix suppresses the copy — one release did — but the scope is
 * an identity field and this is not a Langfuse SDK, so the claim bought a
 * cleaner payload by misreporting who sent it.
 *
 * The copy costs nothing but noise: every attribute in it is already a
 * first-class field, so no data is lost or wrong, only duplicated in a form
 * that re-encodes the JSON strings `model.parameters` and `usage_details` are
 * required to be. The accurate fix belongs on the ingest side, which already
 * receives `x-langfuse-ingestion-version: 4` from this exporter — the
 * documented signal that a client sends complete spans inline, and a claim that
 * is true of any custom exporter making it rather than of one name prefix.
 */
const SCOPE_NAME = name

/** The session store must exist before this plugin has anything to observe. */
export const inject = ['sessions']

/** Litefuse Cloud, the endpoint the project's own documentation issues keys for. */
export const DEFAULT_BASE_URL = 'https://litefuse.cloud'

/** Environment references the Litefuse spec assigns to the project key pair. */
const PUBLIC_KEY_REFS = ['LITEFUSE_PUBLIC_KEY', 'LANGFUSE_PUBLIC_KEY'] as const

/** Secret-key references; `LITEFUSE_*` wins and `LANGFUSE_*` is the ecosystem fallback. */
const SECRET_KEY_REFS = ['LITEFUSE_SECRET_KEY', 'LANGFUSE_SECRET_KEY'] as const

/** Delegation tools shipped by the harness, whose calls host a subagent container. */
const DEFAULT_DELEGATION_TOOLS = ['subagent', 'subagent_fork'] as const

/** Display name in the trace title when a deployment names no other. */
const DEFAULT_AGENT_NAME = 'DeepSeek Harness'

/** Litefuse tracing environment when a deployment names no other. */
const DEFAULT_ENVIRONMENT = 'production'

/** Trace tag identifying this integration, beside the per-trace `model:<name>`. */
const DEFAULT_TAG = 'dsh'

/** Character budget for one observation input or output, from the Litefuse trace spec. */
const DEFAULT_MAX_VALUE_CHARS = 1_000_000

/** How long an ended span waits for others to end before its batch posts. */
const DEFAULT_EXPORT_DELAY_MILLIS = 1_000

/** Per-request deadline for one OTLP post. */
const DEFAULT_REQUEST_TIMEOUT_MILLIS = 10_000

/** Outer bound on the drain performed while the plugin is disposed. */
const DEFAULT_SHUTDOWN_TIMEOUT_MILLIS = 3_000

/** Log file name inside the harness home. */
const LOG_FILE_NAME = 'litefuse.log'

/** Environment switch for verbose logging, matching the other Litefuse integrations. */
const DEBUG_ENV = 'DSH_LITEFUSE_DEBUG'

/** Plugin configuration. */
export interface Config {
  /** Whether to export at all; `false` keeps the plugin mounted and silent. */
  enabled?: boolean
  /** Litefuse base URL; the OTLP trace path is appended to it. */
  baseUrl?: string
  /** Environment-variable name holding the project's public key. */
  publicKeyEnv?: string
  /** Environment-variable name holding the project's secret key. */
  secretKeyEnv?: string
  /** Litefuse tracing environment recorded on every span. */
  environment?: string
  /** Display name used in the trace title. */
  agentName?: string
  /** Trace-level `user.id`; defaults to the OS user name. */
  userId?: string
  /** Extra trace tags added beside the agent tag and `model:<name>`. */
  tags?: string[]
  /** Optional release identifier recorded on every trace. */
  release?: string
  /** How much of a model request a generation records as its input. */
  requestInput?: RequestInputScope
  /** Character budget for one observation input or output. */
  maxValueChars?: number
  /** Tool names whose in-flight call hosts a subagent container. */
  delegationTools?: string[]
  /** How long an ended span waits for company before its batch posts. */
  exportDelayMillis?: number
  /** Per-request deadline for one OTLP post. */
  requestTimeoutMillis?: number
  /** Outer bound on the drain performed while the plugin is disposed. */
  shutdownTimeoutMillis?: number
  /** Where the integration writes its own log; defaults to `$DSH_HOME/litefuse.log`. */
  logFile?: string
  /** Whether to keep verbose lines; defaults to whether `DSH_LITEFUSE_DEBUG` is set. */
  debug?: boolean
}

/**
 * Reject a non-positive bound at load rather than at the first export, where a
 * zero delay would spin and a zero timeout would abort every request.
 * @param label - the configuration field being checked.
 * @param value - the configured value.
 * @returns the value, once it is a positive finite number.
 */
function positive(label: string, value: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`dsh-litefuse-plugin: ${label} must be a positive finite number, got ${String(value)}`)
  }
  return value
}

/**
 * Read one optional string field, rejecting a wrong type at load.
 * @param label - the configuration field being checked.
 * @param value - the configured value.
 * @param fallback - value used when the field is absent.
 * @returns the configured string, or the fallback.
 */
function text(label: string, value: string | undefined, fallback: string): string {
  if (value === undefined) return fallback
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`dsh-litefuse-plugin: ${label} must be a non-empty string, got ${JSON.stringify(value)}`)
  }
  return value
}

/**
 * Read one request-input scope, rejecting an unknown value at load.
 * @param value - the configured scope.
 * @returns the scope, defaulting to the whole request.
 */
function requestInputScope(value: RequestInputScope | undefined): RequestInputScope {
  switch (value) {
    case undefined:
      return 'full'
    case 'full':
    case 'delta':
    case 'none':
      return value
    default:
      throw new Error(`dsh-litefuse-plugin: requestInput must be full, delta, or none, got ${JSON.stringify(value)}`)
  }
}

/**
 * Brand one environment-variable name as a credential reference.
 *
 * The harness's own `credentialRef` performs exactly this check, but importing
 * it would make `@deepseek-ai/dsh-credentials` — and through it Cordis — a
 * runtime dependency of this package. A plugin installed into a profile must
 * never load its own copy of Cordis: the host would then hold two distinct
 * `Context` classes and service wiring would silently fail. Every import here
 * is therefore type-only, and this package ships no runtime dependency at all.
 * @param value - a POSIX shell identifier such as `LITEFUSE_SECRET_KEY`.
 * @returns the same name, typed as the reference the credential seam accepts.
 */
function credentialRef(value: string): CredentialRef {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new Error(`dsh-litefuse-plugin: credential reference "${value}" must be an environment-variable name`)
  }
  return value as CredentialRef
}

/**
 * Resolve one credential from the first reference that has a value: the
 * harness credential store when it is mounted, then the process environment
 * the launcher populated from `~/.dsh/.env`.
 * @param ctx - context whose optional `credentials` service is consulted first.
 * @param refs - reference names in precedence order.
 * @returns the first non-empty value, or `undefined` when none is configured.
 */
async function resolveCredential(ctx: Context, refs: readonly string[]): Promise<string | undefined> {
  const credentials = ctx.get('credentials')
  for (const ref of refs) {
    if (credentials !== undefined) {
      const hit = await credentials.resolve(credentialRef(ref))
      if (hit !== undefined) return hit.value
    }
    const ambient = process.env[ref]
    if (ambient !== undefined && ambient.length > 0) return ambient
  }
  return undefined
}

/**
 * Mount Litefuse tracing on one context.
 * @param ctx - the composing Cordis context.
 * @param config - validated plugin configuration.
 */
export function apply(ctx: Context, config: Config): void {
  const log = createLog(
    text('logFile', config.logFile, join(harnessHome(), LOG_FILE_NAME)),
    config.debug ?? (process.env[DEBUG_ENV] ?? '').length > 0,
  )
  if (config.enabled === false) {
    log.info('disabled by configuration; no trace leaves this process')
    return
  }
  const baseUrl = text('baseUrl', config.baseUrl, DEFAULT_BASE_URL)
  try {
    const parsed = new URL(baseUrl)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error(`dsh-litefuse-plugin: baseUrl must be http(s), got ${parsed.protocol}`)
    }
  } catch (error) {
    if (error instanceof TypeError) throw new Error(`dsh-litefuse-plugin: baseUrl is not a valid URL: ${JSON.stringify(baseUrl)}`)
    throw error
  }
  const publicRefs = config.publicKeyEnv === undefined ? PUBLIC_KEY_REFS : [config.publicKeyEnv, ...PUBLIC_KEY_REFS]
  const secretRefs = config.secretKeyEnv === undefined ? SECRET_KEY_REFS : [config.secretKeyEnv, ...SECRET_KEY_REFS]
  const shutdownTimeoutMillis = positive('shutdownTimeoutMillis', config.shutdownTimeoutMillis ?? DEFAULT_SHUTDOWN_TIMEOUT_MILLIS)

  const exporter = new LitefuseSpanExporter({
    baseUrl,
    credentials: async () => {
      const [publicKey, secretKey] = await Promise.all([
        resolveCredential(ctx, publicRefs),
        resolveCredential(ctx, secretRefs),
      ])
      if (publicKey === undefined || secretKey === undefined) return undefined
      return { publicKey, secretKey }
    },
    exportDelayMillis: positive('exportDelayMillis', config.exportDelayMillis ?? DEFAULT_EXPORT_DELAY_MILLIS),
    requestTimeoutMillis: positive('requestTimeoutMillis', config.requestTimeoutMillis ?? DEFAULT_REQUEST_TIMEOUT_MILLIS),
    onFailure: message => log.warn(message),
    onSuccess: message => log.debug(message),
  }, {
    // `service.*` describes the application being observed, not the observer.
    // Its version is deliberately absent: the harness keeps it in
    // `@deepseek-ai/dsh-llm`'s manifest and exposes it through no service, so a
    // plugin that imports nothing at runtime cannot read it — and stamping this
    // package's version there would assert a `deepseek-harness` release that
    // does not exist. The observer's own version rides the instrumentation
    // scope below, which is where OTel puts it.
    resource: { 'service.name': 'deepseek-harness' },
    scopeName: SCOPE_NAME,
    scopeVersion: version,
  })

  const assembler = new TraceAssembler({
    agentName: text('agentName', config.agentName, DEFAULT_AGENT_NAME),
    environment: text('environment', config.environment, DEFAULT_ENVIRONMENT),
    userId: text('userId', config.userId, defaultUserId()),
    tags: config.tags ?? [DEFAULT_TAG],
    maxValueChars: positive('maxValueChars', config.maxValueChars ?? DEFAULT_MAX_VALUE_CHARS),
    requestInput: requestInputScope(config.requestInput),
    delegationTools: config.delegationTools ?? [...DEFAULT_DELEGATION_TOOLS],
    ...config.release === undefined ? {} : { release: config.release },
  }, (span) => {
    // A span with no parent is a turn root, i.e. one finished trace.
    if (span.parentSpanId === undefined) log.debug(turnSummary(span))
    exporter.enqueue(span)
  })

  // Every handler is self-contained: session dispatch stops on a throwing
  // listener, so nothing from this plugin may escape into the agent loop.
  ctx.on('session/event', (session, event) => {
    try {
      assembler.record(session, event)
    } catch (error) {
      log.warn(`dropped ${event.type}: ${String(error)}`)
    }
  })
  ctx.on('session/disposed', (session) => {
    try {
      assembler.closeSession(String(session.id), Date.now())
    } catch (error) {
      log.warn(`session close failed: ${String(error)}`)
    }
  })
  // The turn boundary; returning void keeps the loop's awaited parallel free of
  // export latency.
  ctx.on('session/flush', () => {
    void exporter.flush(shutdownTimeoutMillis)
  })

  ctx.effect(() => async () => {
    assembler.closeAll(Date.now())
    await exporter.shutdown(shutdownTimeoutMillis)
  }, 'litefuse export')

  void announce(ctx, log, publicRefs, secretRefs, baseUrl)
}

/**
 * Report at load whether a key pair is reachable, so a missing credential
 * surfaces as one clear line instead of silence.
 */
async function announce(
  ctx: Context,
  log: LitefuseLog,
  publicRefs: readonly string[],
  secretRefs: readonly string[],
  baseUrl: string,
): Promise<void> {
  try {
    const [publicKey, secretKey] = await Promise.all([
      resolveCredential(ctx, publicRefs),
      resolveCredential(ctx, secretRefs),
    ])
    if (publicKey === undefined || secretKey === undefined) {
      log.warn(
        `no credentials found — set ${publicRefs[0]} and ${secretRefs[0]} `
        + 'in ~/.dsh/.env or the harness credential store; traces are dropped until then',
      )
      return
    }
    // Only the public key's prefix is ever written: it identifies the project
    // without putting a credential in a file the user may paste into an issue.
    log.info(`v${version} exporting to ${baseUrl} (key ${publicKey.slice(0, 10)}…)`)
  } catch (error) {
    log.warn(`credential probe failed: ${String(error)}`)
  }
}

/**
 * Render one finished trace as a verification line: what the smoke test in the
 * README tells the operator to look for.
 *
 * Metadata is addressed per key beneath its namespace, which is the only form
 * the assembler writes — the namespace attribute itself is deliberately never
 * set, so reading it would report every field as absent.
 * @param root - the turn's root span, taken as it leaves the assembler.
 * @returns the log line describing that turn.
 */
function turnSummary(root: LitefuseSpan): string {
  const field = (name: string): string => String(root.attributes[`${OBSERVATION_METADATA}.agent_${name}`])
  return `turn closed "${root.name}" trace=${root.traceId}`
    + ` session=${field('session_id')}`
    + ` steps=${field('steps')}`
    + ` api=${field('api_calls')}`
    + ` tools=${field('tool_calls')}`
    + ` duration=${field('duration_ms')}ms`
}

/**
 * Default trace `user.id`: the OS user name, with a stable fallback when the
 * platform does not report one.
 * @returns the identifier placed on every trace.
 */
function defaultUserId(): string {
  const name = process.env['USER'] ?? process.env['USERNAME']
  return name !== undefined && name.length > 0 ? name : 'dsh-user'
}
