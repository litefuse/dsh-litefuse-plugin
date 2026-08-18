import { readFileSync, rmSync } from 'node:fs'
import { createServer, type Server } from 'node:http'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { AddressInfo } from 'node:net'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SessionStore from '@deepseek-ai/dsh-session'
import * as litefuse from '../src/index.js'
import { LITEFUSE_TRACES_PATH } from '../src/otlp.js'
import { createSession, TurnWriter } from './fixture.js'

/** One captured OTLP request. */
interface Capture {
  path: string
  ingestionVersion: string | undefined
  authorization: string | undefined
  contentType: string | undefined
  payload: OtlpPayload
}

/** The subset of the OTLP JSON envelope these tests read back. */
interface OtlpPayload {
  resourceSpans: {
    resource: { attributes: { key: string; value: Record<string, unknown> }[] }
    scopeSpans: {
      scope: { name: string; version: string }
      spans: {
        traceId: string
        spanId: string
        parentSpanId?: string
        name: string
        kind: number
        startTimeUnixNano: string
        endTimeUnixNano: string
        attributes: { key: string; value: Record<string, unknown> }[]
      }[]
    }[]
  }[]
}

/** A local stand-in for the Litefuse ingest endpoint. */
class FakeLitefuse {
  readonly captures: Capture[] = []
  status = 200
  private server: Server | undefined

  /** Start listening on an ephemeral port and return its base URL. */
  async start(): Promise<string> {
    const server = createServer((request, response) => {
      const chunks: Buffer[] = []
      request.on('data', (chunk: Buffer) => chunks.push(chunk))
      request.on('end', () => {
        this.captures.push({
          path: request.url ?? '',
          ingestionVersion: request.headers['x-langfuse-ingestion-version'] as string | undefined,
          authorization: request.headers.authorization,
          contentType: request.headers['content-type'],
          payload: JSON.parse(Buffer.concat(chunks).toString()) as OtlpPayload,
        })
        response.writeHead(this.status)
        response.end('{}')
      })
    })
    this.server = server
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const { port } = server.address() as AddressInfo
    return `http://127.0.0.1:${port}`
  }

  /** Every span across every captured request, in arrival order. */
  spans(): OtlpPayload['resourceSpans'][number]['scopeSpans'][number]['spans'] {
    return this.captures.flatMap(capture =>
      capture.payload.resourceSpans.flatMap(resource =>
        resource.scopeSpans.flatMap(scope => scope.spans)))
  }

  /** Stop listening. */
  async stop(): Promise<void> {
    const server = this.server
    if (server === undefined) return
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

/** Read one span attribute back out of the OTLP encoding. */
function attributeOf(
  span: OtlpPayload['resourceSpans'][number]['scopeSpans'][number]['spans'][number],
  key: string,
): unknown {
  const entry = span.attributes.find(candidate => candidate.key === key)
  if (entry === undefined) return undefined
  return Object.values(entry.value)[0]
}

/** A booted tree plus the teardown that drains the exporter. */
interface Booted {
  ctx: Context
  stop: () => Promise<void>
}

/** Drive one complete turn on a fresh session. */
function runTurn(ctx: Context): void {
  const writer = new TurnWriter(createSession(ctx), 1)
  writer.start().prompt('what time is it').header()
  const step = writer.openStep()
  writer.assistant(step, [{ type: 'text', text: 'I cannot tell.' }], { inputTokens: 12, outputTokens: 4 })
  writer.closeStep(step).end()
}

describe('plugin composition', () => {
  let fake: FakeLitefuse
  let baseUrl: string
  const TEST_PUBLIC_KEY = 'pk-lf-00000000-0000-4000-8000-000000000000'
  const TEST_SECRET_KEY = 'sk-lf-00000000-0000-4000-8000-000000000000'
  // Real developer machines carry Litefuse or Langfuse keys for other agents;
  // every reference the plugin consults is cleared so the suite can never post
  // a test trace to somebody's real project.
  const REFERENCES = [
    'LITEFUSE_PUBLIC_KEY', 'LITEFUSE_SECRET_KEY',
    'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY',
  ] as const
  const inherited = new Map<string, string | undefined>()
  // The integration's own log defaults into the real harness home; tests keep
  // it in a temporary file so a run never writes to the developer's ~/.dsh.
  const logFile = join(tmpdir(), 'dsh-litefuse-plugin-test.log')

  beforeEach(async () => {
    fake = new FakeLitefuse()
    baseUrl = await fake.start()
    for (const reference of REFERENCES) {
      inherited.set(reference, process.env[reference])
      delete process.env[reference]
    }
    process.env['LITEFUSE_PUBLIC_KEY'] = TEST_PUBLIC_KEY
    process.env['LITEFUSE_SECRET_KEY'] = TEST_SECRET_KEY
    rmSync(logFile, { force: true })
  })

  afterEach(async () => {
    await fake.stop()
    for (const [reference, value] of inherited) {
      if (value === undefined) delete process.env[reference]
      else process.env[reference] = value
    }
    inherited.clear()
  })

  /** Boot a real tree with the session store and this plugin composed together. */
  async function boot(config: Partial<litefuse.Config> = {}): Promise<Booted> {
    const ctx = new Context()
    ctx.plugin(SessionStore)
    const fiber = ctx.plugin(litefuse, { baseUrl, exportDelayMillis: 10, environment: 'test', logFile, debug: true, ...config })
    await new Promise(resolve => setTimeout(resolve, 30))
    return { ctx, stop: async () => { await fiber.dispose() } }
  }

  it('posts OTLP JSON to the Litefuse trace endpoint with Basic authorization', async () => {
    const { ctx, stop } = await boot()
    runTurn(ctx)
    await stop()

    expect(fake.captures.length).toBeGreaterThan(0)
    const first = fake.captures[0]!
    expect(first.path).toBe(LITEFUSE_TRACES_PATH)
    expect(first.contentType).toBe('application/json')
    const decoded = Buffer.from(first.authorization!.replace('Basic ', ''), 'base64').toString()
    expect(decoded).toBe(`${TEST_PUBLIC_KEY}:${TEST_SECRET_KEY}`)
  })

  it('ships a complete, well-formed trace for one turn', async () => {
    const { ctx, stop } = await boot()
    runTurn(ctx)
    await stop()

    const spans = fake.spans()
    expect(spans.map(span => span.name)).toEqual(['response', 'DeepSeek Harness — Turn 1'])
    const [generation, root] = spans as [typeof spans[number], typeof spans[number]]
    expect(root.traceId).toMatch(/^[0-9a-f]{32}$/)
    expect(root.spanId).toMatch(/^[0-9a-f]{16}$/)
    expect(root.parentSpanId).toBeUndefined()
    expect(generation.parentSpanId).toBe(root.spanId)
    expect(generation.kind).toBe(1)
    expect(Number(generation.endTimeUnixNano)).toBeGreaterThanOrEqual(Number(generation.startTimeUnixNano))
    expect(attributeOf(generation, 'langfuse.observation.type')).toBe('generation')
    expect(attributeOf(generation, 'langfuse.observation.usage_details')).toBe('{"input":12,"output":4}')
    // The ingest gate rejects clients that cannot emit complete spans inline.
    expect(fake.captures[0]!.ingestionVersion).toBe('4')
    expect(attributeOf(root, 'langfuse.trace.output')).toBe('I cannot tell.')
    expect(attributeOf(root, 'langfuse.environment')).toBe('test')
  })

  it('identifies the harness in the resource and this plugin in the scope', async () => {
    const { ctx, stop } = await boot()
    runTurn(ctx)
    await stop()

    const payload = fake.captures[0]!.payload
    const resource = payload.resourceSpans[0]!.resource.attributes
    expect(resource.find(entry => entry.key === 'service.name')?.value['stringValue']).toBe('deepseek-harness')
    // The observed application's version is unknowable from here, and this
    // package's version is not it — carrying one would assert a harness release
    // that does not exist.
    expect(resource.find(entry => entry.key === 'service.version')).toBeUndefined()
    // The observer's version belongs to the instrumentation scope instead.
    expect(payload.resourceSpans[0]!.scopeSpans[0]!.scope.version)
      .toBe(createRequire(import.meta.url)('../package.json').version)
    // The `langfuse-sdk` prefix is what tells the ingest these spans arrive
    // complete, so it does not fall back to copying the raw attribute map into
    // observation metadata. Renaming the scope without that prefix puts an
    // unreadable duplicate of every attribute back on every observation.
    expect(payload.resourceSpans[0]!.scopeSpans[0]!.scope.name).toBe('langfuse-sdk-dsh-litefuse-plugin')
  })

  it('writes each span exactly once', async () => {
    const { ctx, stop } = await boot()
    runTurn(ctx)
    await stop()

    const ids = fake.spans().map(span => span.spanId)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('exports nothing and stays silent when disabled', async () => {
    const { ctx, stop } = await boot({ enabled: false })
    runTurn(ctx)
    await stop()

    expect(fake.captures).toEqual([])
  })

  it('keeps the session usable when the endpoint rejects the batch', async () => {
    fake.status = 500
    const { ctx, stop } = await boot()
    expect(() => runTurn(ctx)).not.toThrow()
    await stop()

    expect(fake.captures.length).toBeGreaterThan(0)
  })

  it('keeps the session usable when the endpoint is unreachable', async () => {
    await fake.stop()
    const { ctx, stop } = await boot({ requestTimeoutMillis: 50 })
    expect(() => runTurn(ctx)).not.toThrow()
    await expect(stop()).resolves.not.toThrow()
  })

  it('drops the batch instead of sending an unauthorized request when no key is configured', async () => {
    for (const reference of REFERENCES) delete process.env[reference]
    const { ctx, stop } = await boot()
    runTurn(ctx)
    await stop()

    expect(fake.captures).toEqual([])
  })

  it('records what happened in its own log, because a dsh profile has no logger', async () => {
    const { ctx, stop } = await boot()
    runTurn(ctx)
    await stop()

    const written = readFileSync(logFile, 'utf8')
    expect(written).toMatch(/exporting to http:\/\/127\.0\.0\.1/)
    // Pinned whole, not by prefix: every field after `trace=` is read back out
    // of the root span's metadata, and reading it the wrong way degrades to a
    // line of `undefined`s that a prefix match still accepts.
    expect(written).toMatch(
      /turn closed "DeepSeek Harness — Turn 1" trace=[0-9a-f]{32} session=\S+ steps=1 api=1 tools=0 duration=\d+ms/,
    )
    expect(written).toMatch(/sent \d+ span\(s\)/)
    // The public key is identified by prefix only, and the secret never appears.
    expect(written).not.toContain(TEST_SECRET_KEY)
    expect(written).not.toContain(TEST_PUBLIC_KEY)
  })

  it('names the missing references in its log when no credentials are configured', async () => {
    for (const reference of REFERENCES) delete process.env[reference]
    const { stop } = await boot()
    await stop()

    expect(readFileSync(logFile, 'utf8')).toMatch(/no credentials found — set LITEFUSE_PUBLIC_KEY and LITEFUSE_SECRET_KEY/)
  })

  it('rejects a malformed endpoint at load rather than at the first export', async () => {
    const ctx = new Context()
    ctx.plugin(SessionStore)
    expect(() => litefuse.apply(ctx, { baseUrl: 'ftp://litefuse.invalid', logFile })).toThrow(/baseUrl must be http/)
    expect(() => litefuse.apply(ctx, { exportDelayMillis: 0, logFile })).toThrow(/exportDelayMillis/)
  })
})
