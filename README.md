# dsh-litefuse

[Litefuse](https://litefuse.cloud) observability for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

Every user turn becomes one trace: an `agent` root, one `generation` per model call with real latency and token usage, one `tool` per tool execution, and a nested container for every subagent the turn delegated to. It implements the [Litefuse agent-trace spec v1.2](https://litefuse.ai/litefuse-agent-trace-spec.md).

```
DeepSeek Harness — Turn 3            AGENT      1.9s   input: "why is the build failing?"
├── plan (2 tools) #1                GENERATION 820ms  in 1.2k · out 96 · cache-read 18k
├── tool: bash (pnpm) #2             TOOL       410ms
├── tool: read (tsconfig.json) #3    TOOL       12ms
├── plan (1 tool) #4                 GENERATION 640ms
├── tool (1 subagent) #5             TOOL       9.4s
│   └── subagent                     AGENT      8.8s   ← delegation overhead: 0.6s
│       ├── plan (1 tool) #1         GENERATION        ← numbering restarts per container
│       ├── tool: grep (TS2345) #2   TOOL
│       └── subagent response        GENERATION
└── response                         GENERATION 1.1s   output: the final answer
```

Zero runtime dependencies — no Langfuse SDK. Spans go straight to the OTLP endpoint, declaring `x-langfuse-ingestion-version: 4`, the documented opt-in for a custom exporter that writes complete spans inline.

Unlike the file-tailing collectors Litefuse ships for other agents, this one runs **in-process** on the harness's own session event stream, so it records what actually happened rather than what a transcript could be reconstructed to mean: true per-call latency, time to first token, tool durations, disjoint cache-token accounting, and the exact request each generation was sent.

## Install

```bash
dsh plugin --profile web add -w dsh-litefuse
```

That registers the package as a patch layer in `$DSH_HOME/profiles/web`. Use `--profile <name>` for whichever profile you boot; repeat it per profile.

To install from a local checkout instead, pass its absolute path:

```bash
dsh plugin --profile web add -w /absolute/path/to/dsh-litefuse
```

Then put your project key pair where the harness can read it — in `~/.dsh/.env`:

```
LITEFUSE_PUBLIC_KEY=pk-lf-…
LITEFUSE_SECRET_KEY=sk-lf-…
```

Keys come from **Settings → API Keys → Create new API keys** in your Litefuse project (sign up at <https://litefuse.cloud/auth/sign-up>). Self-hosting: add `LITEFUSE_BASE_URL=https://your-host`.

Restart dsh. There is nothing else to configure.

### Verify

```bash
DSH_LITEFUSE_DEBUG=1 dsh web
```

Send one message, then read the integration's own log:

```bash
tail -5 ~/.dsh/litefuse.log
```

A working install prints its endpoint at boot, one line per finished turn, and one per delivered batch:

```
[info] v0.1.0 exporting to https://litefuse.cloud (key pk-lf-a1b2…)
[debug] turn closed "DeepSeek Harness — Turn 1" trace=… session=… steps=3 api=2 tools=1 duration=4120ms
[debug] sent 4 span(s) -> https://litefuse.cloud/api/public/otel/v1/traces HTTP 200
```

Open <https://litefuse.cloud> → your project → **Tracing**. Traces appear as soon as a turn's *first* span completes, not at turn end.

If nothing arrives, that log names the reason — missing credentials, an HTTP status, or a transport error. A `totalCost` of 0 in the UI is not a collection problem: it means the project has no price entry for your model (**Settings → Models**).

### Uninstall

```bash
dsh plugin --profile web remove -w dsh-litefuse
```

`DSH_LITEFUSE_DISABLED=1` turns exporting off without uninstalling.

## How it works

The plugin subscribes to `session/event`, the harness's post-commit append feed, and folds the events of each turn into spans:

| Session event | Becomes |
|---|---|
| `turn/start` … `turn/end` | the trace and its `agent` root span |
| `user/message` (`source.kind: user`) | the trace input |
| `step/start` … `assistant/message` | one `generation`, named for what the model did |
| `assistant/chunk` (first of a step) | that generation's `completion_start_time` — time to first token |
| `tool/call` … `tool/result` | one `tool` span, linked to its plan through `agent_plan_step` |
| `request/header`, `request/context` | the model name, sampling parameters, and context window |
| `compaction/end` | a `context compaction` event, which explains the next call's token drop |
| a child session whose parent has a delegation call in flight | a `subagent` container under that call's tool span |

Spans mount **flat** under their container; the only depth is a real subagent run. Generations and tools share one step counter, so `#N` is a single chronological sequence and `tool.agent_plan_step == generation.agent_step_index` joins a tool back to the call that requested it.

Each span is written **once, when it ends** — OTel spans are immutable, so an in-flight step is deliberately invisible until it closes. Trace-level attributes ride on every span, which is what lets a trace appear before its root does.

### Subagents

A delegated run is its own session in the harness. When one starts while a delegation call is in flight in its parent, the child's steps mount under a `subagent` container parented to that call's tool span, numbering restarted at #1, its closing answer named `subagent response`, and its token usage rolled into the parent trace's total. The gap between the tool span and the container is the real cost of delegating.

A child that starts with no delegation call in flight — a background or continuable subagent — gets its own trace instead, with `agent_parent_session_id` in the root metadata.

### Metadata

All metadata is flat under one `agent_` prefix — never a per-agent namespace — so a single Litefuse dashboard query works across every agent integration. Absent fields are omitted rather than padded with nulls.

- **Root and subagent container**: `agent_turn_number`, `agent_session_id`, `agent_parent_session_id`, `agent_cwd`, `agent_provider`, `agent_model`, `agent_api_calls`, `agent_tool_calls`, `agent_steps`, `agent_duration_ms`, `agent_context_window`, `agent_end_reason`, `agent_subagent`, plus the token rollup `agent_input_tokens` / `agent_output_tokens` / `agent_cache_read_tokens` / `agent_cache_write_tokens` / `agent_reasoning_tokens` / `agent_total_tokens` / `agent_accounted_generations`
- **Generation**: `agent_step_index`, `agent_api_duration_ms`, `agent_time_to_first_token_ms`, `agent_tool_call_count`, `agent_thinking_chars`, `agent_reasoning_tokens`, `agent_input_scope`, truncation flags
- **Tool**: `agent_tool_name`, `agent_tool_call_id`, `agent_step_index`, `agent_plan_step`, `agent_duration_ms`, `agent_is_error`, `agent_error_code`, `agent_subagent_count`, truncation flags

Metadata rides as **per-key span attributes** (`langfuse.observation.metadata.agent_step_index`), not as one serialized blob. A JSON string would be stored verbatim beside its parsed copy, leaving JSON nested inside a string in the raw attribute set, and the trace spec forbids pre-serialized JSON as a metadata value because flattening it server-side corrupts the escaping.

Token counts use the keys Litefuse prices and classifies from: `input`, `output`, `output_reasoning_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`. Prompt counts arrive disjoint from the harness, so they sum to billed input unchanged. Completion is the opposite — the harness folds reasoning into `outputTokens` — so reasoning is subtracted back out of `output` and reported as its sibling. Litefuse sums every key containing `output` into the displayed Output figure, and its own ingestion processor normalizes provider payloads exactly this way, so the split keeps both the breakdown and the cost right. A model definition should price `output_reasoning_tokens` alongside `output`; the shipped price table already does for every reasoning model it knows.

**Reasoning tokens ride as `reasoning`, plus an explicit `total`.** Litefuse sums every usage key containing `input` into the input figure, every key containing `output` into the output figure, and — unless you supply `total` — every key into the billed total. A provider already counts reasoning inside its completion tokens, so the ecosystem's `output_reasoning` spelling would bill them twice. Naming the key `reasoning` puts it in the breakdown's *Other* section, and supplying `total` keeps the billed figure equal to input + output + cache. The result matches what the harness's own Trajectory view shows: `1028 output = 677 reasoning + 351 content`.

**The token rollup is metadata only.** An `agent` span — turn root or subagent container — carries the totals for itself and everything nested beneath it as `agent_*_tokens`, but never as `usage_details`. Litefuse prices a trace by summing its spans, so a container that also declared its children's tokens would double the bill. Read `agent_total_tokens` on the root for "how big was this turn"; read `totalCost` for what it cost.

## Configuration

The install patch reads everything from the environment, so most deployments need no configuration. To override, add an entry to `~/.dsh/profiles/<name>/cordis.patch.yml`:

```yaml
- id: litefuse
  config:
    environment: staging
    agentName: My Agent
    requestInput: delta
    tags: [dsh, team-platform]
```

An id-targeted patch replaces the whole `config`, so restate the fields you keep.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` mounts the plugin and exports nothing |
| `baseUrl` | `https://litefuse.cloud` | Litefuse endpoint; the OTLP trace path is appended |
| `publicKeyEnv` / `secretKeyEnv` | `LITEFUSE_PUBLIC_KEY` / `LITEFUSE_SECRET_KEY` | credential *references*; `LANGFUSE_*` is the fallback |
| `environment` | `production` | Litefuse tracing environment on every span |
| `agentName` | `DeepSeek Harness` | the name in `<agent> — Turn N` |
| `userId` | `$USER` | trace-level `user.id` |
| `tags` | `[dsh]` | trace tags, beside the generated `model:<name>` |
| `release` | — | optional release identifier on every trace |
| `requestInput` | `full` | `full` sends the whole request; `delta` only the messages added since the last call; `none` omits inputs |
| `maxValueChars` | `1000000` | truncation budget per input/output |
| `delegationTools` | `[subagent, subagent_fork]` | tool names whose in-flight call hosts a subagent container |
| `exportDelayMillis` | `1000` | how long an ended span waits for company before its batch posts |
| `requestTimeoutMillis` | `10000` | per-request deadline |
| `shutdownTimeoutMillis` | `3000` | outer bound on the drain at teardown |
| `logFile` | `$DSH_HOME/litefuse.log` | the integration's own log |
| `debug` | `$DSH_LITEFUSE_DEBUG` | keep verbose lines |

Credentials are **references, not values**: configuration names an environment variable, and the value comes from the harness credential store (`~/.dsh/.credentials.yaml`) when one is mounted, otherwise from the process environment that `~/.dsh/.env` feeds. Nothing here ever writes a key to a file, and only the public key's first ten characters are ever logged.

## Design notes

**Fail-open, always.** Every handler is self-contained and every failure is logged and swallowed. Session dispatch stops on a throwing listener, so an exception escaping this plugin would starve every observer registered after it. An unreachable Litefuse project costs one request deadline and nothing else.

**Zero runtime dependencies.** The built plugin imports nothing but `node:module` and its own files — every harness and Cordis import is type-only. This is deliberate: a plugin installed into a profile that pulled in its own copy of `@deepseek-ai/cordis` would give the host two distinct `Context` classes, and service wiring would fail in ways that are very hard to diagnose.

**It writes its own log.** A booted dsh profile composes no logger plugin, so `ctx.logger` output is invisible. The file at `$DSH_HOME/litefuse.log` is where this integration reports, matching what Litefuse's other integrations do.

**No replay on load.** The plugin observes from the next `turn/start` onward. A turn already in flight when it loads is skipped rather than reconstructed, which is what keeps it from writing spans a previous process already sent.

## Known limitations

- **Parallel delegations from separate calls can mis-bind.** A child session binds to the parent's most recently started in-flight delegation call. When two delegation calls run concurrently in one step, a child can attach to the wrong one. Ordinary tools running beside a delegation are not affected — the configured `delegationTools` names are preferred.
- **Out-of-process subagents get their own traces.** Providers that run a child in another process (`acp`, `codex`, the SDK providers) publish no session events into this process, so their runs are not subtrees. In-process providers — which the shipped `subagent` and `subagent_fork` tools use — are.
- **Parallel tool spans can overstate duration.** The harness commits tool results in model order, so a fast call that finishes behind a slow sibling records its result timestamp, not its own completion.
- **`requestInput: full` folds the derived history per model call.** That is one pass over the session log per call — negligible beside a model round trip, but `delta` exists for very long sessions.
- **No durable outbox.** Spans buffered when the process dies are lost. Delivery is at-most-once by design; the session log remains the durable record.

## Releasing

Publishing is a tag push; CI builds, tests, and publishes with provenance.

```bash
npm version patch        # or minor / major — writes package.json and tags
git push --follow-tags
```

The workflow refuses a tag that disagrees with `package.json`, and `prepack`
builds before packing, so the tarball always carries `lib/` and installers never
compile anything.

First release only: publish once by hand (`npm publish --access public`) to
create the package, then either add an `NPM_TOKEN` repository secret or
configure npm Trusted Publishing for this workflow, which authenticates over
OIDC and needs no token at all.

## Development

```bash
npm install
npm test        # 32 tests over the real @deepseek-ai/dsh-session Session
npm run typecheck
npm run build
```

The tests drive the shipped harness session store rather than a stand-in, so the assembler is exercised against the real append validation, surface rules, and derived-history projection it reads in production. `tests/plugin.spec.ts` boots the plugin against a local stand-in for the Litefuse ingest endpoint and asserts the OTLP wire form, the Basic authorization, single-write spans, and fail-open behavior on 500s, unreachable hosts, and missing credentials.

To check a real Litefuse deployment without waiting for a model round trip:

```bash
node scripts/send-verification-trace.mjs
```

It drives the real Session through a scripted turn — one tool call plus a delegated subagent run — and posts the result to `$LITEFUSE_BASE_URL`, tagged `environment: development` and named `dsh-litefuse-verify — Turn 1` so it cannot be mistaken for agent traffic. Read it back with [`litefuse-cli`](https://www.npmjs.com/package/litefuse-cli):

```bash
npx -y litefuse-cli api traces get <traceId>
```

Note that `traces list` does not project trace metadata — it reports `{}` even when the metadata is stored. Use `traces get` to see the `agent_*` fields.

## License

MIT
