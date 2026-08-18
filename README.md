# dsh-litefuse-plugin

English | [中文](README-zh.md)

DeepSeek Harness [Litefuse](https://litefuse.ai) Plugin for Agent Observability and Evals.

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

## Install

```bash
npx @deepseek-ai/dsh plugin --profile web add -w dsh-litefuse-plugin
```

Put your project key pair in `~/.dsh/.env`:

```
LITEFUSE_PUBLIC_KEY=pk-lf-…
LITEFUSE_SECRET_KEY=sk-lf-…
```

Keys come from **Settings → API Keys → Create new API keys** in your Litefuse project (sign up at <https://litefuse.cloud/auth/sign-up>).

Then boot as usual:

```bash
npx @deepseek-ai/dsh web
```

Send one message and open <https://litefuse.cloud> → your project → **Tracing**. The trace appears as soon as the turn's *first* span completes, not at turn end.

### Uninstall

```bash
npx @deepseek-ai/dsh plugin --profile web remove -w dsh-litefuse-plugin
```

`DSH_LITEFUSE_DISABLED=1` turns exporting off without uninstalling.

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
| `maxValueChars` | `1000000` | truncation budget per input/output, the trace input included |
| `delegationTools` | `[subagent, subagent_fork]` | tool names whose in-flight call hosts a subagent container |
| `exportDelayMillis` | `1000` | how long an ended span waits for company before its batch posts |
| `requestTimeoutMillis` | `10000` | per-request deadline |
| `shutdownTimeoutMillis` | `3000` | outer bound on the drain at teardown |
| `logFile` | `$DSH_HOME/litefuse.log` | the integration's own log |
| `debug` | `$DSH_LITEFUSE_DEBUG` | keep verbose lines |

Credentials are **references, not values**: configuration names an environment variable, and the value comes from the harness credential store (`~/.dsh/.credentials.yaml`) when one is mounted, otherwise from the process environment that `~/.dsh/.env` feeds. Nothing here ever writes a key to a file, and only the public key's first ten characters are ever logged.

## How it works

Zero runtime dependencies — no Langfuse SDK. Spans go straight to the OTLP endpoint, declaring `x-langfuse-ingestion-version: 4`, the documented opt-in for a custom exporter that writes complete spans inline.

Unlike the file-tailing collectors Litefuse ships for other agents, this one runs **in-process** on the harness's own session event stream, so it records what actually happened rather than what a transcript could be reconstructed to mean: true per-call latency, time to first token, tool durations, disjoint cache-token accounting, and the exact request each generation was sent.

The plugin subscribes to `session/event`, the harness's post-commit append feed, and folds the events of each turn into spans:

| Session event | Becomes |
|---|---|
| `turn/start` … `turn/end` | the trace and its `agent` root span |
| `user/message` (`source.kind: user`) | the trace input |
| `step/start` … `assistant/message` | one `generation`, named for what the model did |
| `assistant/chunk` (first of a step) | that generation's `completion_start_time` — time to first token |
| `tool/call` … `tool/result` | one `tool` span, linked to its plan through `agent_plan_step` |
| `tool/code-dispatch-start` … `tool/code-dispatch` | one nested `tool` span per call a `run_code` program made |
| `request/header`, `request/context` | the model name, sampling parameters, and context window |
| `compaction/end` | a `context compaction` event, which explains the next call's token drop |
| `subagent/descriptor` | which delegation call a child run belongs to, and whether that call awaits it |
| a child session matched to a delegation call | a `subagent` container under that call's tool span |

Spans mount **flat** under their container; the only depth is a real subagent run. Generations and tools share one step counter, so `#N` is a single chronological sequence and `tool.agent_plan_step == generation.agent_step_index` joins a tool back to the call that requested it.

Each span is written **once, when it ends** — OTel spans are immutable, so an in-flight step is deliberately invisible until it closes. Trace-level attributes ride on every span, which is what lets a trace appear before its root does.

### Code mode

A `run_code` program calls tools directly, so those calls never appear as `tool/call` and a whole program's work would otherwise collapse into one opaque span. They mount as nested `tool` spans under the call that ran them — the bridge drains every in-flight dispatch before the parent returns, so the enclosure is structural rather than inferred. They carry no `#N`: the step counter numbers the model's calls and the tools it requested, and a dispatch is neither. `agent_tool_calls` therefore keeps counting only what the model asked for, with `agent_code_dispatches` beside it.

### Subagents

A delegated run is its own session in the harness. When one starts while a delegation call is in flight in its parent, the child's steps mount under a `subagent` container parented to that call's tool span, numbering restarted at #1, its closing answer named `subagent response`, and its token usage rolled into the parent trace's total. The gap between the tool span and the container is the real cost of delegating.

**Which call a child belongs to is decided by identity, once.** The harness hands a child no reference to the call that spawned it — a fresh session id, and a header naming only the parent session. What it does hand over, verbatim, is the delegation's `description`, as the `label` of the child's own `subagent/descriptor`, and its `prompt`, as the child's first user message. Either is an exact key, and one of them is what keeps concurrent delegations apart — concurrency being the normal case, since the subagent tool's own prompt asks the model to start independent delegations together in one assistant message.

A span carries its trace and its parent at the moment it is written and OTel spans are immutable, so a binding that could still change would be one that had already lied. The rule is therefore to bind at the earliest moment the answer is *determined* — an exact key match, or a single candidate call — and never to revise it. Waiting longer is not free: a `continuable` delegation reports its tool result within milliseconds of starting the child, before the child's own prompt is even recorded, so a decision deferred that far would find no candidate left. The descriptor is what makes the early decision an exact one rather than a guess, and for a spawned child it arrives in the constructor seed, which the `session/event` firehose never republishes — so it is read from the session log directly.

**A delegation call returning is not always the run ending.** A one-shot delegation resolves with the child's answer, so the parent's `tool/result` closes the container. A `continuable` one resolves at acceptance, with the entire run still ahead of it; closing there would seal an empty container and drop everything the child then did. Those containers close on the child's own `turn/end` instead, rolling their tokens into the turn that delegated them.

Only a tool named in `delegationTools` can host a container. An ordinary call that happens to be in flight is deliberately *not* a fallback: a background delegation reports its result before its child ever starts, so accepting any call would file a whole agent run — and its tokens — under whatever unrelated tool the parent was running at the time. A child that finds no delegation in flight gets its own trace instead, with `agent_parent_session_id` in the root metadata. A session that merely names a `parentSession` is not enough either; the harness's own `origin: subagent` classification is what admits a session into its parent's trace, so an ordinary fork stays outside it.

### Metadata

All metadata is flat under one `agent_` prefix — never a per-agent namespace — so a single Litefuse dashboard query works across every agent integration. Absent fields are omitted rather than padded with nulls.

- **Root and subagent container**: `agent_turn_number`, `agent_session_id`, `agent_parent_session_id`, `agent_cwd`, `agent_provider`, `agent_model`, `agent_api_calls`, `agent_tool_calls`, `agent_steps`, `agent_duration_ms`, `agent_context_window`, `agent_end_reason`, `agent_subagent`, `agent_code_dispatches`, plus the token rollup `agent_input_tokens` / `agent_output_tokens` / `agent_cache_read_tokens` / `agent_cache_write_tokens` / `agent_reasoning_tokens` / `agent_total_tokens` / `agent_accounted_generations`
- **Generation**: `agent_step_index`, `agent_api_duration_ms`, `agent_time_to_first_token_ms`, `agent_tool_call_count`, `agent_thinking_chars`, `agent_reasoning_tokens`, `agent_input_scope`, truncation flags
- **Tool**: `agent_tool_name`, `agent_tool_call_id`, `agent_step_index`, `agent_plan_step`, `agent_duration_ms`, `agent_is_error`, `agent_error_code`, `agent_subagent_count`, truncation flags
- **Code dispatch**: `agent_code_dispatch` (always `true`), `agent_tool_name`, `agent_tool_call_id`, `agent_parent_call_id`, `agent_root_call_id`, `agent_duration_ms`, `agent_is_error`, truncation flags

Metadata rides as **per-key span attributes** (`langfuse.observation.metadata.agent_step_index`), not as one serialized blob. A JSON string would be stored verbatim beside its parsed copy, leaving JSON nested inside a string in the raw attribute set, and the trace spec forbids pre-serialized JSON as a metadata value because flattening it server-side corrupts the escaping.

Token counts use the keys Litefuse prices and classifies from: `input`, `output`, `output_reasoning_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`. Prompt counts arrive disjoint from the harness, so they sum to billed input unchanged. Completion is the opposite — the harness folds reasoning into `outputTokens` — so reasoning is subtracted back out of `output` and reported as its sibling. Litefuse sums every key containing `output` into the displayed Output figure, and its own ingestion processor normalizes provider payloads exactly this way, so the split keeps both the breakdown and the cost right. A model definition should price `output_reasoning_tokens` alongside `output`; the shipped price table already does for every reasoning model it knows.

**The token rollup is metadata only.** An `agent` span — turn root or subagent container — carries the totals for itself and everything nested beneath it as `agent_*_tokens`, but never as `usage_details`. Litefuse prices a trace by summing its spans, so a container that also declared its children's tokens would double the bill. Read `agent_total_tokens` on the root for "how big was this turn"; read `totalCost` for what it cost.

## Troubleshooting

Nothing arriving? Boot with verbose logging:

```bash
DSH_LITEFUSE_DEBUG=1 npx @deepseek-ai/dsh web
```

then read the integration's own log:

```bash
tail -5 ~/.dsh/litefuse.log
```

A working install prints its endpoint at boot, one line per finished turn, and one per delivered batch:

```
[info] v0.1.0 exporting to https://litefuse.cloud (key pk-lf-a1b2…)
[debug] turn closed "DeepSeek Harness — Turn 1" trace=… session=… steps=3 api=2 tools=1 duration=4120ms
[debug] sent 4 span(s) -> https://litefuse.cloud/api/public/otel/v1/traces HTTP 200
```

That log names the reason it did not — missing credentials, an HTTP status, or a transport error. A `totalCost` of 0 in the UI is not a collection problem: it means the project has no price entry for your model (**Settings → Models**).

## Design notes

**Fail-open, always.** Every handler is self-contained and every failure is logged and swallowed. Session dispatch stops on a throwing listener, so an exception escaping this plugin would starve every observer registered after it. An unreachable Litefuse project costs one request deadline and nothing else.

**Zero runtime dependencies.** The built plugin imports nothing outside `node:` and its own files — every harness and Cordis import is type-only, and CI gates the property. This is deliberate: a plugin installed into a profile that pulled in its own copy of `@deepseek-ai/cordis` would give the host two distinct `Context` classes, and service wiring would fail in ways that are very hard to diagnose.

**The trace header is repeated, so it stays small.** Every span carries the trace's name, tags, session, and user so the trace is queryable before its root is written. The input is the one header field with no natural size, and repeating it would charge a pasted file once per span, so what rides along is a 4096-character preview — the same text on every span, so whichever one the server folds into the trace record reads alike. The root `agent` span carries the input in full, up to `maxValueChars`.

**It reports itself as an SDK scope.** Batches arrive under the instrumentation scope `langfuse-sdk-dsh-litefuse-plugin`. The prefix is what tells the ingest these spans are complete as sent; without it, the server assumes a generic OTel exporter that may not have populated the Langfuse attributes at all and copies the whole raw attribute map into each observation's metadata as `attributes`. That copy is redundant here — every attribute in it is already a first-class field — and unreadable, since it re-encodes the JSON strings that `model.parameters` and `usage_details` are required to be. It is the same claim `x-langfuse-ingestion-version: 4` makes on the wire.

**It writes its own log.** A booted dsh profile composes no logger plugin, so `ctx.logger` output is invisible. The file at `$DSH_HOME/litefuse.log` is where this integration reports, matching what Litefuse's other integrations do.

**No replay on load.** The plugin observes from the next `turn/start` onward. A turn already in flight when it loads is skipped rather than reconstructed, which is what keeps it from writing spans a previous process already sent.

## Known limitations

- **A delegation that shares neither a descriptor nor its prompt can mis-bind.** Concurrent calls are told apart by the child's descriptor `label` or by the `prompt` it was handed verbatim. A provider that records no descriptor *and* rewrites the prompt, or a delegation tool naming those arguments differently, falls back to the most recently started call — and with several in flight, a child can then attach to the wrong one.
- **A background one-shot delegation is not nested.** Its call reports a job id and closes, and the child may be created after that, with no candidate call left in flight; the run gets its own trace, with `agent_parent_session_id` in the root metadata. `continuable` delegations do nest, because their child exists before the call reports.
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

The workflow authenticates through npm Trusted Publishing: it presents the
GitHub OIDC identity for this repository and workflow, and npm exchanges it for
a short-lived credential. No token is stored anywhere, nothing needs rotating,
and the account's 2FA is never bypassed — provenance is attached automatically
rather than by flag.

Trusted Publishing is configured on a package that already exists, so the very
first release of a new name is the one publish that cannot use it: create the
package once by hand (`npm publish --access public`, answering the account's
2FA prompt), then configure the publisher and every later release runs
tokenless.

## Development

```bash
npm install
npm test        # 51 tests over the real @deepseek-ai/dsh-session Session
npm run typecheck
npm run build
```

The tests drive the shipped harness session store rather than a stand-in, so the assembler is exercised against the real append validation, surface rules, and derived-history projection it reads in production. `tests/plugin.spec.ts` boots the plugin against a local stand-in for the Litefuse ingest endpoint and asserts the OTLP wire form, the Basic authorization, single-write spans, and fail-open behavior on 500s, unreachable hosts, and missing credentials.

To check a real Litefuse deployment without waiting for a model round trip:

```bash
node scripts/send-verification-trace.mjs
```

It drives the real Session through a scripted turn — one tool call plus a delegated subagent run — and posts the result to `$LITEFUSE_BASE_URL`, tagged `environment: development` and named `dsh-litefuse-plugin-verify — Turn 1` so it cannot be mistaken for agent traffic. Read it back with [`litefuse-cli`](https://www.npmjs.com/package/litefuse-cli):

```bash
npx -y litefuse-cli api traces get <traceId>
```

Note that `traces list` does not project trace metadata — it reports `{}` even when the metadata is stored. Use `traces get` to see the `agent_*` fields.

## License

MIT
