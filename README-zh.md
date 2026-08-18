# dsh-litefuse-plugin

[English](README.md) | 中文

DeepSeek Harness [Litefuse](https://litefuse.ai) 插件，用于 Agent 可观测性与评测（Agent Observability and Evals）。

每一轮用户对话变成一条 trace：一个 `agent` 根节点、每次模型调用一个 `generation`（带真实延迟与 token 用量）、每次工具执行一个 `tool`，以及该轮委派出去的每个子 agent 各自一个嵌套容器。实现遵循 [Litefuse agent-trace 规范 v1.2](https://litefuse.ai/litefuse-agent-trace-spec.md)。

```
DeepSeek Harness — Turn 3            AGENT      1.9s   input: "why is the build failing?"
├── plan (2 tools) #1                GENERATION 820ms  in 1.2k · out 96 · cache-read 18k
├── tool: bash (pnpm) #2             TOOL       410ms
├── tool: read (tsconfig.json) #3    TOOL       12ms
├── plan (1 tool) #4                 GENERATION 640ms
├── tool (1 subagent) #5             TOOL       9.4s
│   └── subagent                     AGENT      8.8s   ← 委派开销：0.6s
│       ├── plan (1 tool) #1         GENERATION        ← 每个容器内编号重新从 #1 开始
│       ├── tool: grep (TS2345) #2   TOOL
│       └── subagent response        GENERATION
└── response                         GENERATION 1.1s   output: 最终回答
```

## 安装

```bash
npx @deepseek-ai/dsh plugin --profile web add -w dsh-litefuse-plugin
```

把项目的密钥对放进 `~/.dsh/.env`：

```
LITEFUSE_PUBLIC_KEY=pk-lf-…
LITEFUSE_SECRET_KEY=sk-lf-…
```

密钥在 Litefuse 项目里的 **Settings → API Keys → Create new API keys** 生成（注册入口 <https://litefuse.cloud/auth/sign-up>）。

然后照常启动：

```bash
npx @deepseek-ai/dsh web
```

发一条消息，打开 <https://litefuse.cloud> → 你的项目 → **Tracing**。trace 在该轮**第一个** span 完成时就会出现，不必等整轮结束。

### 卸载

```bash
npx @deepseek-ai/dsh plugin --profile web remove -w dsh-litefuse-plugin
```

`DSH_LITEFUSE_DISABLED=1` 可以在不卸载的情况下关闭导出。

## 配置

安装时写入的 patch 全部从环境变量读取，所以多数部署无需配置。要覆盖的话，在 `~/.dsh/profiles/<name>/cordis.patch.yml` 里加一条：

```yaml
- id: litefuse
  config:
    environment: staging
    agentName: My Agent
    requestInput: delta
    tags: [dsh, team-platform]
```

按 id 定位的 patch 会**整体替换** `config`，所以要保留的字段必须一并重写。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enabled` | `true` | `false` 时插件照常挂载但不导出任何东西 |
| `baseUrl` | `https://litefuse.cloud` | Litefuse 端点；OTLP trace 路径会追加在其后 |
| `publicKeyEnv` / `secretKeyEnv` | `LITEFUSE_PUBLIC_KEY` / `LITEFUSE_SECRET_KEY` | 凭据**引用名**；`LANGFUSE_*` 作为兜底 |
| `environment` | `production` | 写在每个 span 上的 Litefuse tracing environment |
| `agentName` | `DeepSeek Harness` | `<agent> — Turn N` 里的名字 |
| `userId` | `$USER` | trace 级的 `user.id` |
| `tags` | `[dsh]` | trace 标签，与自动生成的 `model:<name>` 并列 |
| `release` | — | 可选的发布标识，写在每条 trace 上 |
| `requestInput` | `full` | `full` 记录完整请求；`delta` 只记录上次调用后新增的消息；`none` 不记录输入 |
| `maxValueChars` | `1000000` | 单个 input/output 的截断预算，**包括 trace input** |
| `delegationTools` | `[subagent, subagent_fork]` | 哪些工具的在途调用可以承载子 agent 容器 |
| `exportDelayMillis` | `1000` | 一个已结束的 span 等待同伴凑批的时长 |
| `requestTimeoutMillis` | `10000` | 单次请求的截止时间 |
| `shutdownTimeoutMillis` | `3000` | 卸载时排空缓冲的外层时限 |
| `logFile` | `$DSH_HOME/litefuse.log` | 本集成自己的日志 |
| `debug` | `$DSH_LITEFUSE_DEBUG` | 是否保留 verbose 日志行 |

凭据是**引用，不是值**：配置里写的是环境变量名，值来自 harness 的凭据存储（`~/.dsh/.credentials.yaml`，挂载了的话），否则来自 `~/.dsh/.env` 注入的进程环境。这里从不把密钥写进任何文件，日志里也只会出现 public key 的前十个字符。

## 工作原理

**零运行时依赖** —— 不引入 Langfuse SDK。span 直接发往 OTLP 端点，并声明 `x-langfuse-ingestion-version: 4`，这是官方文档给自定义 exporter 的显式选项，含义是"span 一次写全、发出即完整"。

与 Litefuse 为其他 agent 提供的**读取日志文件**式采集器不同，这个插件**在进程内**运行于 harness 自己的 session 事件流之上。因此它记录的是**实际发生的事**，而不是从一份转录文本里能反推出来的事：真实的单次调用延迟、首 token 时间、工具耗时、互不重叠的缓存 token 记账，以及每个 generation 实际被发送的请求。

插件订阅 `session/event`（harness 提交后的追加事件流），把每一轮的事件折叠成 span：

| Session 事件 | 变成 |
|---|---|
| `turn/start` … `turn/end` | 这条 trace 及其 `agent` 根 span |
| `user/message`（`source.kind: user`） | trace 的输入 |
| `step/start` … `assistant/message` | 一个 `generation`，按模型这一步做了什么来命名 |
| `assistant/chunk`（一步中的第一个） | 该 generation 的 `completion_start_time` —— 首 token 时间 |
| `tool/call` … `tool/result` | 一个 `tool` span，通过 `agent_plan_step` 关联回它的 plan |
| `tool/code-dispatch-start` … `tool/code-dispatch` | `run_code` 程序每发起一次调用，一个嵌套 `tool` span |
| `request/header`、`request/context` | 模型名、采样参数、上下文窗口 |
| `compaction/end` | 一个 `context compaction` 事件，用来解释下次调用 token 数为何骤降 |
| `subagent/descriptor` | 某个子运行属于哪次委派调用，以及那次调用是否等待它 |
| 匹配到某次委派调用的子会话 | 挂在该调用 tool span 下的 `subagent` 容器 |

span 在各自容器内是**扁平**排布的；唯一的层级来自真实的子 agent 运行。generation 和 tool 共享同一个步骤计数器，所以 `#N` 是一条统一的时间序列，而 `tool.agent_plan_step == generation.agent_step_index` 能把工具关联回请求它的那次模型调用。

每个 span **只在结束时写出一次** —— OTel span 是不可变的，所以一个在途的步骤在它关闭前是刻意不可见的。trace 级属性会随每个 span 一起发送，这正是 trace 能先于根节点出现的原因。

### Code mode

`run_code` 程序会直接调用工具，这些调用**不会**出现为 `tool/call`，否则整个程序的工作会塌缩成一个不透明的 span。它们作为嵌套 `tool` span 挂在运行它们的那次调用之下 —— 桥接层会在父调用返回前排空所有在途派发，所以这个包含关系是**结构性成立**的，而非推断出来的。它们不带 `#N`：步骤计数器数的是模型的调用以及模型请求的工具，而一次派发两者都不是。因此 `agent_tool_calls` 仍然只统计模型主动请求的部分，另有 `agent_code_dispatches` 与之并列。

### 子 agent

一次委派运行在 harness 里是**独立的 session**。当它在父会话的某次委派调用在途时启动，子会话的各步骤就挂进一个 `subagent` 容器，该容器以那次调用的 tool span 为父节点，编号重新从 #1 开始，收尾回答命名为 `subagent response`，其 token 用量汇总进父 trace 的总量。tool span 与容器之间的时间差，就是委派的真实开销。

**子运行属于哪次调用，由身份判定，且只判定一次。** harness 不会给子会话任何指向源调用的引用 —— 它只有一个全新的 session id，和一个仅指明父 session 的 header。但 harness 确实原样传递了两样东西：委派的 `description`（作为子会话自己 `subagent/descriptor` 的 `label`），以及 `prompt`（作为子会话的第一条 user message）。两者中任何一个都是精确的键，而正是它们把并发的多次委派区分开 —— 并发是**常态**而非边缘情况，因为 subagent 工具自己的提示词就要求模型把相互独立的委派放在同一条 assistant 消息里一起发起。

span 在写出的那一刻就带上了它的 trace 和父节点，而 OTel span 不可变，所以一个之后还可能改变的绑定，等于是一个已经说了谎的绑定。因此规则是：**在证据足以"确定"答案的最早时刻绑定** —— 要么精确命中一个键，要么候选调用只有一个 —— 并且**永不修改**。等待并非没有代价：`continuable` 类型的委派在启动子会话后**几毫秒内**就会返回它的 tool result，早于子会话自己的 prompt 被记录，所以一个推迟到那时才做的决定会发现已经没有候选了。descriptor 正是让这个"提前决定"成为精确判断而非猜测的东西；而对 spawn 出来的子会话，它位于**构造种子**中，而 `session/event` 事件流从不重播种子 —— 所以要直接从 session 日志里读取。

**委派调用返回，不等于运行结束。** one-shot 委派会带着子会话的答案返回，所以父会话的 `tool/result` 就是容器的终点。而 `continuable` 委派在"受理"那一刻就返回，整个运行还在后面；在那里关闭容器会封上一个空壳，并丢弃子会话之后做的一切。这类容器改为在子会话自己的 `turn/end` 时关闭，并把 token 汇总进当初发起委派的那一轮。

只有 `delegationTools` 里点名的工具才能承载容器。一个恰好在途的普通调用**刻意不作为兜底**：后台委派会在子会话启动前就返回结果，所以"接受任意调用"会把一整个 agent 运行 —— 连同它的 token —— 记到父会话当时碰巧在跑的某个无关工具名下。找不到在途委派的子会话会独立成一条 trace，并在根节点 metadata 里带上 `agent_parent_session_id`。仅仅设置了 `parentSession` 也不够：只有 harness 自己的 `origin: subagent` 分类才能让一个 session 进入父 trace，所以普通的 fork 会被挡在外面。

### Metadata

所有 metadata 统一放在单一的 `agent_` 前缀下 —— 绝不使用按 agent 划分的命名空间 —— 这样同一个 Litefuse 看板查询就能横跨所有 agent 集成。缺失的字段直接省略，而不是填 null。

- **根节点与子 agent 容器**：`agent_turn_number`、`agent_session_id`、`agent_parent_session_id`、`agent_cwd`、`agent_provider`、`agent_model`、`agent_api_calls`、`agent_tool_calls`、`agent_steps`、`agent_duration_ms`、`agent_context_window`、`agent_end_reason`、`agent_subagent`、`agent_code_dispatches`，以及 token 汇总 `agent_input_tokens` / `agent_output_tokens` / `agent_cache_read_tokens` / `agent_cache_write_tokens` / `agent_reasoning_tokens` / `agent_total_tokens` / `agent_accounted_generations`
- **Generation**：`agent_step_index`、`agent_api_duration_ms`、`agent_time_to_first_token_ms`、`agent_tool_call_count`、`agent_thinking_chars`、`agent_reasoning_tokens`、`agent_input_scope`，以及截断标记
- **Tool**：`agent_tool_name`、`agent_tool_call_id`、`agent_step_index`、`agent_plan_step`、`agent_duration_ms`、`agent_is_error`、`agent_error_code`、`agent_subagent_count`，以及截断标记
- **Code dispatch**：`agent_code_dispatch`（恒为 `true`）、`agent_tool_name`、`agent_tool_call_id`、`agent_parent_call_id`、`agent_root_call_id`、`agent_duration_ms`、`agent_is_error`，以及截断标记

metadata 以**逐键的 span 属性**形式发送（`langfuse.observation.metadata.agent_step_index`），而不是打包成一个序列化字符串。因为 JSON 字符串会被原样存储在它的解析副本旁边，导致原始属性集里出现"字符串里套 JSON"的情况；而且 trace 规范明确禁止把预序列化的 JSON 作为 metadata 值 —— 服务端展平它时会破坏转义。

token 计数使用 Litefuse 用来计价和分类的那组键：`input`、`output`、`output_reasoning_tokens`、`cache_read_input_tokens`、`cache_creation_input_tokens`。harness 给出的 prompt 计数是**互不重叠**的，所以三个 prompt 键相加就等于计费 input，无需调整。completion 则相反 —— harness 把 reasoning 计入了 `outputTokens` —— 所以这里把 reasoning 从 `output` 中减出来，作为兄弟键单独上报。Litefuse 会把所有含 `output` 的键相加作为展示的 Output 数字，它自己的摄取处理器也正是这样归一化各家 provider 的载荷，所以这个拆分同时保证了明细和成本都正确。模型定价定义里应当把 `output_reasoning_tokens` 与 `output` 一并计价；官方内置的价格表对它已知的每个推理模型都已如此。

**token 汇总只进 metadata。** 一个 `agent` span —— 无论是轮次根节点还是子 agent 容器 —— 会以 `agent_*_tokens` 携带它自身及其下所有嵌套内容的总量，但**绝不**写进 `usage_details`。因为 Litefuse 是靠累加各个 span 来给 trace 计价的，容器若再声明一遍子会话的 token，账单就会翻倍。想知道"这一轮有多大"看根节点的 `agent_total_tokens`；想知道"花了多少钱"看 `totalCost`。

## 排查

什么都没收到？带 verbose 日志启动：

```bash
DSH_LITEFUSE_DEBUG=1 npx @deepseek-ai/dsh web
```

然后读本集成自己的日志：

```bash
tail -5 ~/.dsh/litefuse.log
```

安装正常的话，启动时会打印端点，每完成一轮打印一行，每投递一批打印一行：

```
[info] v0.1.1 exporting to https://litefuse.cloud (key pk-lf-a1b2…)
[debug] turn closed "DeepSeek Harness — Turn 1" trace=… session=… steps=3 api=2 tools=1 duration=4120ms
[debug] sent 4 span(s) -> https://litefuse.cloud/api/public/otel/v1/traces HTTP 200
```

没收到的话，那份日志会写明原因 —— 缺凭据、某个 HTTP 状态码、或传输错误。UI 里 `totalCost` 显示 0 **不是**采集问题：那表示项目里没有你所用模型的价格条目（**Settings → Models**）。

## 设计说明

**永远 fail-open。** 每个处理函数都是自包含的，任何失败都被记录并吞掉。session 的事件分发会在遇到抛异常的监听器时中断，所以从这个插件逃逸出去的异常会**饿死**在它之后注册的每一个观察者。一个连不上的 Litefuse 项目，代价仅仅是一次请求超时，别无其他。

**零运行时依赖。** 构建产物除了 `node:` 内置模块和自身文件之外不 import 任何东西 —— 所有 harness 和 Cordis 的 import 都是 type-only，并且有 CI 把关这条性质。这是刻意为之：一个安装进 profile 的插件如果引入了自己那份 `@deepseek-ai/cordis`，宿主就会同时存在两个不同的 `Context` 类，服务装配会以极难排查的方式失败。

**trace 头信息会重复发送，所以必须保持小。** 每个 span 都带上 trace 的名字、标签、session 和 user，这样 trace 在根节点写出之前就可查询。输入是这组头信息里唯一没有天然大小上限的字段，重复发送等于把一次粘贴的大文件按 span 数量计费一遍，所以随行的是一段 **4096 字符的预览** —— 每个 span 上的这段文本完全相同，因此服务端无论折叠哪一个进 trace 记录，读到的都一样。根 `agent` span 则携带完整输入，上限为 `maxValueChars`。

**scope 写的是本包，不是某个 SDK。** 批次以 instrumentation scope `dsh-litefuse-plugin` 的身份到达。摄取端会把 span 的整份原始属性表复制进每个 observation 的 metadata（字段名 `attributes`），除非 scope 名以 `langfuse-sdk` 开头；顶着那个前缀确实能抑制这份拷贝，曾有一个版本这么做过，但 scope 是身份字段，而本插件并不是 Langfuse SDK。这份拷贝的代价是噪音而非正确性 —— 里面每个属性都已经是一等字段，没有任何数据丢失或出错，只是被重复了一遍，而且形式上把 `model.parameters` 和 `usage_details` 按规范必须是 JSON 字符串的那些值又编码了一层。准确的修法在摄取端：它本来就收到了本 exporter 发出的 `x-langfuse-ingestion-version: 4` —— 那才是「客户端发出即完整 span」的文档化信号，而且它对任何做出该声明的自定义 exporter 都成立，不像名字前缀那样只认一个。

**它写自己的日志。** 启动后的 dsh profile 不会组合任何 logger 插件，所以 `ctx.logger` 的输出是看不见的。`$DSH_HOME/litefuse.log` 才是这个集成汇报的地方，这也与 Litefuse 其他集成的做法一致。

**加载时不回放。** 插件从下一个 `turn/start` 开始观察。加载时已在进行中的那一轮会被跳过，而不是被重建 —— 这正是它不会重复写出前一个进程已经发送过的 span 的原因。

## 已知限制

- **既不共享 descriptor 也不共享 prompt 的委派可能绑错。** 并发调用靠子会话的 descriptor `label` 或原样传递的 `prompt` 区分。如果某个 provider 既不记录 descriptor **又**改写了 prompt，或者某个委派工具给这些参数起了别的名字，就会退化为"取最近启动的那次调用" —— 此时若有多次在途，子会话可能挂错。
- **后台 one-shot 委派不会嵌套。** 它的调用返回一个 job id 就结束了，子会话可能在那之后才创建，届时已没有候选调用在途；该运行会独立成一条 trace，根节点 metadata 带 `agent_parent_session_id`。`continuable` 委派则能嵌套，因为它的子会话在调用返回前就已存在。
- **跨进程的子 agent 各自成 trace。** 在另一个进程里运行子会话的 provider（`acp`、`codex`、各 SDK provider）不会把 session 事件发布到本进程，所以它们的运行不构成子树。进程内的 provider —— 也就是内置 `subagent` 和 `subagent_fork` 工具所用的 —— 可以。
- **并行的 tool span 可能高估耗时。** harness 按模型给出的顺序提交工具结果，所以一个很快完成、但排在慢兄弟之后的调用，记录的是它**结果被提交**的时间戳，而非它自身完成的时刻。
- **`requestInput: full` 每次模型调用都要折叠一遍派生历史。** 那是每次调用一趟 session 日志遍历 —— 相比一次模型往返可以忽略，但超长会话可以改用 `delta`。
- **没有持久化发件箱。** 进程死亡时仍在缓冲区里的 span 会丢失。投递按设计是 at-most-once；session 日志才是持久记录。

## 发布

发布就是推一个 tag；CI 会构建、测试，并带 provenance 发布。

```bash
npm version patch        # 或 minor / major —— 会写 package.json 并打 tag
git push --follow-tags
```

workflow 会拒绝与 `package.json` 不一致的 tag；`prepack` 在打包前构建，所以 tarball 里始终带着 `lib/`，安装方永远不需要编译任何东西。

workflow 通过 npm **Trusted Publishing** 认证：它出示本仓库与本 workflow 的 GitHub OIDC 身份，npm 用它换取一个短期凭据。没有任何 token 被存储，不需要轮换，账号的 2FA 也从不被绕过 —— provenance 是自动附加的，而不是靠命令行参数。

Trusted Publishing 只能在**已存在的包**上配置，所以一个新包名的首次发布是唯一无法使用它的那一次：先手动发一次把包创建出来（`npm publish --access public`，按提示输入账号的 2FA），配置好 publisher 之后，此后每次发布都无需 token。

## 开发

```bash
npm install
npm test        # 51 个测试，跑在真实的 @deepseek-ai/dsh-session Session 上
npm run typecheck
npm run build
```

测试驱动的是 harness 实际发布的 session store，而不是替身，所以 assembler 是在它生产环境真正读取的那套 append 校验、surface 规则和派生历史投影之上被检验的。`tests/plugin.spec.ts` 会把插件启动起来，对接一个本地的 Litefuse 摄取端点替身，断言 OTLP 线格式、Basic 认证、span 单次写出，以及在 500、主机不可达、缺凭据这三种情况下的 fail-open 行为。

想在不等待模型往返的情况下检查一个真实的 Litefuse 部署：

```bash
node scripts/send-verification-trace.mjs
```

它会驱动真实的 Session 走完一轮脚本化的对话 —— 一次工具调用加一次委派子 agent 运行 —— 并把结果投递到 `$LITEFUSE_BASE_URL`，标记为 `environment: development`，命名为 `dsh-litefuse-plugin-verify — Turn 1`，以免与真实 agent 流量混淆。用 [`litefuse-cli`](https://www.npmjs.com/package/litefuse-cli) 读回：

```bash
npx -y litefuse-cli api traces get <traceId>
```

注意 `traces list` 不会投影 trace metadata —— 即使 metadata 已存储，它也报告 `{}`。要看 `agent_*` 字段请用 `traces get`。

## 许可

MIT
