/**
 * Observation naming, per the Litefuse agent-trace spec v1.2 §3.
 *
 * Names are filterable identifiers, not UI titles: all lowercase, describing
 * the step's primary action rather than the model that performed it, with one
 * short parenthetical of key information and the container-local step number.
 * The trace and root-container name is the one capitalized exception — it is a
 * title.
 *
 * @module dsh-litefuse-plugin/naming
 */

/** Longest key-info fragment allowed inside an observation name. */
export const KEY_INFO_MAX_CHARS = 24

/** Shell words that precede the executable a bash command actually runs. */
const SHELL_PREFIXES = new Set(['cd', 'export', 'sudo', 'time', 'env', 'exec', 'nohup', 'command', 'then', 'do'])

/** Argument fields that identify a filesystem-shaped tool call, in preference order. */
const PATH_FIELDS = ['file_path', 'path', 'notebook_path', 'target'] as const

/** Argument fields that identify a pattern-shaped tool call, in preference order. */
const PATTERN_FIELDS = ['pattern', 'query', 'regex'] as const

/**
 * Clip one key-info fragment to {@link KEY_INFO_MAX_CHARS}.
 * @param value - the raw fragment.
 * @returns the fragment, ellipsized when it exceeds the bound.
 */
function clip(value: string): string {
  return value.length <= KEY_INFO_MAX_CHARS ? value : `${value.slice(0, KEY_INFO_MAX_CHARS - 1)}…`
}

/**
 * Read one string field out of parsed tool arguments.
 * @param args - the tool arguments, already parsed from the model's JSON.
 * @param field - the field to read.
 * @returns the trimmed value, or `undefined` when the field is absent or not a non-empty string.
 */
function stringField(args: Readonly<Record<string, unknown>>, field: string): string | undefined {
  const value = args[field]
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed.length === 0 ? undefined : trimmed
}

/**
 * Reduce a path to its basename — the full path is too long for a name and the
 * observation input carries it in full anyway.
 * @param path - a filesystem path.
 * @returns the last non-empty segment.
 */
function basename(path: string): string {
  const segments = path.replace(/[/\\]+$/, '').split(/[/\\]/)
  return segments[segments.length - 1] ?? path
}

/**
 * Extract the executable a shell command runs, skipping environment
 * assignments, flags, quoted arguments, paths, and shell prefixes.
 * @param command - the command line as the model wrote it.
 * @returns the executable name, or an empty string when nothing recognizable leads the command.
 */
export function shellExecutable(command: string): string {
  for (const line of command.split('\n')) {
    for (const word of line.split(/[\s;|&()]+/)) {
      if (word.length === 0) continue
      if (word.includes('=') || word.startsWith('-') || word.startsWith('"') || word.startsWith('\'')) continue
      if (SHELL_PREFIXES.has(word)) continue
      if (/^[A-Za-z][\w.+-]*$/.test(word)) return clip(word)
    }
  }
  return ''
}

/**
 * Build the short parenthetical that makes a tool observation navigable.
 * Falls back to the most identifying string field so a tool this integration
 * has never heard of still gets useful key info instead of a bare name.
 * @param toolName - the model-facing tool name.
 * @param args - the parsed tool arguments.
 * @returns the key-info fragment, or an empty string when nothing identifies the call.
 */
export function toolKeyInfo(toolName: string, args: unknown): string {
  if (args === null || typeof args !== 'object' || Array.isArray(args)) return ''
  const fields = args as Record<string, unknown>
  switch (toolName) {
    case 'bash':
    case 'pwsh': {
      const command = stringField(fields, 'command')
      return command === undefined ? '' : shellExecutable(command)
    }
    case 'web_fetch': {
      const url = stringField(fields, 'url')
      if (url === undefined) return ''
      try {
        return clip(new URL(url).hostname)
      } catch {
        // A model-written URL that does not parse is still worth showing raw.
        return clip(url)
      }
    }
    case 'todo_write': {
      const todos = fields['todos']
      return Array.isArray(todos) ? `${todos.length} items` : ''
    }
    default:
      break
  }
  for (const field of PATH_FIELDS) {
    const value = stringField(fields, field)
    if (value !== undefined) return clip(basename(value))
  }
  for (const field of PATTERN_FIELDS) {
    const value = stringField(fields, field)
    if (value !== undefined) return clip(value)
  }
  const name = stringField(fields, 'name') ?? stringField(fields, 'description')
  return name === undefined ? '' : clip(name)
}

/**
 * Name one tool observation: `tool: <name> (<key info>) #N`, or the delegation
 * form `tool (n subagents) #N` when the call spawned subagent containers.
 * @param toolName - the model-facing tool name.
 * @param args - the parsed tool arguments, used for key info.
 * @param stepIndex - the container-local step number.
 * @param subagentCount - how many subagent containers mounted under this call.
 * @returns the observation name.
 */
export function toolObservationName(
  toolName: string,
  args: unknown,
  stepIndex: number,
  subagentCount: number,
): string {
  if (subagentCount > 0) {
    return `tool (${subagentCount} ${subagentCount === 1 ? 'subagent' : 'subagents'}) #${stepIndex}`
  }
  const info = toolKeyInfo(toolName, args)
  return info.length === 0 ? `tool: ${toolName} #${stepIndex}` : `tool: ${toolName} (${info}) #${stepIndex}`
}

/**
 * Name one `run_code` sub-dispatch: `tool: <name> (<key info>)`.
 *
 * Deliberately carries no `#N`. The step number is the container's flat
 * sequence of model calls and the tool executions they requested; a dispatch
 * belongs to neither — it is work nested inside one of those steps.
 * @param toolName - the tool the program invoked.
 * @param args - the dispatched arguments, used for key info.
 * @returns the observation name.
 */
export function dispatchObservationName(toolName: string, args: unknown): string {
  const info = toolKeyInfo(toolName, args)
  return info.length === 0 ? `tool: ${toolName}` : `tool: ${toolName} (${info})`
}

/** What an assistant step primarily did, which is what its name states. */
export interface GenerationShape {
  /** How many tool calls the step requested. */
  toolCallCount: number
  /** Whether the step produced visible text. */
  hasText: boolean
  /** Whether the step produced reasoning content. */
  hasReasoning: boolean
}

/**
 * Name one generation by what the model did. A step that requests tools is a
 * `plan` even when it also carries reasoning or transitional text: the name
 * states the primary action, and combinatorial names would break aggregation.
 * The closing answer is `response` with no number — at most one per container,
 * so evaluators can find it by name.
 * @param shape - what the step produced.
 * @param stepIndex - the container-local step number.
 * @param inSubagent - whether this generation belongs to a subagent container.
 * @returns the observation name.
 */
export function generationObservationName(
  shape: GenerationShape,
  stepIndex: number,
  inSubagent: boolean,
): string {
  if (shape.toolCallCount > 0) {
    return `plan (${shape.toolCallCount} ${shape.toolCallCount === 1 ? 'tool' : 'tools'}) #${stepIndex}`
  }
  if (shape.hasText) return inSubagent ? 'subagent response' : 'response'
  if (shape.hasReasoning) return `think #${stepIndex}`
  return `generation #${stepIndex}`
}

/**
 * Name one trace and its root container: the only capitalized name, because it
 * is a title rather than a filter key.
 * @param agentName - the deployment's display name for this agent.
 * @param turn - the session's own turn number, which continues across resume.
 * @returns the trace name.
 */
export function traceName(agentName: string, turn: number): string {
  return `${agentName} — Turn ${turn}`
}
