/**
 * Content-block readers shared by the trace assembler.
 *
 * DeepSeek Harness messages carry provider-neutral block arrays; Litefuse
 * observations want plain text for names and status messages, structured JSON
 * for inputs and outputs, and Anthropic-style token keys for usage. Everything
 * here is a pure read over already-committed session data.
 *
 * @module dsh-litefuse/content
 */

import type { ContentBlock, TokenUsage, ToolCallBlock } from '@deepseek-ai/dsh-llm'

/** Anthropic-style usage keys, the shape Litefuse maps to cost. */
export interface UsageDetails {
  /** Uncached prompt tokens. */
  input?: number
  /** Completion tokens, reasoning included. */
  output?: number
  /** Prompt tokens served from the provider's cache. */
  cache_read_input_tokens?: number
  /** Prompt tokens written into the provider's cache. */
  cache_creation_input_tokens?: number
  /**
   * Reasoning tokens, reported as a BREAKDOWN of {@link output} rather than an
   * addition to it.
   *
   * The key deliberately contains neither `input` nor `output`: Litefuse sums
   * every key containing `input` into the input figure and every key containing
   * `output` into the output figure, so the ecosystem's `output_reasoning`
   * spelling would report these tokens twice — the provider already counts
   * them inside its completion total.
   */
  reasoning?: number
  /**
   * Billed total, supplied explicitly.
   *
   * Litefuse otherwise derives it by summing every provided key, which would
   * add {@link reasoning} on top of the completion tokens that already contain
   * it. Providing the figure keeps the breakdown visible and the total honest.
   */
  total?: number
}

/** A value serialized for a Litefuse `input`/`output` attribute. */
export interface SerializedValue {
  /** The text actually placed on the span, already truncated. */
  text: string
  /** Whether {@link text} was cut short. */
  truncated: boolean
  /** Length before truncation, in characters. */
  originalLength: number
}

/**
 * Concatenate every visible text block, ignoring reasoning and tool blocks.
 * @param blocks - the message's content blocks.
 * @returns the joined visible text, empty when the message carries none.
 */
export function visibleText(blocks: readonly ContentBlock[]): string {
  let text = ''
  for (const block of blocks) {
    if (block.type === 'text') text += block.text
  }
  return text
}

/**
 * Count reasoning characters, the signal that distinguishes a thinking step
 * from a bare one without putting reasoning in the observation name.
 * @param blocks - the message's content blocks.
 * @returns total reasoning characters across every reasoning block.
 */
export function reasoningChars(blocks: readonly ContentBlock[]): number {
  let count = 0
  for (const block of blocks) {
    if (block.type === 'reasoning') count += block.text.length
  }
  return count
}

/**
 * Collect the tool calls a step requested, in model order.
 * @param blocks - the assistant message's content blocks.
 * @returns the tool-call blocks; empty when the step requested none.
 */
export function toolCalls(blocks: readonly ContentBlock[]): ToolCallBlock[] {
  return blocks.filter((block): block is ToolCallBlock => block.type === 'tool-call')
}

/**
 * Reduce assistant content to the block structure a Litefuse generation output
 * should preserve: text, reasoning, and tool calls stay distinguishable, while
 * attachment references collapse to a type marker.
 * @param blocks - the assistant message's content blocks.
 * @returns a JSON-safe projection of the blocks, or the bare string when the message is one text block.
 */
export function assistantOutput(blocks: readonly ContentBlock[]): unknown {
  const projected = blocks.map((block) => {
    switch (block.type) {
      case 'text':
        return { type: 'text', text: block.text }
      case 'reasoning':
        return { type: 'reasoning', text: block.text }
      case 'tool-call':
        return { type: 'tool-call', id: block.id, name: block.name, arguments: block.arguments }
      default:
        return { type: block.type }
    }
  })
  if (projected.length === 1 && projected[0]?.type === 'text') return visibleText(blocks)
  return projected
}

/**
 * Map harness token accounting onto the Anthropic-style keys Litefuse prices.
 *
 * The harness reports disjoint counts — `inputTokens` excludes cache hits — so
 * the three prompt keys add up to billed input with no adjustment. Reasoning is
 * a breakdown of the completion tokens, not an addition to them, so it travels
 * under a key Litefuse's input/output classifier ignores and the billed total
 * is supplied explicitly rather than derived by summing every key.
 * @param usage - the step's accounting, when the adapter reported any.
 * @returns the usage map, or `undefined` when nothing was reported.
 */
export function usageDetails(usage: TokenUsage | undefined): UsageDetails | undefined {
  if (usage === undefined) return undefined
  const details: UsageDetails = {}
  let total = 0
  if (usage.inputTokens > 0) total += (details.input = usage.inputTokens)
  if (usage.outputTokens > 0) total += (details.output = usage.outputTokens)
  if (usage.cacheReadTokens !== undefined && usage.cacheReadTokens > 0) {
    total += (details.cache_read_input_tokens = usage.cacheReadTokens)
  }
  if (usage.cacheWriteTokens !== undefined && usage.cacheWriteTokens > 0) {
    total += (details.cache_creation_input_tokens = usage.cacheWriteTokens)
  }
  if (Object.keys(details).length === 0) return undefined
  // Assigned after the sum so the breakdown never joins the billed figure.
  if (usage.reasoningTokens !== undefined && usage.reasoningTokens > 0) {
    details.reasoning = usage.reasoningTokens
  }
  details.total = total
  return details
}

/**
 * Serialize one observation value, bounded by the configured character budget.
 * Strings pass through as themselves so a prompt is not JSON-quoted twice.
 * @param value - the value to place on an `input`/`output` attribute.
 * @param maxChars - the truncation budget in characters.
 * @returns the bounded text plus the flags a trace records when it cut one.
 */
export function serializeValue(value: unknown, maxChars: number): SerializedValue {
  let text: string
  if (typeof value === 'string') {
    text = value
  } else {
    try {
      text = JSON.stringify(value) ?? ''
    } catch {
      // Session events are JSON-validated at append, so the only way here is a
      // cyclic value introduced by a plugin's own projection.
      text = '[unserializable]'
    }
  }
  const originalLength = text.length
  if (originalLength <= maxChars) return { text, truncated: false, originalLength }
  return { text: `${text.slice(0, maxChars)}…`, truncated: true, originalLength }
}
