/**
 * Content-block readers shared by the trace assembler.
 *
 * DeepSeek Harness messages carry provider-neutral block arrays; Litefuse
 * observations want plain text for names and status messages, structured JSON
 * for inputs and outputs, and Anthropic-style token keys for usage. Everything
 * here is a pure read over already-committed session data.
 *
 * @module dsh-litefuse-plugin/content
 */

import type { ContentBlock, TokenUsage, ToolCallBlock } from '@deepseek-ai/dsh-llm'

/** Usage keys in the spelling Litefuse's classifier and price table expect. */
export interface UsageDetails {
  /** Uncached prompt tokens. */
  input?: number
  /**
   * Completion tokens EXCLUDING reasoning, which travels as its own sibling
   * key. This is the ecosystem's split, not a choice: Litefuse sums every key
   * containing `output` into the output figure, and its own ingestion
   * processor normalizes provider payloads by subtracting reasoning out of
   * `output` for exactly that reason.
   */
  output?: number
  /** Prompt tokens served from the provider's cache. */
  cache_read_input_tokens?: number
  /** Prompt tokens written into the provider's cache. */
  cache_creation_input_tokens?: number
  /**
   * Reasoning tokens, the other half of the completion total. The shipped
   * price table prices this key beside `output`, so splitting the two keeps
   * both the displayed breakdown and the computed cost right.
   */
  output_reasoning_tokens?: number
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
 * Map harness token accounting onto the keys Litefuse prices and displays.
 *
 * The harness reports disjoint prompt counts — `inputTokens` excludes cache
 * hits — so the three prompt keys add up to billed input with no adjustment.
 * Completion is the opposite: `outputTokens` INCLUDES reasoning, while Litefuse
 * expects the two as siblings that sum to the completion total, so reasoning is
 * subtracted back out of `output` here.
 * @param usage - the step's accounting, when the adapter reported any.
 * @returns the usage map, or `undefined` when nothing was reported.
 */
export function usageDetails(usage: TokenUsage | undefined): UsageDetails | undefined {
  if (usage === undefined) return undefined
  const details: UsageDetails = {}
  const reasoning = usage.reasoningTokens ?? 0
  if (usage.inputTokens > 0) details.input = usage.inputTokens
  // Clamped: a provider that reported reasoning larger than its own completion
  // total would otherwise produce a negative token count.
  const content = Math.max(usage.outputTokens - reasoning, 0)
  if (content > 0) details.output = content
  if (reasoning > 0) details.output_reasoning_tokens = reasoning
  if (usage.cacheReadTokens !== undefined && usage.cacheReadTokens > 0) {
    details.cache_read_input_tokens = usage.cacheReadTokens
  }
  if (usage.cacheWriteTokens !== undefined && usage.cacheWriteTokens > 0) {
    details.cache_creation_input_tokens = usage.cacheWriteTokens
  }
  return Object.keys(details).length === 0 ? undefined : details
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
