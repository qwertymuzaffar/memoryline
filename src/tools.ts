import type { Message, ToolCall } from './types.js';

/** Tool call arguments as a JSON string, however they were given. */
export function argumentsText(call: ToolCall): string {
  return typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments);
}

/**
 * Tool call arguments as an object. A JSON string is parsed; an empty string is `{}`; a string that
 * is not a JSON object is kept under `arguments` so nothing is lost.
 */
export function parseArguments(call: ToolCall): Record<string, unknown> {
  if (typeof call.arguments !== 'string') return call.arguments;
  if (!call.arguments.trim()) return {};
  try {
    const parsed: unknown = JSON.parse(call.arguments);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
  } catch {
    // not JSON: fall through
  }
  return { arguments: call.arguments };
}

/** Message content with any tool calls appended as text, for transcripts, summaries and token counts. */
export function messageText(message: Message): string {
  if (!message.toolCalls?.length) return message.content;
  const calls = message.toolCalls.map((call) => `[call ${call.name}(${argumentsText(call)})]`).join(' ');
  return message.content.trim() ? `${message.content} ${calls}` : calls;
}

export interface ToolGroup {
  /** Index of the assistant message that made the calls. */
  start: number;
  /** One past the last tool result that answers them. */
  end: number;
}

/**
 * The call/result group that contains the message at `index`, or null when it is not part of one.
 * A group is an assistant message with tool calls plus the run of tool messages right after it that
 * answer those calls by id.
 */
export function toolGroupAt(messages: readonly Message[], index: number): ToolGroup | null {
  const message = messages[index];
  if (!message) return null;
  let start = index;
  if (message.role === 'tool' && message.toolCallId) {
    let cursor = index - 1;
    while (cursor >= 0 && messages[cursor]!.role === 'tool' && messages[cursor]!.toolCallId) cursor--;
    const caller = messages[cursor];
    if (!caller || caller.role !== 'assistant' || !caller.toolCalls?.some((call) => call.id === message.toolCallId)) return null;
    start = cursor;
  } else if (!(message.role === 'assistant' && message.toolCalls?.length)) {
    return null;
  }
  const ids = new Set(messages[start]!.toolCalls!.map((call) => call.id));
  let end = start + 1;
  while (end < messages.length) {
    const candidate = messages[end]!;
    if (candidate.role !== 'tool' || !candidate.toolCallId || !ids.has(candidate.toolCallId)) break;
    end++;
  }
  return { start, end };
}
