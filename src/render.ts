import type { Fact, MemoryContext, Message, RecalledMessage } from './types.js';
import { oneLine } from './summarize.js';
import { argumentsText, messageText, parseArguments } from './tools.js';

export interface RenderOptions {
  headings?: { summary?: string; facts?: string; recalled?: string };
}

const HEADINGS = {
  summary: 'Conversation so far',
  facts: 'Known facts',
  recalled: 'Relevant earlier messages',
};

export function renderFacts(facts: Fact[]): string {
  return facts.map((f) => `- ${f.key}: ${f.value}`).join('\n');
}

export function renderRecalled(recalled: RecalledMessage[]): string {
  return recalled.map((r) => `- ${r.message.name ?? r.message.role}: ${oneLine(messageText(r.message))}`).join('\n');
}

/** Summary, facts and recalled messages as one Markdown block for a system prompt. */
export function renderMemory(
  parts: { summary: string; facts: Fact[]; recalled: RecalledMessage[] },
  options: RenderOptions = {},
): string {
  const h = { ...HEADINGS, ...options.headings };
  const sections: string[] = [];
  if (parts.summary.trim()) sections.push(`## ${h.summary}\n${parts.summary.trim()}`);
  if (parts.facts.length) sections.push(`## ${h.facts}\n${renderFacts(parts.facts)}`);
  if (parts.recalled.length) sections.push(`## ${h.recalled}\n${renderRecalled(parts.recalled)}`);
  return sections.join('\n\n');
}

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

/**
 * Messages for the OpenAI chat format: one system message carrying the memory block, then the recent
 * window. Assistant tool calls become `tool_calls` with the arguments as a JSON string, and a tool
 * message with `toolCallId` carries it as `tool_call_id`.
 */
export function toOpenAI(ctx: MemoryContext, options: { systemPrompt?: string } = {}): OpenAIMessage[] {
  const system = [options.systemPrompt?.trim(), ctx.system.trim()].filter(Boolean).join('\n\n');
  const out: OpenAIMessage[] = [];
  if (system) out.push({ role: 'system', content: system });
  for (const message of ctx.recent) {
    const item: OpenAIMessage = { role: message.role, content: message.content };
    if (message.name) item.name = message.name;
    if (message.role === 'assistant' && message.toolCalls?.length) {
      item.tool_calls = message.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: argumentsText(call) },
      }));
    }
    if (message.role === 'tool' && message.toolCallId) item.tool_call_id = message.toolCallId;
    out.push(item);
  }
  return out;
}

export type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string };

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  /** A plain string until a turn carries tool blocks; then an array of content blocks. */
  content: string | AnthropicContentBlock[];
}

/** One message in the Anthropic shape: a string for plain text, blocks when tool calls or results are involved. */
function anthropicContent(message: Message): string | AnthropicContentBlock[] {
  if (message.role === 'assistant' && message.toolCalls?.length) {
    const blocks: AnthropicContentBlock[] = [];
    if (message.content.trim()) blocks.push({ type: 'text', text: message.content });
    for (const call of message.toolCalls) blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: parseArguments(call) });
    return blocks;
  }
  if (message.role === 'tool') {
    if (message.toolCallId) return [{ type: 'tool_result', tool_use_id: message.toolCallId, content: message.content }];
    return `[${message.name ?? 'tool'}] ${message.content}`;
  }
  return message.content;
}

function toBlocks(content: string | AnthropicContentBlock[]): AnthropicContentBlock[] {
  if (typeof content !== 'string') return content;
  return content ? [{ type: 'text', text: content }] : [];
}

/** Joins two same-role turns: strings stay strings, anything with blocks becomes one block array. */
function mergeContent(existing: string | AnthropicContentBlock[], next: string | AnthropicContentBlock[]): string | AnthropicContentBlock[] {
  if (typeof existing === 'string' && typeof next === 'string') return `${existing}\n\n${next}`;
  return [...toBlocks(existing), ...toBlocks(next)];
}

/**
 * The Anthropic shape: a top-level system string and strictly alternating user/assistant messages.
 * System-role messages in the window join the system string, assistant tool calls become `tool_use`
 * blocks, tool messages with `toolCallId` become `tool_result` blocks in the following user turn,
 * tool messages without one become user text tagged with the tool name, consecutive same-role
 * messages merge, and a window that starts on an assistant turn gets a one-line user message in
 * front of it.
 */
export function toAnthropic(
  ctx: MemoryContext,
  options: { systemPrompt?: string } = {},
): { system: string; messages: AnthropicMessage[] } {
  const systemParts = [options.systemPrompt?.trim(), ctx.system.trim()].filter((part): part is string => !!part);
  const messages: AnthropicMessage[] = [];
  for (const message of ctx.recent) {
    if (message.role === 'system') {
      systemParts.push(message.content);
      continue;
    }
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    const content = anthropicContent(message);
    const last = messages[messages.length - 1];
    if (last && last.role === role) last.content = mergeContent(last.content, content);
    else messages.push({ role, content });
  }
  if (messages[0]?.role === 'assistant') messages.unshift({ role: 'user', content: '(continuing the conversation)' });
  return { system: systemParts.join('\n\n'), messages };
}
