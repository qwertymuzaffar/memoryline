import type { Fact, MemoryContext, RecalledMessage } from './types.js';
import { oneLine } from './summarize.js';

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
  return recalled.map((r) => `- ${r.message.name ?? r.message.role}: ${oneLine(r.message.content)}`).join('\n');
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

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
}

/** Messages for the OpenAI chat format: one system message carrying the memory block, then the recent window. */
export function toOpenAI(ctx: MemoryContext, options: { systemPrompt?: string } = {}): OpenAIMessage[] {
  const system = [options.systemPrompt?.trim(), ctx.system.trim()].filter(Boolean).join('\n\n');
  const out: OpenAIMessage[] = [];
  if (system) out.push({ role: 'system', content: system });
  for (const m of ctx.recent) {
    const item: OpenAIMessage = { role: m.role, content: m.content };
    if (m.name) item.name = m.name;
    out.push(item);
  }
  return out;
}

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * The Anthropic shape: a top-level system string and strictly alternating user/assistant messages.
 * System-role messages in the window join the system string, tool messages become user messages
 * tagged with the tool name, consecutive same-role messages merge, and a window that starts on an
 * assistant turn gets a one-line user message in front of it.
 */
export function toAnthropic(
  ctx: MemoryContext,
  options: { systemPrompt?: string } = {},
): { system: string; messages: AnthropicMessage[] } {
  const systemParts = [options.systemPrompt?.trim(), ctx.system.trim()].filter((s): s is string => !!s);
  const messages: AnthropicMessage[] = [];
  for (const m of ctx.recent) {
    if (m.role === 'system') {
      systemParts.push(m.content);
      continue;
    }
    const role = m.role === 'assistant' ? 'assistant' : 'user';
    const content = m.role === 'tool' ? `[${m.name ?? 'tool'}] ${m.content}` : m.content;
    const last = messages[messages.length - 1];
    if (last && last.role === role) last.content += `\n\n${content}`;
    else messages.push({ role, content });
  }
  if (messages[0]?.role === 'assistant') messages.unshift({ role: 'user', content: '(continuing the conversation)' });
  return { system: systemParts.join('\n\n'), messages };
}
