import type { ExtractFactsInput, FactInput, Message, SummarizeInput } from './types.js';
import { estimateTokens } from './tokens.js';
import { messageText } from './tools.js';

const MAX_LINE = 160;

export function oneLine(text: string, max = Infinity): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > max ? `${flat.slice(0, max - 3).trimEnd()}...` : flat;
}

function speaker(m: Message): string {
  return m.name ?? m.role;
}

/** Messages as a plain "role: text" transcript, one message per line. */
export function transcript(messages: Message[]): string {
  return messages.map((m) => `${speaker(m)}: ${messageText(m).trim()}`).join('\n');
}

/**
 * Dependency-free fallback summarizer: keeps the previous summary's lines and adds one clipped line per
 * folded message, dropping the oldest lines until the result fits `maxTokens`. Good enough for tests and
 * for apps that only need "what was said", not an abstract of it. Pass a model-backed `summarize` for
 * real summaries.
 */
export function extractiveSummary(input: SummarizeInput, countTokens: (text: string) => number = estimateTokens): string {
  const lines = [
    ...input.previousSummary
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean),
    ...input.messages.filter((m) => m.role !== 'system').map((m) => `- ${speaker(m)}: ${oneLine(messageText(m), MAX_LINE)}`),
  ];
  while (lines.length > 1 && countTokens(lines.join('\n')) > input.maxTokens) lines.shift();
  return lines.join('\n');
}

function factLines(facts: { key: string; value: string }[]): string {
  return facts.length ? facts.map((f) => `- ${f.key}: ${f.value}`).join('\n') : '(none)';
}

/** A prompt for a chat model that returns the updated summary as plain text. */
export function summaryPrompt(input: SummarizeInput): string {
  return [
    'You maintain a running summary of a conversation for an assistant that will continue it.',
    'Update the summary so it still covers everything important from the previous summary plus the new messages:',
    "the user's goals, decisions made, open questions, and anything the assistant promised.",
    `Write plain prose or short bullet points, at most about ${input.maxTokens} tokens.`,
    'Do not add anything that is not in the conversation. Reply with the summary only.',
    '',
    'Previous summary:',
    input.previousSummary.trim() || '(none)',
    '',
    'Known facts:',
    factLines(input.facts),
    '',
    'New messages:',
    transcript(input.messages) || '(none)',
    '',
    'Updated summary:',
  ].join('\n');
}

/** A prompt for a chat model that returns facts as `key: value` lines, which `parseFacts` reads back. */
export function factsPrompt(input: ExtractFactsInput): string {
  return [
    'Extract durable facts about the user or the task from the messages below:',
    'names, preferences, constraints, identifiers, dates, and decisions.',
    'Skip anything already in the known facts unless it changed.',
    'Reply with one fact per line as `key: value` with snake_case keys, or reply `none`.',
    '',
    'Known facts:',
    factLines(input.facts),
    '',
    'Messages:',
    transcript(input.messages) || '(none)',
  ].join('\n');
}

const LINE = /^\s*(?:[-*•]\s*)?([A-Za-z0-9][A-Za-z0-9 _.\-/]*?)\s*:\s*(.+?)\s*$/;

/** Reads facts from a model reply: a JSON array of `{key, value}`, a JSON object, or `key: value` lines. */
export function parseFacts(text: string): FactInput[] {
  const body = text.replace(/^\s*```[a-z]*\s*|\s*```\s*$/g, '').trim();
  if (!body || /^none\.?$/i.test(body)) return [];
  if (body.startsWith('[') || body.startsWith('{')) {
    try {
      const json: unknown = JSON.parse(body);
      if (Array.isArray(json)) {
        return json
          .filter((f): f is { key: unknown; value: unknown } => !!f && typeof f === 'object' && 'key' in f)
          .map((f) => ({ key: String(f.key), value: String(f.value ?? '') }));
      }
      if (json && typeof json === 'object') {
        return Object.entries(json as Record<string, unknown>).map(([key, value]) => ({ key, value: String(value ?? '') }));
      }
    } catch {
      // fall through to line parsing
    }
  }
  const out: FactInput[] = [];
  for (const raw of body.split('\n')) {
    const m = LINE.exec(raw);
    if (m) out.push({ key: m[1]!, value: m[2]! });
  }
  return out;
}
