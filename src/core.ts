import type {
  Embedder,
  FactExtractor,
  FactInput,
  MemoryOptions,
  Message,
  Retriever,
  SessionState,
  Store,
  StoredMessage,
  Summarizer,
} from './types.js';

/** Resolved options shared by every session of one Memory. */
export interface Config {
  store: Store;
  maxTokens: number;
  targetTokens: number;
  keepRecent: number;
  summaryMaxTokens: number;
  perMessageOverhead: number;
  countTokens: (text: string) => number;
  summarize: Summarizer;
  extractFacts: FactExtractor | undefined;
  embed: Embedder | undefined;
  retrieve: Retriever | undefined;
  topK: number;
  minScore: number;
  recallMaxTokens: number;
  archiveMax: number;
  alignToUser: boolean;
  onArchive: MemoryOptions['onArchive'];
  clock: () => number;
}

/** The config plus the per-session lock a Session runs its work under. */
export interface Core {
  cfg: Config;
  run<T>(id: string, fn: () => Promise<T> | T): Promise<T>;
}

export function fresh(id: string, now: number): SessionState {
  return { version: 1, id, recent: [], archive: [], summary: '', facts: [], seq: 1, compactions: 0, createdAt: now, updatedAt: now };
}

export function sum(messages: StoredMessage[]): number {
  let total = 0;
  for (const message of messages) total += message.tokens;
  return total;
}

/** Copies the optional message fields that are stored as given. */
export function copyOptional(from: Message, to: Message): void {
  if (from.name !== undefined) to.name = from.name;
  if (from.toolCalls !== undefined) to.toolCalls = from.toolCalls;
  if (from.toolCallId !== undefined) to.toolCallId = from.toolCallId;
  if (from.meta !== undefined) to.meta = from.meta;
}

export function toMessage(stored: StoredMessage): Message {
  const out: Message = { role: stored.role, content: stored.content, at: stored.at };
  copyOptional(stored, out);
  return out;
}

export function normalizeFactKey(key: string): string {
  return key
    .trim()
    .toLowerCase()
    .replace(/[\s\-./]+/g, '_')
    .replace(/[^a-z0-9_]/g, '')
    .replace(/^_+|_+$/g, '');
}

export function mergeFacts(state: SessionState, inputs: FactInput[], source: 'pinned' | 'extracted', now: number): void {
  for (const input of inputs) {
    const key = normalizeFactKey(String(input?.key ?? ''));
    const value = String(input?.value ?? '').trim();
    if (!key || !value) continue;
    const existing = state.facts.find((fact) => fact.key === key);
    if (!existing) {
      state.facts.push({ key, value, source, at: now });
    } else if (source === 'pinned' || existing.source === 'extracted') {
      existing.value = value;
      existing.source = source;
      existing.at = now;
    }
  }
}
