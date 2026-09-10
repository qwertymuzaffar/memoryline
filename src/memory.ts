import type {
  AddResult,
  ArchivedMessage,
  CompactResult,
  ContextOptions,
  Embedder,
  FactExtractor,
  FactInput,
  MemoryContext,
  MemoryOptions,
  Message,
  RecalledMessage,
  Retriever,
  SessionState,
  Store,
  StoredMessage,
  Summarizer,
  ToolCall,
} from './types.js';
import { estimateTokens } from './tokens.js';
import { messageText, toolGroupAt } from './tools.js';
import { extractiveSummary } from './summarize.js';
import { renderFacts, renderMemory } from './render.js';
import { cosine } from './similarity.js';
import { MemoryStore } from './stores.js';

interface Config {
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

interface Core {
  cfg: Config;
  run<T>(id: string, fn: () => Promise<T> | T): Promise<T>;
}

const ROLES = new Set(['user', 'assistant', 'system', 'tool']);

function positive(name: string, value: number, min = 0): number {
  if (!Number.isFinite(value) || value < min) throw new RangeError(`memoryline: ${name} must be a number >= ${min}`);
  return value;
}

function resolve(options: MemoryOptions): Config {
  const countTokens = options.countTokens ?? estimateTokens;
  const maxTokens = positive('budget.maxTokens', options.budget?.maxTokens ?? 4000, 1);
  const targetTokens = positive('budget.targetTokens', options.budget?.targetTokens ?? Math.floor(maxTokens / 2));
  if (targetTokens > maxTokens) throw new RangeError('memoryline: budget.targetTokens must not exceed budget.maxTokens');
  return {
    store: options.store ?? new MemoryStore(),
    maxTokens,
    targetTokens,
    keepRecent: positive('budget.keepRecent', options.budget?.keepRecent ?? 4),
    summaryMaxTokens: positive('budget.summaryMaxTokens', options.budget?.summaryMaxTokens ?? 500, 1),
    perMessageOverhead: positive('budget.perMessageOverhead', options.budget?.perMessageOverhead ?? 4),
    countTokens,
    summarize: options.summarize ?? ((input) => extractiveSummary(input, countTokens)),
    extractFacts: options.extractFacts,
    embed: options.embed,
    retrieve: options.retrieve,
    topK: positive('recall.topK', options.recall?.topK ?? 3),
    minScore: options.recall?.minScore ?? 0.25,
    recallMaxTokens: positive('recall.maxTokens', options.recall?.maxTokens ?? 800),
    archiveMax: positive('archive.max', options.archive?.max ?? 1000),
    alignToUser: options.alignToUser ?? true,
    onArchive: options.onArchive,
    clock: options.clock ?? Date.now,
  };
}

function fresh(id: string, now: number): SessionState {
  return { version: 1, id, recent: [], archive: [], summary: '', facts: [], seq: 1, compactions: 0, createdAt: now, updatedAt: now };
}

export function assertSessionState(state: unknown): asserts state is SessionState {
  if (!state || typeof state !== 'object') throw new TypeError('memoryline: session state must be an object');
  const s = state as { version?: unknown; id?: unknown };
  if (s.version !== 1) throw new TypeError(`memoryline: unsupported session state version ${String(s.version)}`);
  if (typeof s.id !== 'string' || !s.id) throw new TypeError('memoryline: session state needs an id');
}

function assertToolCalls(toolCalls: unknown): asserts toolCalls is ToolCall[] {
  if (!Array.isArray(toolCalls)) throw new TypeError('memoryline: toolCalls must be an array');
  for (const call of toolCalls) {
    const { id, name, arguments: args } = (call ?? {}) as { id?: unknown; name?: unknown; arguments?: unknown };
    if (typeof id !== 'string' || !id) throw new TypeError('memoryline: every tool call needs a string id');
    if (typeof name !== 'string' || !name) throw new TypeError('memoryline: every tool call needs a string name');
    const objectArgs = !!args && typeof args === 'object' && !Array.isArray(args);
    if (typeof args !== 'string' && !objectArgs) throw new TypeError('memoryline: tool call arguments must be a string or an object');
  }
}

function assertMessage(m: unknown): asserts m is Message {
  if (!m || typeof m !== 'object') throw new TypeError('memoryline: message must be an object');
  const { role, content, toolCalls, toolCallId } = m as { role?: unknown; content?: unknown; toolCalls?: unknown; toolCallId?: unknown };
  if (typeof role !== 'string' || !ROLES.has(role)) throw new TypeError(`memoryline: unknown message role ${String(role)}`);
  if (typeof content !== 'string') throw new TypeError('memoryline: message content must be a string');
  if (toolCalls !== undefined) assertToolCalls(toolCalls);
  if (toolCallId !== undefined && typeof toolCallId !== 'string') throw new TypeError('memoryline: toolCallId must be a string');
}

/** Copies the optional message fields that are stored as given. */
function copyOptional(from: Message, to: Message): void {
  if (from.name !== undefined) to.name = from.name;
  if (from.toolCalls !== undefined) to.toolCalls = from.toolCalls;
  if (from.toolCallId !== undefined) to.toolCallId = from.toolCallId;
  if (from.meta !== undefined) to.meta = from.meta;
}

function toMessage(m: StoredMessage): Message {
  const out: Message = { role: m.role, content: m.content, at: m.at };
  copyOptional(m, out);
  return out;
}

/**
 * Moves a fold boundary off the middle of a call/result group: forward past the group when
 * keepRecent allows, otherwise back to the group's start so the whole group stays in the window.
 */
function wholeGroups(recent: StoredMessage[], boundary: number, keepRecent: number): number {
  if (boundary <= 0 || boundary >= recent.length) return boundary;
  const group = toolGroupAt(recent, boundary);
  if (!group || group.start === boundary) return boundary;
  return recent.length - group.end >= keepRecent ? group.end : group.start;
}

export function normalizeFactKey(key: string): string {
  return key
    .trim()
    .toLowerCase()
    .replace(/[\s\-./]+/g, '_')
    .replace(/[^a-z0-9_]/g, '')
    .replace(/^_+|_+$/g, '');
}

function mergeFacts(state: SessionState, inputs: FactInput[], source: 'pinned' | 'extracted', now: number): void {
  for (const input of inputs) {
    const key = normalizeFactKey(String(input?.key ?? ''));
    const value = String(input?.value ?? '').trim();
    if (!key || !value) continue;
    const existing = state.facts.find((f) => f.key === key);
    if (!existing) {
      state.facts.push({ key, value, source, at: now });
    } else if (source === 'pinned' || existing.source === 'extracted') {
      existing.value = value;
      existing.source = source;
      existing.at = now;
    }
  }
}

async function load(core: Core, id: string): Promise<SessionState> {
  const state = await core.cfg.store.load(id);
  if (!state) return fresh(id, core.cfg.clock());
  assertSessionState(state);
  return state;
}

function sum(messages: StoredMessage[]): number {
  let total = 0;
  for (const m of messages) total += m.tokens;
  return total;
}

async function compact(core: Core, state: SessionState, force: boolean): Promise<CompactResult> {
  const { cfg } = core;
  const { recent } = state;
  const none = (): CompactResult => ({ folded: 0, summary: state.summary, facts: [...state.facts] });
  let tokens = sum(recent);
  if (!force && tokens <= cfg.maxTokens) return none();

  let n = 0;
  while (recent.length - n > cfg.keepRecent && (force || tokens > cfg.targetTokens)) {
    tokens -= recent[n]!.tokens;
    n++;
  }
  if (cfg.alignToUser) {
    while (n > 0 && n < recent.length && recent[n]!.role !== 'user' && recent.length - n > cfg.keepRecent) n++;
  }
  n = wholeGroups(recent, n, cfg.keepRecent);
  if (n === 0) return none();

  const now = cfg.clock();
  const folded = recent.splice(0, n);
  const plain = folded.map(toMessage);
  const compaction = ++state.compactions;

  const summary = await cfg.summarize({
    previousSummary: state.summary,
    messages: plain,
    facts: [...state.facts],
    maxTokens: cfg.summaryMaxTokens,
  });
  state.summary = String(summary ?? '').trim();

  if (cfg.extractFacts) {
    const extracted = await cfg.extractFacts({ messages: plain, facts: [...state.facts], summary: state.summary });
    mergeFacts(state, extracted ?? [], 'extracted', now);
  }

  let embeddings: number[][] | undefined;
  if (cfg.embed) embeddings = await cfg.embed(folded.map(messageText));

  const archived: ArchivedMessage[] = folded.map((m, i) => {
    const item: ArchivedMessage = { ...m, compaction };
    const vector = embeddings?.[i];
    if (vector) item.embedding = vector;
    return item;
  });
  state.archive.push(...archived);
  if (state.archive.length > cfg.archiveMax) state.archive.splice(0, state.archive.length - cfg.archiveMax);
  if (cfg.onArchive) await cfg.onArchive(state.id, archived);

  return { folded: n, summary: state.summary, facts: [...state.facts] };
}

function rank(archive: ArchivedMessage[], query: number[], topK: number, minScore: number): RecalledMessage[] {
  const scored: RecalledMessage[] = [];
  for (const message of archive) {
    if (!message.embedding) continue;
    const score = cosine(query, message.embedding);
    if (score >= minScore) scored.push({ message, score });
  }
  scored.sort((a, b) => b.score - a.score || a.message.id - b.message.id);
  return scored.slice(0, topK);
}

function withinTokens(items: RecalledMessage[], max: number): RecalledMessage[] {
  const out: RecalledMessage[] = [];
  let used = 0;
  for (const item of items) {
    if (used + item.message.tokens > max) continue;
    used += item.message.tokens;
    out.push(item);
  }
  return out;
}

function lastUserContent(recent: StoredMessage[]): string | undefined {
  for (let i = recent.length - 1; i >= 0; i--) {
    if (recent[i]!.role === 'user') return recent[i]!.content;
  }
  return undefined;
}

export class Session {
  constructor(
    private readonly core: Core,
    readonly id: string,
  ) {}

  /** Appends one or more messages and compacts when the recent window is over budget. */
  add(input: Message | Message[]): Promise<AddResult> {
    const messages = Array.isArray(input) ? input : [input];
    for (const m of messages) assertMessage(m);
    return this.core.run(this.id, async () => {
      const { cfg } = this.core;
      const state = await load(this.core, this.id);
      if (messages.length === 0) return { added: 0, compacted: false, folded: 0, summary: state.summary, facts: [...state.facts] };
      const now = cfg.clock();
      for (const m of messages) {
        const stored: StoredMessage = {
          role: m.role,
          content: m.content,
          id: state.seq++,
          at: m.at ?? now,
          tokens: cfg.countTokens(messageText(m)) + cfg.perMessageOverhead,
        };
        copyOptional(m, stored);
        state.recent.push(stored);
      }
      const result = await compact(this.core, state, false);
      state.updatedAt = now;
      await cfg.store.save(state);
      return { added: messages.length, compacted: result.folded > 0, ...result };
    });
  }

  /** Folds older messages into the summary. `force` folds everything but the last `keepRecent`. */
  compact(options: { force?: boolean } = {}): Promise<CompactResult> {
    return this.core.run(this.id, async () => {
      const state = await load(this.core, this.id);
      const result = await compact(this.core, state, options.force ?? false);
      if (result.folded > 0) {
        state.updatedAt = this.core.cfg.clock();
        await this.core.cfg.store.save(state);
      }
      return result;
    });
  }

  /** Records facts by hand. Pinned facts are never overwritten by extracted ones. */
  pin(key: string, value: string): Promise<import('./types.js').Fact[]>;
  pin(facts: FactInput | FactInput[]): Promise<import('./types.js').Fact[]>;
  pin(a: string | FactInput | FactInput[], b?: string): Promise<import('./types.js').Fact[]> {
    const inputs: FactInput[] = typeof a === 'string' ? [{ key: a, value: b ?? '' }] : Array.isArray(a) ? a : [a];
    return this.core.run(this.id, async () => {
      const state = await load(this.core, this.id);
      const now = this.core.cfg.clock();
      mergeFacts(state, inputs, 'pinned', now);
      state.updatedAt = now;
      await this.core.cfg.store.save(state);
      return [...state.facts];
    });
  }

  unpin(key: string): Promise<boolean> {
    const k = normalizeFactKey(key);
    return this.core.run(this.id, async () => {
      const state = await load(this.core, this.id);
      const index = state.facts.findIndex((f) => f.key === k);
      if (index < 0) return false;
      state.facts.splice(index, 1);
      state.updatedAt = this.core.cfg.clock();
      await this.core.cfg.store.save(state);
      return true;
    });
  }

  facts(): Promise<import('./types.js').Fact[]> {
    return this.core.run(this.id, async () => [...(await load(this.core, this.id)).facts]);
  }

  /** The recent window. */
  messages(): Promise<StoredMessage[]> {
    return this.core.run(this.id, async () => [...(await load(this.core, this.id)).recent]);
  }

  /** A copy of the whole state, including the archive. */
  state(): Promise<SessionState> {
    return this.core.run(this.id, () => load(this.core, this.id));
  }

  /** Everything the next model call should see. Reads only; never changes the session. */
  context(options: ContextOptions = {}): Promise<MemoryContext> {
    return this.core.run(this.id, async () => {
      const { cfg } = this.core;
      const state = await load(this.core, this.id);
      const recent = state.recent.map(toMessage);

      let recalled: RecalledMessage[] = [];
      const query = options.query ?? lastUserContent(state.recent);
      const canRecall = options.recall !== false && (cfg.retrieve || cfg.embed) && state.archive.length > 0;
      if (canRecall && query) {
        const topK = options.topK ?? cfg.topK;
        const embedding = cfg.embed ? (await cfg.embed([query]))[0] : undefined;
        if (cfg.retrieve) {
          const input = { sessionId: this.id, query, topK, minScore: cfg.minScore, archive: state.archive, ...(embedding ? { embedding } : {}) };
          recalled = [...(await cfg.retrieve(input))];
        } else if (embedding) {
          recalled = rank(state.archive, embedding, topK, cfg.minScore);
        }
        recalled = withinTokens(recalled, cfg.recallMaxTokens).sort((a, b) => a.message.id - b.message.id);
      }

      const facts = [...state.facts];
      const system = renderMemory({ summary: state.summary, facts, recalled });
      const recentTokens = sum(state.recent);
      return {
        sessionId: this.id,
        summary: state.summary,
        facts,
        recalled,
        recent,
        system,
        tokens: {
          summary: cfg.countTokens(state.summary),
          facts: cfg.countTokens(renderFacts(facts)),
          recalled: recalled.reduce((t, r) => t + r.message.tokens, 0),
          recent: recentTokens,
          total: cfg.countTokens(system) + recentTokens,
        },
      };
    });
  }

  /** Deletes the session from the store. */
  clear(): Promise<void> {
    return this.core.run(this.id, () => this.core.cfg.store.delete(this.id));
  }
}

export class Memory {
  private readonly core: Core;
  private readonly locks = new Map<string, Promise<void>>();

  constructor(options: MemoryOptions = {}) {
    const cfg = resolve(options);
    this.core = { cfg, run: (id, fn) => this.run(id, fn) };
  }

  session(id: string): Session {
    if (typeof id !== 'string' || !id) throw new TypeError('memoryline: session id must be a non-empty string');
    return new Session(this.core, id);
  }

  forget(id: string): Promise<void> {
    return this.run(id, () => this.core.cfg.store.delete(id));
  }

  export(id: string): Promise<SessionState | null> {
    return this.run(id, async () => {
      const state = await this.core.cfg.store.load(id);
      return state ? (JSON.parse(JSON.stringify(state)) as SessionState) : null;
    });
  }

  import(state: SessionState): Promise<void> {
    assertSessionState(state);
    return this.run(state.id, () => this.core.cfg.store.save(JSON.parse(JSON.stringify(state)) as SessionState));
  }

  /** Serializes work per session so two adds cannot interleave a compaction. */
  private run<T>(id: string, fn: () => Promise<T> | T): Promise<T> {
    const previous = this.locks.get(id) ?? Promise.resolve();
    const result = previous.then(fn);
    const settled = result.then(
      () => undefined,
      () => undefined,
    );
    this.locks.set(id, settled);
    void settled.then(() => {
      if (this.locks.get(id) === settled) this.locks.delete(id);
    });
    return result;
  }
}

export function createMemory(options: MemoryOptions = {}): Memory {
  return new Memory(options);
}
