import type {
  AddResult,
  ArchivedMessage,
  CompactResult,
  ContextOptions,
  Fact,
  FactInput,
  MemoryContext,
  MemoryOptions,
  Message,
  RecalledMessage,
  SessionState,
  StoredMessage,
  ToolCall,
} from './types.js';
import { compact } from './compact.js';
import { copyOptional, fresh, mergeFacts, normalizeFactKey, sum, toMessage, type Config, type Core } from './core.js';
import { estimateTokens } from './tokens.js';
import { messageText } from './tools.js';
import { extractiveSummary } from './summarize.js';
import { renderFacts, renderMemory } from './render.js';
import { cosine } from './similarity.js';
import { MemoryStore } from './stores.js';

export { normalizeFactKey } from './core.js';

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

export function assertSessionState(state: unknown): asserts state is SessionState {
  if (!state || typeof state !== 'object') throw new TypeError('memoryline: session state must be an object');
  const candidate = state as { version?: unknown; id?: unknown };
  if (candidate.version !== 1) throw new TypeError(`memoryline: unsupported session state version ${String(candidate.version)}`);
  if (typeof candidate.id !== 'string' || !candidate.id) throw new TypeError('memoryline: session state needs an id');
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

function assertMessage(candidate: unknown): asserts candidate is Message {
  if (!candidate || typeof candidate !== 'object') throw new TypeError('memoryline: message must be an object');
  const { role, content, toolCalls, toolCallId } = candidate as { role?: unknown; content?: unknown; toolCalls?: unknown; toolCallId?: unknown };
  if (typeof role !== 'string' || !ROLES.has(role)) throw new TypeError(`memoryline: unknown message role ${String(role)}`);
  if (typeof content !== 'string') throw new TypeError('memoryline: message content must be a string');
  if (toolCalls !== undefined) assertToolCalls(toolCalls);
  if (toolCallId !== undefined && typeof toolCallId !== 'string') throw new TypeError('memoryline: toolCallId must be a string');
}

async function load(core: Core, id: string): Promise<SessionState> {
  const state = await core.cfg.store.load(id);
  if (!state) return fresh(id, core.cfg.clock());
  assertSessionState(state);
  return state;
}

function rank(archive: ArchivedMessage[], query: number[], topK: number, minScore: number): RecalledMessage[] {
  const scored: RecalledMessage[] = [];
  for (const message of archive) {
    if (!message.embedding) continue;
    const score = cosine(query, message.embedding);
    if (score >= minScore) scored.push({ message, score });
  }
  scored.sort((left, right) => right.score - left.score || left.message.id - right.message.id);
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
  for (let index = recent.length - 1; index >= 0; index--) {
    if (recent[index]!.role === 'user') return recent[index]!.content;
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
    for (const message of messages) assertMessage(message);
    return this.mutate(
      async (state, now) => {
        const { cfg } = this.core;
        if (messages.length === 0) return { added: 0, compacted: false, folded: 0, summary: state.summary, facts: [...state.facts] };
        for (const message of messages) {
          const stored: StoredMessage = {
            role: message.role,
            content: message.content,
            id: state.seq++,
            at: message.at ?? now,
            tokens: cfg.countTokens(messageText(message)) + cfg.perMessageOverhead,
          };
          copyOptional(message, stored);
          state.recent.push(stored);
        }
        const result = await compact(this.core, state, false);
        return { added: messages.length, compacted: result.folded > 0, ...result };
      },
      (result) => result.added > 0,
    );
  }

  /** Folds older messages into the summary. `force` folds everything but the last `keepRecent`. */
  compact(options: { force?: boolean } = {}): Promise<CompactResult> {
    return this.mutate((state) => compact(this.core, state, options.force ?? false), (result) => result.folded > 0);
  }

  /** Records facts by hand. Pinned facts are never overwritten by extracted ones. */
  pin(key: string, value: string): Promise<Fact[]>;
  pin(facts: FactInput | FactInput[]): Promise<Fact[]>;
  pin(keyOrFacts: string | FactInput | FactInput[], value?: string): Promise<Fact[]> {
    const inputs: FactInput[] =
      typeof keyOrFacts === 'string' ? [{ key: keyOrFacts, value: value ?? '' }] : Array.isArray(keyOrFacts) ? keyOrFacts : [keyOrFacts];
    return this.mutate((state, now) => {
      mergeFacts(state, inputs, 'pinned', now);
      return [...state.facts];
    });
  }

  unpin(key: string): Promise<boolean> {
    const normalized = normalizeFactKey(key);
    const removeFact = (state: SessionState): boolean => {
      const index = state.facts.findIndex((fact) => fact.key === normalized);
      if (index < 0) return false;
      state.facts.splice(index, 1);
      return true;
    };
    return this.mutate(removeFact, (removed) => removed);
  }

  facts(): Promise<Fact[]> {
    return this.read((state) => [...state.facts]);
  }

  /** The recent window. */
  messages(): Promise<StoredMessage[]> {
    return this.read((state) => [...state.recent]);
  }

  /** A copy of the whole state, including the archive. */
  state(): Promise<SessionState> {
    return this.read((state) => state);
  }

  /** Everything the next model call should see. Reads only; never changes the session. */
  context(options: ContextOptions = {}): Promise<MemoryContext> {
    return this.read(async (state) => {
      const { cfg } = this.core;
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
        recalled = withinTokens(recalled, cfg.recallMaxTokens).sort((left, right) => left.message.id - right.message.id);
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
          recalled: recalled.reduce((total, item) => total + item.message.tokens, 0),
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

  /** Under the session lock: loads the state and hands it to `body`. Never saves. */
  private read<T>(body: (state: SessionState) => Promise<T> | T): Promise<T> {
    return this.core.run(this.id, async () => body(await load(this.core, this.id)));
  }

  /**
   * Under the session lock: loads the state, runs `body` with it and the current time, and
   * saves it with `updatedAt` stamped when `persist` accepts the result (always, by default).
   */
  private mutate<T>(body: (state: SessionState, now: number) => Promise<T> | T, persist: (result: T) => boolean = () => true): Promise<T> {
    return this.core.run(this.id, async () => {
      const state = await load(this.core, this.id);
      const now = this.core.cfg.clock();
      const result = await body(state, now);
      if (persist(result)) {
        state.updatedAt = now;
        await this.core.cfg.store.save(state);
      }
      return result;
    });
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
