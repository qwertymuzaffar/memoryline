import { createMemory, Memory, normalizeFactKey } from './memory.js';
import { MemoryStore } from './stores.js';
import type { Message, SessionState, Store, SummarizeInput } from './types.js';

/** One token per word, no per-message overhead: budgets in the tests read as word counts. */
const words = (text: string) => (text.trim() ? text.trim().split(/\s+/).length : 0);
const budget = (maxTokens: number, keepRecent = 2) => ({ maxTokens, keepRecent, perMessageOverhead: 0 });

const user = (content: string): Message => ({ role: 'user', content });
const bot = (content: string): Message => ({ role: 'assistant', content });

/** Keyword embedder: one dimension per topic word. */
const TOPICS = ['parking', 'allergy', 'birthday', 'patio'];
const embed = (texts: string[]) => texts.map((t) => TOPICS.map((w) => (t.toLowerCase().includes(w) ? 1 : 0)));

describe('add and windowing', () => {
  it('stores messages with ids, timestamps and token counts', async () => {
    let now = 1000;
    const memory = createMemory({ countTokens: words, budget: budget(100), clock: () => now });
    const s = memory.session('a');
    expect(await s.add(user('hello there'))).toMatchObject({ added: 1, compacted: false, folded: 0 });
    now = 2000;
    await s.add([bot('hi'), { role: 'tool', name: 'clock', content: 'noon', at: 5, meta: { x: 1 } }]);
    const recent = await s.messages();
    expect(recent.map((m) => [m.id, m.at, m.tokens])).toEqual([
      [1, 1000, 2],
      [2, 2000, 1],
      [3, 5, 1],
    ]);
    expect(recent[2]).toMatchObject({ name: 'clock', meta: { x: 1 } });
    const state = await s.state();
    expect(state).toMatchObject({ version: 1, id: 'a', seq: 4, compactions: 0, createdAt: 1000, updatedAt: 2000 });
  });

  it('charges the per-message overhead by default', async () => {
    const s = createMemory({ countTokens: words }).session('a');
    await s.add(user('one two'));
    expect((await s.messages())[0]!.tokens).toBe(6);
  });

  it('stores tool calls and results as given and counts call text as tokens', async () => {
    const s = createMemory({ countTokens: words, budget: budget(100) }).session('a');
    const calls = [{ id: 'call_1', name: 'find_slots', arguments: { party: 6 } }];
    await s.add([
      { role: 'assistant', content: 'Let me check', toolCalls: calls },
      { role: 'tool', name: 'find_slots', toolCallId: 'call_1', content: '7pm free' },
    ]);
    const [call, result] = await s.messages();
    expect(call).toMatchObject({ role: 'assistant', toolCalls: calls, tokens: 5 });
    expect(call).not.toHaveProperty('toolCallId');
    expect(result).toMatchObject({ role: 'tool', toolCallId: 'call_1', name: 'find_slots', tokens: 2 });
    expect(result).not.toHaveProperty('toolCalls');
    const exported = (await createMemory().export('missing')) ?? null;
    expect(exported).toBeNull();
  });

  it('rejects malformed tool calls', () => {
    const s = createMemory().session('a');
    const assistant = (extra: Record<string, unknown>) => ({ role: 'assistant', content: '', ...extra }) as unknown as Message;
    expect(() => s.add(assistant({ toolCalls: {} }))).toThrow(/array/);
    expect(() => s.add(assistant({ toolCalls: [{ name: 'f', arguments: '{}' }] }))).toThrow(/string id/);
    expect(() => s.add(assistant({ toolCalls: [{ id: 'c', arguments: '{}' }] }))).toThrow(/string name/);
    expect(() => s.add(assistant({ toolCalls: [{ id: 'c', name: 'f', arguments: 5 }] }))).toThrow(/arguments/);
    expect(() => s.add(assistant({ toolCalls: [{ id: 'c', name: 'f', arguments: ['x'] }] }))).toThrow(/arguments/);
    expect(() => s.add({ role: 'tool', content: 'r', toolCallId: 7 } as unknown as Message)).toThrow(/toolCallId/);
  });

  it('rejects malformed messages and empty ids', () => {
    const memory = createMemory();
    expect(() => memory.session('')).toThrow(/session id/);
    const s = memory.session('a');
    expect(() => s.add({ role: 'nope', content: 'x' } as unknown as Message)).toThrow(/role/);
    expect(() => s.add({ role: 'user', content: 5 } as unknown as Message)).toThrow(/content/);
    expect(() => s.add(null as unknown as Message)).toThrow(/object/);
  });

  it('adding nothing is a no-op', async () => {
    const s = createMemory().session('a');
    expect(await s.add([])).toMatchObject({ added: 0, compacted: false });
  });

  it('rejects bad options', () => {
    expect(() => createMemory({ budget: { maxTokens: 0 } })).toThrow(RangeError);
    expect(() => createMemory({ budget: { keepRecent: -1 } })).toThrow(RangeError);
    expect(() => createMemory({ recall: { topK: Number.NaN } })).toThrow(RangeError);
    expect(() => createMemory({ budget: { maxTokens: 10, targetTokens: 11 } })).toThrow(/targetTokens/);
  });
});

describe('compaction', () => {
  it('folds the oldest messages into the summary once the window is over budget', async () => {
    const calls: SummarizeInput[] = [];
    const memory = createMemory({
      countTokens: words,
      budget: { ...budget(6, 2), targetTokens: 4, summaryMaxTokens: 50 },
      alignToUser: false,
      summarize: (input) => {
        calls.push(input);
        return `${input.previousSummary} | ${input.messages.map((m) => m.content).join(' ')}`.trim();
      },
    });
    const s = memory.session('a');
    await s.add([user('one'), bot('two'), user('three'), bot('four')]);
    expect(calls).toHaveLength(0);
    const result = await s.add([user('five six'), bot('seven')]);
    expect(result).toMatchObject({ added: 2, compacted: true, folded: 3 });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ previousSummary: '', maxTokens: 50 });
    expect(calls[0]!.messages.map((m) => m.content)).toEqual(['one', 'two', 'three']);
    expect((await s.messages()).map((m) => m.content)).toEqual(['four', 'five six', 'seven']);
    const state = await s.state();
    expect(state.summary).toBe('| one two three');
    expect(state.compactions).toBe(1);
    expect(state.archive.map((m) => [m.id, m.compaction])).toEqual([
      [1, 1],
      [2, 1],
      [3, 1],
    ]);

    await s.add([user('eight nine ten'), bot('eleven')]);
    expect(calls[1]!.previousSummary).toBe('| one two three');
    expect(state.summary).toBe('| one two three');
    expect((await s.state()).summary).toBe('| one two three | four five six seven');
  });

  it('extends the fold so the window starts on a user turn', async () => {
    const memory = createMemory({ countTokens: words, budget: budget(4, 1), summarize: () => 'S' });
    const s = memory.session('a');
    await s.add([user('a'), bot('b'), bot('c'), user('d'), bot('e')]);
    expect((await s.messages()).map((m) => m.content)).toEqual(['d', 'e']);
  });

  it('never folds below keepRecent, even when those messages alone exceed the budget', async () => {
    const summarize = vi.fn(() => 'S');
    const memory = createMemory({ countTokens: words, budget: budget(3, 2), summarize });
    const s = memory.session('a');
    const result = await s.add([user('one two three'), bot('four five six')]);
    expect(result.folded).toBe(0);
    expect(summarize).not.toHaveBeenCalled();
    expect(await s.compact()).toMatchObject({ folded: 0 });
  });

  it('force compaction folds everything but keepRecent and saves', async () => {
    const memory = createMemory({ countTokens: words, budget: budget(100, 2), summarize: (i) => i.messages.map((m) => m.content).join(',') });
    const s = memory.session('a');
    await s.add([user('a'), bot('b'), user('c'), bot('d'), user('e'), bot('f')]);
    expect(await s.compact({ force: true })).toMatchObject({ folded: 4, summary: 'a,b,c,d' });
    expect((await s.messages()).map((m) => m.content)).toEqual(['e', 'f']);
    expect((await s.state()).summary).toBe('a,b,c,d');
  });

  it('uses the extractive fallback when no summarizer is given', async () => {
    const memory = createMemory({ countTokens: words, budget: budget(3, 1), alignToUser: false });
    const s = memory.session('a');
    await s.add([user('I need a table'), bot('For how many?'), user('Six')]);
    expect((await s.state()).summary).toBe('- user: I need a table\n- assistant: For how many?');
  });

  it('folds a whole call/result group together when the boundary lands inside it', async () => {
    const calls: SummarizeInput[] = [];
    const memory = createMemory({
      countTokens: words,
      budget: { ...budget(5, 2), targetTokens: 4 },
      alignToUser: false,
      summarize: (input) => {
        calls.push(input);
        return 'S';
      },
    });
    const s = memory.session('a');
    const toolCall = { id: 'call_1', name: 'f', arguments: '{}' };
    // 6 tokens: the fold reaches targetTokens on the tool result, so the whole group folds with it
    const result = await s.add([
      user('a'),
      { role: 'assistant', content: '', toolCalls: [toolCall] },
      { role: 'tool', toolCallId: 'call_1', content: 'r' },
      bot('b'),
      user('c'),
    ]);
    expect(result.folded).toBe(3);
    expect(calls[0]!.messages.map((message) => message.content)).toEqual(['a', '', 'r']);
    expect(calls[0]!.messages[1]).toMatchObject({ toolCalls: [toolCall] });
    expect(calls[0]!.messages[2]).toMatchObject({ toolCallId: 'call_1' });
    expect((await s.messages()).map((message) => message.content)).toEqual(['b', 'c']);
    expect((await s.state()).archive.map((message) => message.toolCallId ?? message.toolCalls?.[0]?.id ?? message.content)).toEqual(['a', 'call_1', 'call_1']);
  });

  it('keeps a whole call/result group in the window when keepRecent forbids folding it', async () => {
    const call = { role: 'assistant' as const, content: '', toolCalls: [{ id: 'call_1', name: 'f', arguments: '{}' }] };
    const toolResult = { role: 'tool' as const, toolCallId: 'call_1', content: 'r' };
    // 5 tokens, fold reaches targetTokens on the result; folding past it would leave fewer than keepRecent
    const s = createMemory({ countTokens: words, budget: { ...budget(4, 2), targetTokens: 3 }, alignToUser: false, summarize: () => 'S' }).session('a');
    expect(await s.add([user('a'), call, toolResult, bot('b')])).toMatchObject({ folded: 1 });
    const window = await s.messages();
    expect(window.map((message) => message.role)).toEqual(['assistant', 'tool', 'assistant']);
    expect(window[0]).toMatchObject({ toolCalls: [{ id: 'call_1' }] });

    // nothing before the group: nothing folds rather than splitting it
    const fresh = createMemory({ countTokens: words, budget: { ...budget(3, 2), targetTokens: 3 }, alignToUser: false, summarize: () => 'S' }).session('b');
    await fresh.add([call, toolResult]);
    expect(await fresh.add(user('c'))).toMatchObject({ folded: 0 });
    expect(await fresh.messages()).toHaveLength(3);
  });

  it('treats a tool result whose call is not in the window as a plain message', async () => {
    const memory = createMemory({ countTokens: words, budget: { ...budget(2, 1), targetTokens: 2 }, alignToUser: false, summarize: () => 'S' });
    const s = memory.session('a');
    await s.add([user('a'), { role: 'tool', toolCallId: 'orphan', content: 'r' }, bot('b')]);
    expect((await s.messages()).map((message) => message.content)).toEqual(['r', 'b']);
  });

  it('keeps the archive capped and reports folded messages through onArchive', async () => {
    const seen: number[][] = [];
    const memory = createMemory({
      countTokens: words,
      budget: { ...budget(2, 1), targetTokens: 2 },
      alignToUser: false,
      archive: { max: 3 },
      summarize: () => 'S',
      onArchive: (id, messages) => void seen.push(messages.map((m) => m.id)),
    });
    const s = memory.session('a');
    for (const w of ['a', 'b', 'c', 'd', 'e', 'f']) await s.add(user(w));
    expect(seen).toEqual([[1], [2], [3], [4]]);
    expect((await s.state()).archive.map((m) => m.id)).toEqual([2, 3, 4]);
  });
});

describe('facts', () => {
  it('pins, normalizes keys, overwrites, and unpins', async () => {
    const s = createMemory({ clock: () => 7 }).session('a');
    expect(await s.pin('Party Size', '6')).toEqual([{ key: 'party_size', value: '6', source: 'pinned', at: 7 }]);
    await s.pin([{ key: 'party-size', value: '8' }, { key: '  ', value: 'ignored' }, { key: 'name', value: '  ' }]);
    expect(await s.facts()).toEqual([{ key: 'party_size', value: '8', source: 'pinned', at: 7 }]);
    expect(await s.unpin('PARTY SIZE')).toBe(true);
    expect(await s.unpin('party_size')).toBe(false);
    expect(await s.facts()).toEqual([]);
    expect(normalizeFactKey('  Full.Name / Preferred  ')).toBe('full_name_preferred');
  });

  it('extracts facts on compaction; extracted facts never overwrite pinned ones', async () => {
    const memory = createMemory({
      countTokens: words,
      budget: budget(2, 1),
      alignToUser: false,
      summarize: () => 'S',
      extractFacts: ({ messages }) => messages.map((m) => ({ key: m.content.split('=')[0]!, value: m.content.split('=')[1]! })),
    });
    const s = memory.session('a');
    await s.pin('city', 'Fairfax');
    await s.add([user('city=Reston'), user('guests=4')]);
    expect((await s.facts()).map((f) => f.key)).toEqual(['city']);
    await s.add(user('guests=5'));
    expect((await s.facts()).map((f) => [f.key, f.value, f.source])).toEqual([
      ['city', 'Fairfax', 'pinned'],
      ['guests', '4', 'extracted'],
    ]);
    await s.add([user('guests=6'), user('noise=1')]);
    expect((await s.facts()).find((f) => f.key === 'guests')).toMatchObject({ value: '6', source: 'extracted' });
    await s.pin('guests', '7');
    expect((await s.facts()).find((f) => f.key === 'guests')).toMatchObject({ value: '7', source: 'pinned' });
  });
});

describe('context and recall', () => {
  async function seeded(extra: Parameters<typeof createMemory>[0] = {}) {
    const memory = createMemory({ countTokens: words, budget: budget(4, 2), alignToUser: false, summarize: () => 'Talked about the venue.', embed, ...extra });
    const s = memory.session('a');
    await s.add([user('Do you have parking'), bot('Yes garage parking'), user('My friend has a nut allergy'), bot('Noted the allergy')]);
    await s.add([user('Is the patio open'), bot('The patio is open')]);
    await s.pin('name', 'Dana');
    return s;
  }

  it('recalls archived messages by cosine similarity to the query, in conversation order', async () => {
    const s = await seeded();
    const ctx = await s.context({ query: 'Where do I find parking and what about the allergy' });
    expect(ctx.recalled.map((r) => r.message.content)).toEqual(['Do you have parking', 'Yes garage parking', 'My friend has a nut allergy']);
    for (const r of ctx.recalled) expect(r.score).toBeCloseTo(Math.SQRT1_2);
    expect(ctx.system).toBe(
      '## Conversation so far\nTalked about the venue.\n\n## Known facts\n- name: Dana\n\n## Relevant earlier messages\n- user: Do you have parking\n- assistant: Yes garage parking\n- user: My friend has a nut allergy',
    );
    expect(ctx.recent.map((m) => m.content)).toEqual(['Is the patio open', 'The patio is open']);
    expect(ctx.tokens).toEqual({ summary: 4, facts: 3, recalled: 13, recent: 8, total: words(ctx.system) + 8 });
  });

  it('defaults the query to the latest user message and honors topK, minScore and recall:false', async () => {
    const s = await seeded({ recall: { topK: 1 } });
    await s.add(user('parking again please'));
    const ctx = await s.context();
    expect(ctx.recalled.map((r) => r.message.content)).toEqual(['Do you have parking']);
    expect((await s.context({ topK: 2 })).recalled).toHaveLength(2);
    expect((await s.context({ recall: false })).recalled).toEqual([]);
    expect((await s.context({ query: 'birthday cake' })).recalled).toEqual([]);
  });

  it('does not recall without an embedder, a query, or an archive', async () => {
    const plain = createMemory({ countTokens: words, budget: budget(4, 2), alignToUser: false, summarize: () => 'S' }).session('p');
    await plain.add([user('parking'), bot('yes'), user('more'), bot('more'), user('x')]);
    expect((await plain.context({ query: 'parking' })).recalled).toEqual([]);
    const s = await seeded();
    expect((await s.context({ query: '' })).recalled).toEqual([]);
    const empty = createMemory({ embed }).session('e');
    await empty.add(bot('hello'));
    expect((await empty.context({ query: 'parking' })).recalled).toEqual([]);
  });

  it('caps recalled messages by tokens', async () => {
    const s = await seeded({ recall: { maxTokens: 4 } });
    const ctx = await s.context({ query: 'parking allergy' });
    expect(ctx.recalled.map((r) => r.message.content)).toEqual(['Do you have parking']);
  });

  it('uses a custom retriever, passing the query embedding when there is one', async () => {
    const retrieve = vi.fn(({ archive }) => [{ message: archive[archive.length - 1]!, score: 1 }]);
    const s = await seeded({ retrieve });
    const ctx = await s.context({ query: 'anything' });
    expect(retrieve).toHaveBeenCalledWith(expect.objectContaining({ sessionId: 'a', query: 'anything', topK: 3, minScore: 0.25, embedding: [0, 0, 0, 0] }));
    expect(ctx.recalled.map((r) => r.message.content)).toEqual(['Noted the allergy']);
    const noEmbed = await seeded({ retrieve, embed: undefined });
    await noEmbed.context({ query: 'q' });
    expect(retrieve.mock.calls[1]![0]).not.toHaveProperty('embedding');
  });

  it('context on an unknown session is empty and creates nothing', async () => {
    const store = new MemoryStore();
    const ctx = await createMemory({ store }).session('ghost').context();
    expect(ctx).toMatchObject({ summary: '', facts: [], recalled: [], recent: [], system: '' });
    expect(store.ids()).toEqual([]);
  });
});

describe('sessions, stores and concurrency', () => {
  it('serializes concurrent adds on one session and keeps sessions apart', async () => {
    const slow: Store = {
      async load(id) {
        await new Promise((r) => setTimeout(r, 2));
        return inner.load(id);
      },
      async save(state) {
        await new Promise((r) => setTimeout(r, 2));
        inner.save(state);
      },
      delete: (id) => inner.delete(id),
    };
    const inner = new MemoryStore();
    const memory = createMemory({ store: slow, countTokens: words, budget: budget(3, 1), alignToUser: false, summarize: () => 'S' });
    const a = memory.session('a');
    const b = memory.session('b');
    await Promise.all([a.add(user('1')), a.add(user('2')), b.add(user('x')), a.add(user('3')), a.add(user('4'))]);
    const state = await a.state();
    expect(state.seq).toBe(5);
    expect([...state.archive, ...state.recent].map((m) => m.content)).toEqual(['1', '2', '3', '4']);
    expect((await b.messages()).map((m) => m.content)).toEqual(['x']);
  });

  it('keeps working after a failed operation', async () => {
    const memory = createMemory({ countTokens: words, budget: budget(1, 0), summarize: () => { throw new Error('llm down'); } });
    const s = memory.session('a');
    await s.add(user('one'));
    await expect(s.add(user('two'))).rejects.toThrow('llm down');
    expect((await s.messages()).map((m) => m.content)).toEqual(['one']);
  });

  it('forgets, clears, exports and imports sessions', async () => {
    const memory = createMemory();
    const s = memory.session('a');
    await s.add(user('hi'));
    const exported = (await memory.export('a'))!;
    expect(exported.recent[0]!.content).toBe('hi');
    await memory.forget('a');
    expect(await memory.export('a')).toBeNull();
    await memory.import(exported);
    expect((await s.messages()).map((m) => m.content)).toEqual(['hi']);
    await s.clear();
    expect(await s.messages()).toEqual([]);
    expect(() => memory.import({ ...exported, version: 2 } as unknown as SessionState)).toThrow(/version/);
    expect(() => memory.import({ ...exported, id: '' })).toThrow(/id/);
    expect(() => memory.import(null as unknown as SessionState)).toThrow(/object/);
  });

  it('rejects stored state it does not understand', async () => {
    const store: Store = { load: () => ({ version: 9 }) as unknown as SessionState, save: () => undefined, delete: () => undefined };
    await expect(new Memory({ store }).session('a').messages()).rejects.toThrow(/version/);
  });
});
