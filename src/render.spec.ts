import { renderMemory, toAnthropic, toOpenAI } from './render.js';
import type { MemoryContext, RecalledMessage } from './types.js';

const recalled: RecalledMessage[] = [
  { message: { id: 1, role: 'user', content: 'Do you\nhave parking?', at: 0, tokens: 5, compaction: 1 }, score: 0.9 },
  { message: { id: 2, role: 'assistant', name: 'Mia', content: 'Yes, a garage next door.', at: 0, tokens: 6, compaction: 1 }, score: 0.8 },
];

function ctx(partial: Partial<MemoryContext> = {}): MemoryContext {
  return {
    sessionId: 's',
    summary: '',
    facts: [],
    recalled: [],
    recent: [],
    system: '',
    tokens: { summary: 0, facts: 0, recalled: 0, recent: 0, total: 0 },
    ...partial,
  };
}

describe('renderMemory', () => {
  it('renders only the sections that have content', () => {
    expect(renderMemory({ summary: '', facts: [], recalled: [] })).toBe('');
    expect(renderMemory({ summary: ' Booked for Friday. ', facts: [], recalled: [] })).toBe('## Conversation so far\nBooked for Friday.');
    const full = renderMemory({
      summary: 'S',
      facts: [{ key: 'party_size', value: '6', source: 'pinned', at: 0 }],
      recalled,
    });
    expect(full).toBe(
      '## Conversation so far\nS\n\n## Known facts\n- party_size: 6\n\n## Relevant earlier messages\n- user: Do you have parking?\n- Mia: Yes, a garage next door.',
    );
  });

  it('accepts custom headings', () => {
    expect(renderMemory({ summary: 'S', facts: [], recalled: [] }, { headings: { summary: 'Recap' } })).toBe('## Recap\nS');
  });
});

describe('toOpenAI', () => {
  it('puts the memory block and system prompt into one system message before the window', () => {
    const out = toOpenAI(
      ctx({ system: '## Known facts\n- a: b', recent: [{ role: 'user', content: 'hi' }, { role: 'assistant', name: 'Mia', content: 'hello' }] }),
      { systemPrompt: 'You are Mia.' },
    );
    expect(out).toEqual([
      { role: 'system', content: 'You are Mia.\n\n## Known facts\n- a: b' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello', name: 'Mia' },
    ]);
  });

  it('omits the system message when there is nothing to say', () => {
    expect(toOpenAI(ctx({ recent: [{ role: 'user', content: 'hi' }] }))).toEqual([{ role: 'user', content: 'hi' }]);
  });
});

describe('toAnthropic', () => {
  it('merges same-role runs, folds system messages into system, tags tool messages, and starts on a user turn', () => {
    const out = toAnthropic(
      ctx({
        system: 'MEMORY',
        recent: [
          { role: 'assistant', content: 'Welcome back.' },
          { role: 'system', content: 'Store closes at 9.' },
          { role: 'user', content: 'Any tables?' },
          { role: 'tool', name: 'calendar', content: '7pm free' },
          { role: 'assistant', content: '7pm works.' },
          { role: 'assistant', content: 'Shall I book it?' },
        ],
      }),
      { systemPrompt: 'You are Mia.' },
    );
    expect(out.system).toBe('You are Mia.\n\nMEMORY\n\nStore closes at 9.');
    expect(out.messages).toEqual([
      { role: 'user', content: '(continuing the conversation)' },
      { role: 'assistant', content: 'Welcome back.' },
      { role: 'user', content: 'Any tables?\n\n[calendar] 7pm free' },
      { role: 'assistant', content: '7pm works.\n\nShall I book it?' },
    ]);
  });

  it('handles an empty window', () => {
    expect(toAnthropic(ctx())).toEqual({ system: '', messages: [] });
  });
});
