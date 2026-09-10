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

  it('renders tool calls as tool_calls and results with tool_call_id', () => {
    const out = toOpenAI(
      ctx({
        recent: [
          { role: 'user', content: 'Any tables?' },
          {
            role: 'assistant',
            content: '',
            toolCalls: [
              { id: 'call_1', name: 'find_slots', arguments: { party: 6 } },
              { id: 'call_2', name: 'weather', arguments: '{"day":"friday"}' },
            ],
          },
          { role: 'tool', name: 'find_slots', toolCallId: 'call_1', content: '7pm free' },
          { role: 'tool', name: 'weather', toolCallId: 'call_2', content: 'clear' },
          { role: 'assistant', content: '7pm works.' },
        ],
      }),
    );
    expect(out).toEqual([
      { role: 'user', content: 'Any tables?' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 'call_1', type: 'function', function: { name: 'find_slots', arguments: '{"party":6}' } },
          { id: 'call_2', type: 'function', function: { name: 'weather', arguments: '{"day":"friday"}' } },
        ],
      },
      { role: 'tool', content: '7pm free', name: 'find_slots', tool_call_id: 'call_1' },
      { role: 'tool', content: 'clear', name: 'weather', tool_call_id: 'call_2' },
      { role: 'assistant', content: '7pm works.' },
    ]);
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

  it('renders tool calls as tool_use blocks and results as tool_result blocks in the next user turn', () => {
    const out = toAnthropic(
      ctx({
        recent: [
          { role: 'user', content: 'Any tables?' },
          {
            role: 'assistant',
            content: 'Let me check.',
            toolCalls: [
              { id: 'call_1', name: 'find_slots', arguments: '{"party":6}' },
              { id: 'call_2', name: 'weather', arguments: { day: 'friday' } },
            ],
          },
          { role: 'tool', name: 'find_slots', toolCallId: 'call_1', content: '7pm free' },
          { role: 'tool', name: 'weather', toolCallId: 'call_2', content: 'clear' },
          { role: 'assistant', content: '7pm works.' },
          { role: 'user', content: 'Book it.' },
        ],
      }),
    );
    expect(out.messages).toEqual([
      { role: 'user', content: 'Any tables?' },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'Let me check.' },
          { type: 'tool_use', id: 'call_1', name: 'find_slots', input: { party: 6 } },
          { type: 'tool_use', id: 'call_2', name: 'weather', input: { day: 'friday' } },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'call_1', content: '7pm free' },
          { type: 'tool_result', tool_use_id: 'call_2', content: 'clear' },
        ],
      },
      { role: 'assistant', content: '7pm works.' },
      { role: 'user', content: 'Book it.' },
    ]);
  });

  it('turns a merged turn into blocks, keeps unparsable arguments, and still tags results without an id', () => {
    const out = toAnthropic(
      ctx({
        recent: [
          { role: 'assistant', content: '', toolCalls: [{ id: 'call_1', name: 'lookup', arguments: 'not json' }] },
          { role: 'tool', toolCallId: 'call_1', content: 'found' },
          { role: 'user', content: 'Thanks.' },
          { role: 'tool', name: 'clock', content: 'noon' },
        ],
      }),
    );
    expect(out.messages).toEqual([
      { role: 'user', content: '(continuing the conversation)' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'call_1', name: 'lookup', input: { arguments: 'not json' } }] },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'call_1', content: 'found' },
          { type: 'text', text: 'Thanks.' },
          { type: 'text', text: '[clock] noon' },
        ],
      },
    ]);
  });
});

describe('renderRecalled', () => {
  it('shows tool calls as text after the message content', () => {
    const memoryContext = ctx({
      recalled: [
        {
          message: { id: 3, role: 'assistant', content: 'Checking.', toolCalls: [{ id: 'c', name: 'find', arguments: { q: 1 } }], at: 0, tokens: 4, compaction: 1 },
          score: 0.5,
        },
      ],
    });
    expect(renderMemory({ summary: '', facts: [], recalled: memoryContext.recalled })).toBe(
      '## Relevant earlier messages\n- assistant: Checking. [call find({"q":1})]',
    );
  });
});
