import { extractiveSummary, factsPrompt, oneLine, parseFacts, summaryPrompt, transcript } from './summarize.js';
import type { Fact } from './types.js';

const facts: Fact[] = [{ key: 'name', value: 'Dana', source: 'pinned', at: 0 }];

describe('extractiveSummary', () => {
  it('keeps previous lines and adds one clipped line per message, skipping system messages', () => {
    const out = extractiveSummary({
      previousSummary: '- user: earlier point',
      messages: [
        { role: 'system', content: 'ignored' },
        { role: 'user', content: 'Hello   there\n\nfriend' },
        { role: 'assistant', name: 'Mia', content: 'x'.repeat(200) },
      ],
      facts: [],
      maxTokens: 1000,
    });
    const lines = out.split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe('- user: earlier point');
    expect(lines[1]).toBe('- user: Hello there friend');
    expect(lines[2]!.startsWith('- Mia: ')).toBe(true);
    expect(lines[2]!.endsWith('...')).toBe(true);
    expect(lines[2]!.length).toBeLessThanOrEqual('- Mia: '.length + 160);
  });

  it('drops the oldest lines until the summary fits', () => {
    const out = extractiveSummary(
      {
        previousSummary: 'one\ntwo\nthree',
        messages: [{ role: 'user', content: 'four' }],
        facts: [],
        maxTokens: 2,
      },
      (text) => text.split('\n').length,
    );
    expect(out).toBe('three\n- user: four');
  });

  it('never drops the last line even when it is over budget', () => {
    const out = extractiveSummary({ previousSummary: '', messages: [{ role: 'user', content: 'x'.repeat(100) }], facts: [], maxTokens: 1 });
    expect(out.split('\n')).toHaveLength(1);
  });
});

describe('prompts', () => {
  it('summaryPrompt carries the previous summary, facts and the transcript', () => {
    const text = summaryPrompt({ previousSummary: 'They want a table for six.', messages: [{ role: 'user', content: 'Friday works' }], facts, maxTokens: 300 });
    expect(text).toContain('They want a table for six.');
    expect(text).toContain('- name: Dana');
    expect(text).toContain('user: Friday works');
    expect(text).toContain('about 300 tokens');
  });

  it('summaryPrompt says (none) when there is nothing yet', () => {
    const text = summaryPrompt({ previousSummary: '', messages: [], facts: [], maxTokens: 100 });
    expect(text.match(/\(none\)/g)).toHaveLength(3);
  });

  it('factsPrompt lists known facts and messages', () => {
    const text = factsPrompt({ messages: [{ role: 'assistant', name: 'Mia', content: 'Noted.' }], facts, summary: '' });
    expect(text).toContain('- name: Dana');
    expect(text).toContain('Mia: Noted.');
  });

  it('transcript uses the name when present', () => {
    expect(transcript([{ role: 'user', content: ' hi ' }, { role: 'tool', name: 'calendar', content: 'ok' }])).toBe('user: hi\ncalendar: ok');
  });

  it('oneLine flattens whitespace and clips', () => {
    expect(oneLine('a\n\n  b   c')).toBe('a b c');
    expect(oneLine('abcdefghij', 8)).toBe('abcde...');
  });
});

describe('parseFacts', () => {
  it('reads key: value lines with or without bullets', () => {
    expect(parseFacts('- party_size: 6\n* Preferred day: Friday\nname: Dana Lee\njunk line')).toEqual([
      { key: 'party_size', value: '6' },
      { key: 'Preferred day', value: 'Friday' },
      { key: 'name', value: 'Dana Lee' },
    ]);
  });

  it('reads a JSON array or object, with or without a code fence', () => {
    expect(parseFacts('```json\n[{"key":"a","value":1},{"key":"b"},{"nope":true}]\n```')).toEqual([
      { key: 'a', value: '1' },
      { key: 'b', value: '' },
    ]);
    expect(parseFacts('{"city":"Fairfax","guests":4}')).toEqual([
      { key: 'city', value: 'Fairfax' },
      { key: 'guests', value: '4' },
    ]);
  });

  it('falls back to line parsing when JSON is broken', () => {
    expect(parseFacts('{not json\nkey: value')).toEqual([{ key: 'key', value: 'value' }]);
  });

  it('returns nothing for none or empty replies', () => {
    expect(parseFacts('none')).toEqual([]);
    expect(parseFacts('None.')).toEqual([]);
    expect(parseFacts('   ')).toEqual([]);
  });
});
