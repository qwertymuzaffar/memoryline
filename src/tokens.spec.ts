import { estimateTokens } from './tokens.js';

describe('estimateTokens', () => {
  it('returns 0 for empty text', () => {
    expect(estimateTokens('')).toBe(0);
  });

  it('counts about four ASCII characters per token', () => {
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('abcde')).toBe(2);
    expect(estimateTokens('The quick brown fox jumps over the lazy dog.')).toBe(11);
  });

  it('counts each non-ASCII character as a token', () => {
    expect(estimateTokens('日本語')).toBe(3);
    expect(estimateTokens('café')).toBe(2);
  });
});
