/**
 * Character-based token estimate: about four ASCII characters per token, which is the usual rule of
 * thumb for English, and one token per non-ASCII character, which is closer to the truth for CJK text
 * and only slightly pessimistic for accented Latin. Pass a real tokenizer through `countTokens` when the
 * budget has to be exact.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  let ascii = 0;
  let other = 0;
  for (const ch of text) {
    if (ch.charCodeAt(0) < 128) ascii++;
    else other++;
  }
  return Math.ceil(ascii / 4) + other;
}
