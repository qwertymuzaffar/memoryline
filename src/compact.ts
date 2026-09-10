import type { ArchivedMessage, CompactResult, SessionState, StoredMessage } from './types.js';
import { mergeFacts, sum, toMessage, type Config, type Core } from './core.js';
import { messageText, toolGroupAt } from './tools.js';

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

/**
 * How many messages from the front of the window to fold: enough to get under the target
 * (everything but keepRecent when forced), extended to the next user turn when aligning, and
 * never splitting a tool call from its results. Zero means nothing to fold.
 */
function foldBoundary(cfg: Config, recent: StoredMessage[], force: boolean): number {
  let tokens = sum(recent);
  if (!force && tokens <= cfg.maxTokens) return 0;

  let boundary = 0;
  while (recent.length - boundary > cfg.keepRecent && (force || tokens > cfg.targetTokens)) {
    tokens -= recent[boundary]!.tokens;
    boundary++;
  }
  if (cfg.alignToUser) {
    while (
      boundary > 0 &&
      boundary < recent.length &&
      recent[boundary]!.role !== 'user' &&
      recent.length - boundary > cfg.keepRecent
    ) {
      boundary++;
    }
  }
  return wholeGroups(recent, boundary, cfg.keepRecent);
}

/** Folds the first `count` messages into the summary, extracts facts, and archives them. */
async function foldAndArchive(core: Core, state: SessionState, count: number): Promise<CompactResult> {
  const { cfg } = core;
  const now = cfg.clock();
  const folded = state.recent.splice(0, count);
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

  const archived: ArchivedMessage[] = folded.map((message, index) => {
    const item: ArchivedMessage = { ...message, compaction };
    const vector = embeddings?.[index];
    if (vector) item.embedding = vector;
    return item;
  });
  state.archive.push(...archived);
  if (state.archive.length > cfg.archiveMax) state.archive.splice(0, state.archive.length - cfg.archiveMax);
  if (cfg.onArchive) await cfg.onArchive(state.id, archived);

  return { folded: count, summary: state.summary, facts: [...state.facts] };
}

/** Compacts the window when it is over budget (always, when forced). Mutates `state`. */
export async function compact(core: Core, state: SessionState, force: boolean): Promise<CompactResult> {
  const count = foldBoundary(core.cfg, state.recent, force);
  if (count === 0) return { folded: 0, summary: state.summary, facts: [...state.facts] };
  return foldAndArchive(core, state, count);
}
