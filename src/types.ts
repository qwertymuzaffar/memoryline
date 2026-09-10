export type Role = 'user' | 'assistant' | 'system' | 'tool';

export interface Message {
  role: Role;
  content: string;
  /** Speaker or tool name. Passed through untouched. */
  name?: string;
  /** Unix milliseconds. Set when the message is added if absent. */
  at?: number;
  /** Anything you want to keep with the message. Stored, never read. */
  meta?: Record<string, unknown>;
}

/** A message as kept in the recent window. */
export interface StoredMessage extends Message {
  id: number;
  at: number;
  tokens: number;
}

/** A message that compaction folded into the summary and moved to the archive. */
export interface ArchivedMessage extends StoredMessage {
  /** Which compaction folded it, counting from 1. */
  compaction: number;
  embedding?: number[];
}

export interface Fact {
  key: string;
  value: string;
  source: 'pinned' | 'extracted';
  at: number;
}

export interface FactInput {
  key: string;
  value: string;
}

export interface SessionState {
  version: 1;
  id: string;
  recent: StoredMessage[];
  archive: ArchivedMessage[];
  summary: string;
  facts: Fact[];
  /** Next message id. */
  seq: number;
  compactions: number;
  createdAt: number;
  updatedAt: number;
}

export interface SummarizeInput {
  previousSummary: string;
  /** The messages being folded away, oldest first. */
  messages: Message[];
  facts: Fact[];
  /** Soft target for the new summary. */
  maxTokens: number;
}
export type Summarizer = (input: SummarizeInput) => Promise<string> | string;

export interface ExtractFactsInput {
  messages: Message[];
  facts: Fact[];
  summary: string;
}
export type FactExtractor = (input: ExtractFactsInput) => Promise<FactInput[]> | FactInput[];

export type Embedder = (texts: string[]) => Promise<number[][]> | number[][];

export interface RecalledMessage {
  message: ArchivedMessage;
  score: number;
}

export interface RetrieveInput {
  sessionId: string;
  query: string;
  /** The query embedding when an embedder is configured. */
  embedding?: number[];
  topK: number;
  minScore: number;
  archive: ArchivedMessage[];
}
export type Retriever = (input: RetrieveInput) => Promise<RecalledMessage[]> | RecalledMessage[];

export interface Store {
  load(id: string): Promise<SessionState | null> | SessionState | null;
  save(state: SessionState): Promise<void> | void;
  delete(id: string): Promise<void> | void;
}

export interface MemoryOptions {
  /** Where sessions live. Defaults to an in-process MemoryStore. */
  store?: Store;
  budget?: {
    /** Token budget for the verbatim recent window. Default 4000. */
    maxTokens?: number;
    /** Where a fold stops once it starts, so compaction runs in batches rather than on every message. Default half of maxTokens. */
    targetTokens?: number;
    /** Messages at the tail that compaction never folds. Default 4. */
    keepRecent?: number;
    /** Soft target passed to the summarizer. Default 500. */
    summaryMaxTokens?: number;
    /** Tokens charged per message on top of its content. Default 4. */
    perMessageOverhead?: number;
  };
  /** Token counter. Defaults to a character-based estimate. */
  countTokens?: (text: string) => number;
  /** Turns the previous summary plus folded messages into a new summary. Defaults to an extractive fallback. */
  summarize?: Summarizer;
  /** Pulls durable facts out of folded messages. Off by default. */
  extractFacts?: FactExtractor;
  /** Embeds folded messages and queries so context() can recall them. Off by default. */
  embed?: Embedder;
  /** Replaces the built-in cosine search over the archive. */
  retrieve?: Retriever;
  recall?: {
    /** Default 3. */
    topK?: number;
    /** Cosine floor. Default 0.25. */
    minScore?: number;
    /** Token cap for recalled messages. Default 800. */
    maxTokens?: number;
  };
  archive?: {
    /** Archived messages kept per session, oldest dropped first. Default 1000. */
    max?: number;
  };
  /** Extend a fold so the recent window starts on a user turn. Default true. */
  alignToUser?: boolean;
  /** Called after each compaction with the messages that were folded. */
  onArchive?: (sessionId: string, messages: ArchivedMessage[]) => void | Promise<void>;
  clock?: () => number;
}

export interface ContextOptions {
  /** Recall query. Defaults to the latest user message. */
  query?: string;
  /** Set false to skip recall for this call. */
  recall?: boolean;
  topK?: number;
}

export interface MemoryContext {
  sessionId: string;
  summary: string;
  facts: Fact[];
  recalled: RecalledMessage[];
  recent: Message[];
  /** Summary, facts and recalled messages rendered as one Markdown block. */
  system: string;
  tokens: {
    summary: number;
    facts: number;
    recalled: number;
    recent: number;
    /** Rendered memory block plus the recent window. */
    total: number;
  };
}

export interface CompactResult {
  folded: number;
  summary: string;
  facts: Fact[];
}

export interface AddResult extends CompactResult {
  added: number;
  compacted: boolean;
}
