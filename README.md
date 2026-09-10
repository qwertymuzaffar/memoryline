# memoryline

Conversation memory for LLM apps in TypeScript. A token-budgeted window of recent turns, a running summary that compacts older ones, pinned and extracted facts, and semantic recall of what was folded away. Bring your own model, embeddings and store. Zero dependencies, ESM and CommonJS, Node 18+.

Every chat product hits the same wall: the conversation outgrows the context window, and "just send the last twenty messages" forgets the party size the user gave in message three. memoryline keeps the recent turns verbatim, folds the rest into a summary with a model call you control, pulls durable facts out on the way, and can bring back an earlier message when a new one is about the same thing. What you get back is one Markdown block for the system prompt plus the recent messages, ready for OpenAI or Anthropic.

```
recent window (verbatim, under budget) ── over budget? ──> fold oldest turns
                                                             │
                                          summarize(previous summary + folded) ──> summary
                                          extractFacts(folded)                 ──> facts
                                          embed(folded)                        ──> archive (for recall)
```

## Install

```sh
npm install memoryline
```

## Quick start

```ts
import OpenAI from 'openai';
import { createMemory, summaryPrompt, toOpenAI } from 'memoryline';

const openai = new OpenAI();

const memory = createMemory({
  budget: { maxTokens: 3000, keepRecent: 6 },
  summarize: async (input) => {
    const res = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [{ role: 'user', content: summaryPrompt(input) }],
    });
    return res.choices[0]?.message.content ?? input.previousSummary;
  },
});

const session = memory.session(`user:${userId}:thread:${threadId}`);

await session.add({ role: 'user', content: text });

const ctx = await session.context();
const res = await openai.chat.completions.create({
  model: 'gpt-4.1-mini',
  messages: toOpenAI(ctx, { systemPrompt: 'You are Mia, the booking assistant for Rosetta.' }),
});

await session.add({ role: 'assistant', content: res.choices[0]!.message.content! });
```

`add` appends and, when the recent window is over budget, compacts. `context` returns everything the next call should see. Nothing else is required; a session with no summarizer configured still works, using a built-in extractive fallback.

## How it works

A session holds four things:

| Part | What it is | Where it goes in the prompt |
|---|---|---|
| Recent window | The latest messages, verbatim | As messages |
| Summary | A running abstract of everything folded out of the window | System block, "Conversation so far" |
| Facts | Key/value pairs pinned by you or extracted by a model | System block, "Known facts" |
| Archive | Folded messages, with embeddings when you provide an embedder | System block, "Relevant earlier messages", when recalled |

Compaction runs inside `add` when the window's token count exceeds `budget.maxTokens`. It folds the oldest messages until the window is under `budget.targetTokens` (half of `maxTokens` by default), so the summarizer runs in batches rather than on every message. It never folds the last `budget.keepRecent` messages, and by default it extends the fold so the window starts on a user turn, which keeps user/assistant alternation intact.

Each fold does, in order: `summarize`, `extractFacts` (if configured), `embed` (if configured), then archives the folded messages and calls `onArchive`.

```ts
const result = await session.add(messages);
// { added: 2, compacted: true, folded: 9, summary: '...', facts: [...] }

await session.compact({ force: true }); // fold everything but keepRecent, for example at the end of a call
```

## Plugging in a model

`summarize` receives the previous summary, the folded messages, the current facts and a soft token target, and returns the new summary. `summaryPrompt` turns that input into a prompt for any chat model. `extractFacts` receives the folded messages and returns `{ key, value }` pairs; `factsPrompt` and `parseFacts` do the prompt and the parsing.

```ts
import Anthropic from '@anthropic-ai/sdk';
import { createMemory, summaryPrompt, factsPrompt, parseFacts } from 'memoryline';

const anthropic = new Anthropic();

async function ask(prompt: string): Promise<string> {
  const res = await anthropic.messages.create({
    model: 'claude-sonnet-5',
    max_tokens: 600,
    messages: [{ role: 'user', content: prompt }],
  });
  return res.content[0]?.type === 'text' ? res.content[0].text : '';
}

const memory = createMemory({
  summarize: (input) => ask(summaryPrompt(input)),
  extractFacts: async (input) => parseFacts(await ask(factsPrompt(input))),
});
```

`parseFacts` accepts `key: value` lines (with or without bullets), a JSON array of `{ key, value }`, or a JSON object, and ignores code fences and a bare `none`. Keys are normalized to snake_case when stored, so `Party Size` and `party-size` are the same fact.

Without a `summarize` function, `extractiveSummary` is used: one clipped line per folded message, oldest lines dropped to fit `budget.summaryMaxTokens`. It is a record of what was said, not an abstract of it.

## Facts

Pin what you already know. Pinned facts are never overwritten by extracted ones; extracted facts overwrite earlier extracted values for the same key.

```ts
await session.pin('name', 'Dana');
await session.pin([{ key: 'party_size', value: '6' }, { key: 'preferred_day', value: 'Friday' }]);
await session.unpin('preferred_day');
const facts = await session.facts();
// [{ key: 'name', value: 'Dana', source: 'pinned', at: 1757... }, ...]
```

## Recall

Give the memory an embedder and folded messages become searchable. `context` embeds the query (the latest user message unless you pass one), scores the archive by cosine similarity, and includes the best matches, oldest first, under `recall.maxTokens`.

```ts
const memory = createMemory({
  embed: async (texts) => {
    const res = await openai.embeddings.create({ model: 'text-embedding-3-small', input: texts });
    return res.data.map((d) => d.embedding);
  },
  recall: { topK: 3, minScore: 0.3, maxTokens: 600 },
});

const ctx = await session.context({ query: 'did they mention parking?' });
ctx.recalled; // [{ message: { role: 'user', content: 'Is there parking nearby?', ... }, score: 0.71 }]
```

The built-in search is exact cosine over the session's archive, which is the right tool up to a few thousand messages per session. For more, or for cross-session search, pass `retrieve` and keep the index wherever you like. With [minivec](https://www.npmjs.com/package/minivec):

```ts
import { MiniVec } from 'minivec';

const index = new MiniVec({ dimensions: 1536 });

const memory = createMemory({
  embed,
  onArchive: async (sessionId, messages) => {
    for (const m of messages) {
      if (m.embedding) index.add(`${sessionId}:${m.id}`, m.embedding, { sessionId, id: m.id });
    }
  },
  retrieve: ({ sessionId, embedding, topK, minScore, archive }) => {
    if (!embedding) return [];
    return index
      .search(embedding, { k: topK, filter: (meta) => meta.sessionId === sessionId })
      .filter((hit) => hit.score >= minScore)
      .flatMap((hit) => {
        const message = archive.find((m) => m.id === hit.metadata.id);
        return message ? [{ message, score: hit.score }] : [];
      });
  },
});
```

Pass `{ recall: false }` to `context` to skip recall for one call.

## Rendering

`context` returns the parts and a rendered `system` block:

```md
## Conversation so far
Dana wants a table for six on Friday evening and asked about the patio.

## Known facts
- name: Dana
- party_size: 6

## Relevant earlier messages
- user: Is there parking nearby?
- assistant: There is a garage next door, two dollars an hour.
```

`toOpenAI(ctx, { systemPrompt })` returns a messages array with one system message (your prompt plus the block) followed by the recent window. `toAnthropic(ctx, { systemPrompt })` returns `{ system, messages }` with strictly alternating user/assistant turns: system-role messages in the window join the system string, tool messages become user messages tagged `[toolname]`, consecutive same-role messages merge, and a window that begins on an assistant turn gets a one-line user message in front. `renderMemory(parts, { headings })` renders the block on its own with headings of your choosing.

## Stores

The default `MemoryStore` keeps sessions in process. For anything that restarts, use SQL or Redis. Both are bring-your-own-client: nothing is imported.

```ts
import { Pool } from 'pg';
import { createMemory, SqlStore } from 'memoryline';

const pool = new Pool();
await pool.query(SqlStore.ddl()); // CREATE TABLE IF NOT EXISTS memoryline_sessions (...)

const memory = createMemory({
  store: new SqlStore({ query: (sql, params) => pool.query(sql, params) }),
});
```

`SqlStore` speaks `postgres` (default, `$1` placeholders and `ON CONFLICT`), `sqlite` (`?` and `ON CONFLICT`) and `mysql` (`?` and `ON DUPLICATE KEY UPDATE`). The query function returns `{ rows }` or a plain array. One row per session, state as JSON text.

```ts
import { createClient } from 'redis';
import { createMemory, RedisStore } from 'memoryline';

const client = createClient();
await client.connect();

const memory = createMemory({
  store: new RedisStore(client, { prefix: 'chat:', ttlSeconds: 60 * 60 * 24 * 30 }),
});
```

`RedisStore` needs `get`, `set`, `del` and, when a TTL is set, `expire`. node-redis and ioredis both fit. Any object with `load`, `save` and `delete` works as a store.

Sessions are loaded from the store on every operation and saved after every change. Operations on the same session are serialized in process, so a compaction cannot interleave with an add. Across processes the last writer wins; route a session to one worker if that matters.

## Budgets and tokens

| Option | Default | Meaning |
|---|---|---|
| `budget.maxTokens` | 4000 | Compaction starts when the recent window exceeds this |
| `budget.targetTokens` | half of maxTokens | Compaction stops once the window is under this |
| `budget.keepRecent` | 4 | Messages at the tail that are never folded |
| `budget.summaryMaxTokens` | 500 | Soft target handed to the summarizer |
| `budget.perMessageOverhead` | 4 | Tokens charged per message on top of its content |
| `alignToUser` | true | Extend a fold so the window starts on a user turn |
| `recall.topK` / `minScore` / `maxTokens` | 3 / 0.25 / 800 | Recall limits |
| `archive.max` | 1000 | Archived messages kept per session, oldest dropped first |

Tokens are estimated at about four ASCII characters per token and one token per non-ASCII character. That is close enough to keep a window inside a model's limit with headroom. For exact budgets pass a tokenizer:

```ts
import { encoding_for_model } from 'tiktoken';
const enc = encoding_for_model('gpt-4o');
const memory = createMemory({ countTokens: (text) => enc.encode(text).length });
```

`ctx.tokens` reports the estimated size of the summary, facts, recalled messages, recent window, and the total the model will see.

## Sessions, tenants and forgetting

A session id is any string; namespace it however your app is organized (`user:42`, `user:42:thread:7`, `org:acme:support`). Everything about a session lives under that id, so one call removes it:

```ts
await memory.forget('user:42:thread:7');      // or session.clear()
const state = await memory.export('user:42'); // full state, JSON-serializable
await memory.import(state);                    // into another memory or store
```

`session.state()` returns the same snapshot for debugging, and `session.messages()` the recent window alone.

## Design notes and limitations

- The summary is only as good as the model behind `summarize`. The extractive fallback keeps text, not meaning; production apps should pass a model call.
- Summaries drift. Every compaction rewrites the whole summary from the previous one plus the folded turns, so a mistake can persist. Pinned facts exist so the details that must not drift live outside the summary.
- Recall is per session. Cross-session memory ("what did this user say last month") is a `retrieve` function over your own index; the `onArchive` hook is where to feed it.
- Tool-call payloads are stored as plain content. OpenAI's `tool_call_id` and Anthropic's tool-use blocks are not modeled; put a short textual result in the message and keep the raw payload in `meta`.
- `perMessageOverhead` approximates chat-format framing tokens. It is a constant, not a per-model table.
- The archive is capped per session (`archive.max`) and stored inline with the session, embeddings included. For long-lived sessions with large embeddings, lower the cap or move recall to an external index.
- No encryption, no PII handling. Pair with [deidentify](https://www.npmjs.com/package/deidentify) if messages carry personal data that should not reach the model or the store.

## Alternatives

LangChain's memory classes and the Vercel AI SDK's message helpers solve the same window problem inside their frameworks. Hosted memory services (mem0 and others) add cross-session user memory as an API. memoryline is the plain-library option: no framework, no service, a store you already run, and every model call made by code you wrote.

## License

MIT
