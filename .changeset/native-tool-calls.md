---
'memoryline': minor
---

Model tool calls natively. An assistant message can carry `toolCalls: [{ id, name, arguments }]` and a `tool` message carries `toolCallId`; `toOpenAI` renders them as `tool_calls` / `tool_call_id` and `toAnthropic` as `tool_use` / `tool_result` blocks. Compaction never splits a call from its result: a fold boundary that lands inside a call/result group moves past the group when `keepRecent` allows, otherwise back to its start. Tool calls show as `[call name(arguments)]` in summaries, transcripts and the memory block, and count toward the window's tokens. New exports: `messageText`, `argumentsText`, `parseArguments`, `toolGroupAt`, and the `ToolCall`, `OpenAIToolCall`, `AnthropicContentBlock` types. Messages without the new fields behave exactly as before.
