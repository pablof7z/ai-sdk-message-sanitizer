# ai-sdk-message-sanitizer

AI SDK middleware that fixes malformed message arrays before they reach the LLM provider.

Some APIs (Anthropic in particular) reject calls with trailing assistant messages, empty content arrays, malformed assistant tool-call inputs, or tool results that appear out of order. These conditions arise naturally from context management, multi-step agents, and streaming — and they produce cryptic 400 errors. This middleware fixes them silently before the call goes out.

## Installation

```bash
npm install ai @ai-sdk/provider ai-sdk-message-sanitizer
```

`@opentelemetry/api` is an optional peer dependency. Install it if you want span events.

## Quick Start

```ts
import { wrapLanguageModel } from "ai";
import { createMessageSanitizerMiddleware } from "ai-sdk-message-sanitizer";

const model = wrapLanguageModel({
  model: baseModel,
  middleware: createMessageSanitizerMiddleware(),
});
```

That's it. Wrap the model and all calls through it are sanitized transparently.

## What It Fixes

### Malformed assistant tool-call inputs

Providers expect assistant `tool-call` parts to carry an object/dictionary `input`. In real agent traces, malformed tool calls sometimes survive into replayed history as strings, arrays, or other non-object payloads, which causes provider-side 400s on the next turn.

```
Before: [assistant: tool-call(input: "<parameter ...>")]
After:  [assistant: tool-call(input: { rawInput: "<parameter ...>", ... })]
```

The sanitizer wraps non-object inputs into a valid object so the prompt remains replayable, while preserving the raw malformed payload for debugging.

### Trailing assistant messages

Anthropic's API rejects prompts where the last message is an assistant message without a tool call. This happens when context management strategies strip the user turn that normally follows, leaving a dangling assistant message.

```
Before: [user] → [assistant: "I found…"] ← Anthropic rejects this
After:  [user]
```

Assistant messages that end with tool calls are left untouched — they are valid and expected.

### Empty content arrays

User and assistant messages with `content: []` are invalid for most providers. These can arise from message construction code or after content is stripped by other middleware.

```
Before: [user: []] → [user: "Real question"]
After:  [user: "Real question"]
```

System messages (string content) and tool messages are never touched.

### Misplaced tool results

In parallel or batched tool-call flows, tool results can end up in the wrong position: later in the prompt than the assistant block that issued the call. Providers require that each assistant block's tool results appear in the immediately following user/tool block.

```
Before:
  [assistant: call-A, call-B, call-C]
  [tool: result-A]                    ← only A resolved
  [assistant: call-D]                 ← B and C still dangling
  [tool: result-B, result-C, result-D]

After:
  [assistant: call-A, call-B, call-C]
  [tool: result-A, result-B, result-C]  ← all three resolved here
  [assistant: call-D]
  [tool: result-D]
```

It also repairs the stricter singleton shape that Anthropic rejects:

```
Before:
  [assistant: call-A]
  [assistant: call-B]
  [assistant: call-C]
  [tool: result-A]
  [tool: result-B]
  [tool: result-C]

After:
  [assistant: call-A]
  [tool: result-A]
  [assistant: call-B]
  [tool: result-B]
  [assistant: call-C]
  [tool: result-C]
```

If a tool result is missing entirely, the sanitizer now strips only the unmatched `tool-call` part instead of forwarding a prompt that the provider will reject. Any surviving assistant text stays in place, and the normal cleanup pass removes assistant messages that become empty or newly invalid after stripping.

## Options

```ts
createMessageSanitizerMiddleware({
  onFix?: (entry: MessageSanitizerFixEntry) => void;
})
```

`onFix` is called once per fix applied. Use it to log to a file, emit metrics, or write to your observability system. If omitted, fixes are applied silently.

```ts
const middleware = createMessageSanitizerMiddleware({
  onFix: (entry) => {
    fs.appendFileSync("warn.log", JSON.stringify(entry) + "\n");
  },
});
```

### Fix entry shape

```ts
interface MessageSanitizerFixEntry {
  ts: string;          // ISO timestamp
  fix: string;         // one of the fix types below
  model: string;       // "provider:modelId"
  callType: string;    // "stream" | "generate" | "object"
  [key: string]: unknown;
}
```

Fix types:

| `fix` value | Triggered when |
|---|---|
| `tool-call-input-wrapped` | One or more assistant tool calls had non-object `input` and were wrapped into a valid dictionary |
| `empty-content-stripped` | One or more user/assistant messages had `content: []` |
| `trailing-assistant-stripped` | One or more trailing assistant messages had no tool calls |
| `invalid-tool-order-detected` | An assistant tool-call message's tool results appeared too late (diagnostic only) |
| `tool-ordering-repaired` | Tool results were successfully relocated to the correct position |
| `unresolved-tool-call-stripped` | One or more assistant tool calls had no matching tool result anywhere in the prompt, so the unmatched tool-call parts were removed |

`invalid-tool-order-detected` fires even when repair isn't possible. When a result is missing entirely, the sanitizer follows that diagnostic with `unresolved-tool-call-stripped` so the prompt can still be sent.

## OpenTelemetry

When `@opentelemetry/api` is installed and an active span exists, the middleware adds span events automatically:

| Event | Attributes |
|---|---|
| `message-sanitizer.fix-applied` | fixes, original/fixed count, removed indices and roles, wrapped tool-call count, model, call type |
| `message-sanitizer.tool-call-input-wrapped` | repairs count, repaired tool call IDs, input types, model, call type |
| `message-sanitizer.invalid-tool-order-detected` | issue count, block starts, missing tool call IDs, model, call type |
| `message-sanitizer.tool-ordering-repaired` | repairs count, repaired tool call IDs, model, call type |
| `message-sanitizer.unresolved-tool-call-stripped` | stripped count, stripped tool call IDs/names, assistant message indices, model, call type |

OTel is optional — if the package is not installed, the middleware runs without it.

## Runnable Examples

| Example | What to look for |
|---|---|
| [01-basic-sanitization.ts](./examples/01-basic-sanitization.ts) | Trailing assistant and empty content removed before the call |
| [02-tool-ordering-repair.ts](./examples/02-tool-ordering-repair.ts) | Misplaced tool results relocated to their correct positions |
| [03-with-logging.ts](./examples/03-with-logging.ts) | `onFix` callback captures every applied fix as structured data |

Run any example from the repo root:

```bash
cd examples && npx tsx 01-basic-sanitization.ts
```

## Running Locally

```bash
bun test
bun run typecheck
bun run build
```

## Why Not Fix This Upstream?

These conditions are bugs in the prompt construction layer, not in the provider. But the right fix (tracing back through multi-step agent code, multi-middleware stacks, or context management) takes time. This middleware is a reliable shim at the model boundary — catches problems before they become provider errors, logs what it changed, and gets out of the way.

Use it alongside `ai-sdk-context-management` or any other prompt-rewriting middleware. Stack order doesn't matter: the sanitizer runs on the final prompt regardless of what other middleware produced.
