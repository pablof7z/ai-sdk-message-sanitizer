/**
 * Logging with onFix — capture every fix as structured data
 *
 * The onFix callback receives a structured entry for each fix applied.
 * Use it to write to a log file, push to your observability system,
 * increment a metric counter, or alert on unexpected patterns.
 *
 * What to look for:
 * - Each fix produces a separate entry with its own fix type and metadata
 * - The entries contain enough context to diagnose prompt construction bugs
 * - onFix is synchronous; it will not delay the LLM call
 */
import { generateText, wrapLanguageModel } from "ai";
import { createMessageSanitizerMiddleware, type MessageSanitizerFixEntry } from "ai-sdk-message-sanitizer";
import { createMockModel } from "./helpers.js";

async function main() {
    const log: MessageSanitizerFixEntry[] = [];

    const model = wrapLanguageModel({
        model: createMockModel(),
        middleware: createMessageSanitizerMiddleware({
            onFix: (entry) => {
                log.push(entry);
                // In production you might write to a file:
                // fs.appendFileSync("warn.log", JSON.stringify(entry) + "\n");
            },
        }),
    });

    const messages = [
        { role: "user" as const, content: [] },                                                              // empty content
        { role: "user" as const, content: [{ type: "text" as const, text: "What is 2 + 2?" }] },
        { role: "assistant" as const, content: [{ type: "text" as const, text: "4." }] },                   // trailing assistant
        { role: "assistant" as const, content: [{ type: "text" as const, text: "Is there anything else?" }] }, // trailing assistant
    ];

    await generateText({ model, messages });

    console.log(`\n${log.length} fix entries captured:\n`);
    for (const entry of log) {
        console.log(JSON.stringify(entry, null, 2));
        console.log();
    }

    console.log("Entry fields:");
    console.log("  ts          — ISO timestamp");
    console.log("  fix         — what was changed (empty-content-stripped, trailing-assistant-stripped, ...)");
    console.log("  model       — provider:modelId");
    console.log("  callType    — stream | generate | object");
    console.log("  removed     — array of { index, role } for removed messages");
    console.log("  original_count / fixed_count — prompt length before and after");
}

main().catch(console.error);
