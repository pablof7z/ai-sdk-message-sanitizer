/**
 * Tool ordering repair — misplaced tool results relocated to correct positions
 *
 * This reproduces a real production pattern: an agent issues multiple parallel tool calls,
 * but results arrive and get appended in a different order than expected. After a context
 * management pass the results end up in the wrong block — after the next assistant turn
 * instead of immediately after the assistant turn that issued the calls.
 *
 * Anthropic's API requires that each assistant block's tool results appear in the
 * immediately following user/tool block. Any gap causes a 400.
 *
 * What to look for:
 * - Before: assistant(A,B,C) → tool(A only) → assistant(D) → tool(B,C,D mixed)
 * - After:  assistant(A,B,C) → tool(A,B,C)  → assistant(D) → tool(D)
 */
import { generateText, wrapLanguageModel } from "ai";
import type { LanguageModelV3Prompt } from "@ai-sdk/provider";
import { createMessageSanitizerMiddleware } from "ai-sdk-message-sanitizer";
import { createMockModel, createPromptCaptureMiddleware, printPrompt } from "./helpers.js";

async function main() {
    const capturedPrompts: LanguageModelV3Prompt[] = [];
    const fixes: string[] = [];

    const model = wrapLanguageModel({
        model: wrapLanguageModel({
            model: createMockModel(),
            middleware: createPromptCaptureMiddleware(capturedPrompts),
        }),
        middleware: createMessageSanitizerMiddleware({
            onFix: (entry) => fixes.push(entry.fix),
        }),
    });

    // Malformed: call-B and call-C results are in the wrong block
    const messages = [
        { role: "user" as const, content: [{ type: "text" as const, text: "Search for multiple things." }] },
        {
            role: "assistant" as const,
            content: [
                { type: "tool-call" as const, toolCallId: "call-A", toolName: "search", input: { query: "france" } },
                { type: "tool-call" as const, toolCallId: "call-B", toolName: "search", input: { query: "germany" } },
                { type: "tool-call" as const, toolCallId: "call-C", toolName: "search", input: { query: "italy" } },
            ],
        },
        {
            role: "tool" as const,
            content: [
                // Only call-A resolved here — B and C are missing
                { type: "tool-result" as const, toolCallId: "call-A", toolName: "search", output: { type: "text" as const, value: "France: Paris" } },
            ],
        },
        {
            role: "assistant" as const,
            content: [
                { type: "tool-call" as const, toolCallId: "call-D", toolName: "summarize", input: { countries: ["france", "germany", "italy"] } },
            ],
        },
        {
            role: "tool" as const,
            content: [
                // call-B and call-C results ended up here instead of after the first assistant block
                { type: "tool-result" as const, toolCallId: "call-B", toolName: "search", output: { type: "text" as const, value: "Germany: Berlin" } },
                { type: "tool-result" as const, toolCallId: "call-C", toolName: "search", output: { type: "text" as const, value: "Italy: Rome" } },
                { type: "tool-result" as const, toolCallId: "call-D", toolName: "summarize", output: { type: "text" as const, value: "Summary ready." } },
            ],
        },
    ];

    console.log("Input prompt (malformed — B and C results in wrong block):");
    printPrompt("Input", messages as Parameters<typeof printPrompt>[1]);

    await generateText({ model, messages });

    console.log("\nPrompt received by model after tool ordering repair:");
    printPrompt("Repaired", capturedPrompts[0]);

    console.log("\nFixes applied:", fixes);
    console.log("\nWhat changed:");
    console.log("- call-B and call-C results moved to immediately after the first assistant block");
    console.log("- The second tool block now only contains call-D's result");
    console.log("- Both assistant blocks now have their results directly adjacent");
}

main().catch(console.error);
