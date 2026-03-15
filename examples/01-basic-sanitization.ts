/**
 * Basic sanitization — trailing assistant messages and empty content
 *
 * What to look for:
 * - The trailing assistant message is stripped before the call reaches the model
 * - The empty user message is removed
 * - The model receives a clean, valid prompt
 */
import { generateText, wrapLanguageModel } from "ai";
import type { LanguageModelV3Prompt } from "@ai-sdk/provider";
import { createMessageSanitizerMiddleware } from "ai-sdk-message-sanitizer";
import { createMockModel, createPromptCaptureMiddleware, printPrompt } from "./helpers.js";

async function main() {
    const capturedPrompts: LanguageModelV3Prompt[] = [];

    const model = wrapLanguageModel({
        model: wrapLanguageModel({
            model: createMockModel(),
            middleware: createPromptCaptureMiddleware(capturedPrompts),
        }),
        middleware: createMessageSanitizerMiddleware(),
    });

    const messages = [
        { role: "system" as const, content: "You are a helpful assistant." },
        // Empty user message — invalid, will be removed
        { role: "user" as const, content: [] },
        { role: "user" as const, content: [{ type: "text" as const, text: "What is the capital of France?" }] },
        { role: "assistant" as const, content: [{ type: "text" as const, text: "Paris." }] },
        // Trailing assistant message — Anthropic rejects this, will be removed
        { role: "assistant" as const, content: [{ type: "text" as const, text: "Let me know if you need more info." }] },
    ];

    console.log("Input prompt (5 messages — 1 empty user, 2 trailing assistants, 1 system, 1 real user):");
    printPrompt("Input", messages as Parameters<typeof printPrompt>[1]);

    await generateText({ model, messages });

    console.log("\nPrompt received by model after sanitization:");
    printPrompt("Sanitized", capturedPrompts[0]);

    console.log("\nWhat changed:");
    console.log("- Empty user message at index 1 removed");
    console.log("- Both trailing assistant messages removed (only the last user matters for the next turn)");
    console.log("- System message preserved");
    console.log("- Real user message preserved");
}

main().catch(console.error);
