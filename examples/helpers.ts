import type {
    LanguageModelV3GenerateResult,
    LanguageModelV3Message,
    LanguageModelV3Middleware,
    LanguageModelV3Prompt,
    LanguageModelV3Usage,
} from "@ai-sdk/provider";
import { MockLanguageModelV3 } from "ai/test";

export function usage(): LanguageModelV3Usage {
    return {
        inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
        outputTokens: { total: 10, text: 10, reasoning: undefined },
    };
}

export function createMockModel(text = "ok"): MockLanguageModelV3 {
    return new MockLanguageModelV3({
        doGenerate: async (): Promise<LanguageModelV3GenerateResult> => ({
            content: [{ type: "text", text }],
            finishReason: { unified: "stop", raw: "stop" },
            usage: usage(),
            warnings: [],
        }),
    });
}

export function createPromptCaptureMiddleware(
    capturedPrompts: LanguageModelV3Prompt[]
): LanguageModelV3Middleware {
    return {
        specificationVersion: "v3",
        transformParams: async ({ params }) => {
            capturedPrompts.push(structuredClone(params.prompt));
            return params;
        },
    };
}

export function printPrompt(label: string, prompt: LanguageModelV3Message[]): void {
    console.log(`\n${label} (${prompt.length} messages)`);
    for (const [i, msg] of prompt.entries()) {
        if (msg.role === "system") {
            console.log(`  [${i}] system: "${(msg.content as string).slice(0, 60)}"`);
        } else if (msg.role === "assistant" || msg.role === "user") {
            const parts = msg.content as Array<{ type: string; text?: string; toolCallId?: string; toolName?: string }>;
            const summary = parts.map((p) => {
                if (p.type === "text") return `text:"${p.text?.slice(0, 40)}"`;
                if (p.type === "tool-call") return `tool-call(${p.toolName}/${p.toolCallId})`;
                return p.type;
            }).join(", ");
            console.log(`  [${i}] ${msg.role}: [${summary}]`);
        } else if (msg.role === "tool") {
            const parts = msg.content as Array<{ type: string; toolCallId?: string; toolName?: string }>;
            const summary = parts.map((p) => `tool-result(${p.toolName}/${p.toolCallId})`).join(", ");
            console.log(`  [${i}] tool: [${summary}]`);
        }
    }
}
