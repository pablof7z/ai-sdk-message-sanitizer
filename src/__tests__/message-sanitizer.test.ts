import { describe, test, expect, beforeEach, mock } from "bun:test";
import type { LanguageModelV3Message } from "@ai-sdk/provider";
import type { LanguageModelV3 } from "@ai-sdk/provider";

// Track OTel span events
let spanEvents: Array<{ name: string; attributes?: Record<string, unknown> }> = [];

const mockSpan = {
    addEvent: (name: string, attributes?: Record<string, unknown>) => {
        spanEvents.push({ name, attributes });
    },
    setAttribute: mock(() => {}),
    setStatus: mock(() => {}),
    end: mock(() => {}),
    isRecording: () => true,
    recordException: mock(() => {}),
    updateName: mock(() => {}),
    setAttributes: mock(() => {}),
    spanContext: () => ({ traceId: "test", spanId: "test", traceFlags: 0 }),
};
const mockContext = {
    getValue: () => undefined,
    setValue: () => mockContext,
    deleteValue: () => mockContext,
};

mock.module("@opentelemetry/api", () => ({
    createContextKey: mock((name: string) => Symbol.for(name)),
    DiagLogLevel: { NONE: 0, ERROR: 1, WARN: 2, INFO: 3, DEBUG: 4, VERBOSE: 5, ALL: 6 },
    diag: {
        setLogger: mock(() => {}),
        debug: mock(() => {}),
        error: mock(() => {}),
        warn: mock(() => {}),
        info: mock(() => {}),
    },
    SpanKind: { INTERNAL: 0, SERVER: 1, CLIENT: 2, PRODUCER: 3, CONSUMER: 4 },
    ROOT_CONTEXT: mockContext,
    trace: {
        getActiveSpan: () => mockSpan,
        getTracer: () => ({
            startSpan: () => mockSpan,
            startActiveSpan: (_name: string, fn: (span: typeof mockSpan) => unknown) => fn(mockSpan),
        }),
        setSpan: () => mockContext,
    },
    SpanStatusCode: { ERROR: 2, OK: 1 },
    TraceFlags: { NONE: 0, SAMPLED: 1 },
    context: {
        active: () => mockContext,
        with: (_ctx: unknown, fn: () => unknown) => fn(),
    },
}));

import { createMessageSanitizerMiddleware, type MessageSanitizerFixEntry } from "../index";

function getToolCallIdsFromMsg(msg: LanguageModelV3Message): string[] {
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) return [];
    return (msg.content as Array<{ type: string; toolCallId?: string }>)
        .filter((p) => p.type === "tool-call" && typeof p.toolCallId === "string")
        .map((p) => p.toolCallId!);
}

function getToolResultIdsFromMsg(msg: LanguageModelV3Message): string[] {
    if (msg.role !== "tool" || !Array.isArray(msg.content)) return [];
    return (msg.content as Array<{ type: string; toolCallId?: string }>)
        .filter((p) => p.type === "tool-result" && typeof p.toolCallId === "string")
        .map((p) => p.toolCallId!);
}

const fakeModel: LanguageModelV3 = {
    specificationVersion: "v3",
    provider: "anthropic",
    modelId: "claude-opus-4-6",
    supportedUrls: {},
    doGenerate: async () => { throw new Error("not implemented"); },
    doStream: async () => { throw new Error("not implemented"); },
};

function makeParams(prompt: LanguageModelV3Message[]) {
    return {
        prompt,
        maxOutputTokens: 4096,
    };
}

describe("message-sanitizer middleware", () => {
    let fixEntries: MessageSanitizerFixEntry[];

    beforeEach(() => {
        spanEvents = [];
        fixEntries = [];
    });

    const middleware = createMessageSanitizerMiddleware({
        onFix: (entry) => fixEntries.push(entry),
    });
    const transformParams = middleware.transformParams!;

    describe("trailing assistant messages", () => {
        test("strips a single trailing assistant message", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "system", content: "You are a helpful assistant" },
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Hi there" }] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(2);
            expect(result.prompt[0].role).toBe("system");
            expect(result.prompt[1].role).toBe("user");
        });

        test("strips multiple consecutive trailing assistant messages", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Response 1" }] },
                { role: "assistant", content: [{ type: "text", text: "Response 2" }] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "generate",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(1);
            expect(result.prompt[0].role).toBe("user");
        });

        test("preserves trailing assistant tool-call messages", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Use a tool" }] },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-1",
                        toolName: "search",
                        input: { query: "test" },
                    }],
                },
            ];

            const params = makeParams(prompt);
            const result = await transformParams({
                params,
                type: "stream",
                model: fakeModel,
            });

            expect(result).toBe(params);
            expect(result.prompt).toEqual(prompt);
        });

        test("does not strip non-trailing assistant messages", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Hi" }] },
                { role: "user", content: [{ type: "text", text: "How are you?" }] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(3);
            expect(result.prompt).toEqual(prompt);
        });
    });

    describe("empty content messages", () => {
        test("strips user messages with empty content array", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "system", content: "You are helpful" },
                { role: "user", content: [] },
                { role: "user", content: [{ type: "text", text: "Real question" }] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(2);
            expect(result.prompt[0].role).toBe("system");
            expect(result.prompt[1].role).toBe("user");
            expect((result.prompt[1] as { role: "user"; content: Array<{ type: string; text: string }> }).content[0].text).toBe("Real question");
        });

        test("strips assistant messages with empty content array", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [] },
                { role: "user", content: [{ type: "text", text: "Still here?" }] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(2);
            expect(result.prompt[0].role).toBe("user");
            expect(result.prompt[1].role).toBe("user");
        });
    });

    describe("tool messages are never stripped", () => {
        test("preserves tool messages even with minimal content", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Use a tool" }] },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-1",
                        toolName: "search",
                        input: { query: "test" },
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-1",
                        toolName: "search",
                        output: { type: "text", value: "result" },
                    }],
                },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(3);
            expect(result.prompt).toEqual(prompt);
        });
    });

    describe("system messages are never stripped", () => {
        test("preserves system messages with empty string content", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "system", content: "" },
                { role: "user", content: [{ type: "text", text: "Hello" }] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(2);
            expect(result.prompt[0].role).toBe("system");
        });
    });

    describe("assistant tool-call inputs", () => {
        test("wraps malformed tool-call input into a dictionary", async () => {
            const prompt: LanguageModelV3Message[] = [
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-1",
                        toolName: "scratchpad",
                        input: "{\"setEntries\": \n<parameter name=\"objective\">debug it</parameter>",
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-1",
                        toolName: "scratchpad",
                        output: { type: "text", value: "Tool execution failed" },
                    }],
                },
                { role: "user", content: [{ type: "text", text: "Continue" }] },
            ];

            const params = makeParams(prompt);
            const result = await transformParams({
                params,
                type: "stream",
                model: fakeModel,
            });

            expect(result).not.toBe(params);

            const toolCallPart = (
                result.prompt[0] as { content: Array<Record<string, unknown>> }
            ).content[0];
            expect(toolCallPart.input).toEqual({
                _sanitizerInvalidInput: true,
                _sanitizerOriginalInputType: "string",
                rawInput: "{\"setEntries\": \n<parameter name=\"objective\">debug it</parameter>",
            });
        });
    });

    describe("onFix callback", () => {
        test("calls onFix with structured entry when fix is applied", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Trailing" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(fixEntries).toHaveLength(1);
            expect(fixEntries[0].fix).toBe("trailing-assistant-stripped");
            expect(fixEntries[0].model).toBe("anthropic:claude-opus-4-6");
            expect(fixEntries[0].callType).toBe("stream");
            expect(fixEntries[0].original_count).toBe(2);
            expect(fixEntries[0].fixed_count).toBe(1);
            expect(fixEntries[0].removed).toEqual([{ index: 1, role: "assistant" }]);
            expect(fixEntries[0].ts).toBeDefined();
        });

        test("does not call onFix when no fixes needed", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(fixEntries).toHaveLength(0);
        });

        test("calls onFix multiple times for multiple fixes", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [] },
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Trailing" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "generate",
                model: fakeModel,
            });

            expect(fixEntries).toHaveLength(2);
            expect(fixEntries[0].fix).toBe("empty-content-stripped");
            expect(fixEntries[1].fix).toBe("trailing-assistant-stripped");
        });

        test("calls onFix with per-message diagnostic entries when tool-call ordering is invalid", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-a",
                        toolName: "search",
                        input: { query: "a" },
                    }],
                },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-b",
                        toolName: "search",
                        input: { query: "b" },
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-a",
                        toolName: "search",
                        output: { type: "text", value: "result-a" },
                    }],
                },
                {
                    role: "user",
                    content: [{ type: "text", text: "Continue" }],
                },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const diagnosticEntries = fixEntries.filter((e) => e.fix === "invalid-tool-order-detected");
            expect(diagnosticEntries).toHaveLength(2);
            expect(diagnosticEntries[0]).toMatchObject({
                assistant_block_start: 1,
                next_block_start: 2,
                next_block_role: "assistant",
                tool_call_ids: ["call-a"],
                resolved_tool_call_ids: [],
                missing_tool_call_ids: ["call-a"],
            });
            expect(diagnosticEntries[1]).toMatchObject({
                assistant_block_start: 2,
                next_block_start: 3,
                next_block_role: "user",
                tool_call_ids: ["call-b"],
                resolved_tool_call_ids: ["call-a"],
                missing_tool_call_ids: ["call-b"],
            });
        });

        test("calls onFix when tool-call input is wrapped", async () => {
            const prompt: LanguageModelV3Message[] = [
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-1",
                        toolName: "scratchpad",
                        input: "{\"setEntries\": \n<parameter name=\"objective\">debug it</parameter>",
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-1",
                        toolName: "scratchpad",
                        output: { type: "text", value: "Tool execution failed" },
                    }],
                },
                { role: "user", content: [{ type: "text", text: "Continue" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const fixEntry = fixEntries.find((entry) => entry.fix === "tool-call-input-wrapped");
            expect(fixEntry).toBeDefined();
            expect(fixEntry!.model).toBe("anthropic:claude-opus-4-6");
            expect(fixEntry!.callType).toBe("stream");
            expect(fixEntry!.tool_call_id).toBe("call-1");
            expect(fixEntry!.tool_name).toBe("scratchpad");
            expect(fixEntry!.input_type).toBe("string");
            expect(fixEntry!.original_count).toBe(3);
            expect(fixEntry!.fixed_count).toBe(3);
        });

        test("works without onFix callback (no crash)", async () => {
            const middlewareNoCallback = createMessageSanitizerMiddleware();
            const transform = middlewareNoCallback.transformParams!;

            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Trailing" }] },
            ];

            const result = await transform({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(1);
        });
    });

    describe("OTel span events", () => {
        test("adds span event when fixes are applied", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Trailing" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const sanitizerEvents = spanEvents.filter(
                (e) => e.name === "message-sanitizer.fix-applied"
            );
            expect(sanitizerEvents).toHaveLength(1);

            const attrs = sanitizerEvents[0].attributes!;
            expect(attrs["sanitizer.fixes"]).toBe("trailing-assistant-stripped");
            expect(attrs["sanitizer.original_count"]).toBe(2);
            expect(attrs["sanitizer.fixed_count"]).toBe(1);
            expect(attrs["sanitizer.model"]).toBe("anthropic:claude-opus-4-6");
            expect(attrs["sanitizer.call_type"]).toBe("stream");
        });

        test("does not add span event when no fixes needed", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const sanitizerEvents = spanEvents.filter(
                (e) => e.name === "message-sanitizer.fix-applied"
            );
            expect(sanitizerEvents).toHaveLength(0);
        });

        test("adds a diagnostic span event when tool-call ordering is invalid", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-a",
                        toolName: "search",
                        input: { query: "a" },
                    }],
                },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-b",
                        toolName: "search",
                        input: { query: "b" },
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-a",
                        toolName: "search",
                        output: { type: "text", value: "result-a" },
                    }],
                },
                {
                    role: "user",
                    content: [{ type: "text", text: "Continue" }],
                },
            ];

            const params = makeParams(prompt);
            const result = await transformParams({
                params,
                type: "stream",
                model: fakeModel,
            });

            expect(result).not.toBe(params);
            expect(result.prompt).toEqual([
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-a",
                        toolName: "search",
                        input: { query: "a" },
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-a",
                        toolName: "search",
                        output: { type: "text", value: "result-a" },
                    }],
                },
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-b",
                        toolName: "search",
                        input: { query: "b" },
                    }],
                },
                {
                    role: "user",
                    content: [{ type: "text", text: "Continue" }],
                },
            ]);

            const diagnosticEvents = spanEvents.filter(
                (e) => e.name === "message-sanitizer.invalid-tool-order-detected"
            );
            expect(diagnosticEvents).toHaveLength(1);

            const attrs = diagnosticEvents[0].attributes!;
            expect(attrs["sanitizer.issue_count"]).toBe(2);
            expect(attrs["sanitizer.issue_block_starts"]).toBe("1,2");
            expect(attrs["sanitizer.missing_tool_call_ids"]).toBe("call-a,call-b");
        });

        test("adds a span event when tool-call input is wrapped", async () => {
            const prompt: LanguageModelV3Message[] = [
                {
                    role: "assistant",
                    content: [{
                        type: "tool-call",
                        toolCallId: "call-1",
                        toolName: "scratchpad",
                        input: "{\"setEntries\": \n<parameter name=\"objective\">debug it</parameter>",
                    }],
                },
                {
                    role: "tool",
                    content: [{
                        type: "tool-result",
                        toolCallId: "call-1",
                        toolName: "scratchpad",
                        output: { type: "text", value: "Tool execution failed" },
                    }],
                },
                { role: "user", content: [{ type: "text", text: "Continue" }] },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const repairEvents = spanEvents.filter(
                (e) => e.name === "message-sanitizer.tool-call-input-wrapped"
            );
            expect(repairEvents).toHaveLength(1);

            const attrs = repairEvents[0].attributes!;
            expect(attrs["sanitizer.repairs_count"]).toBe(1);
            expect(attrs["sanitizer.repaired_tool_call_ids"]).toBe("call-1");
            expect(attrs["sanitizer.input_types"]).toBe("string");
        });
    });

    describe("params passthrough", () => {
        test("returns params unchanged when no fixes needed", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "system", content: "System" },
                { role: "user", content: [{ type: "text", text: "Hello" }] },
            ];
            const params = makeParams(prompt);

            const result = await transformParams({
                params,
                type: "stream",
                model: fakeModel,
            });

            expect(result).toBe(params);
        });

        test("preserves other params when prompt is modified", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Hello" }] },
                { role: "assistant", content: [{ type: "text", text: "Trailing" }] },
            ];
            const params = {
                ...makeParams(prompt),
                temperature: 0.7,
                stopSequences: ["END"],
            };

            const result = await transformParams({
                params,
                type: "stream",
                model: fakeModel,
            });

            expect(result.temperature).toBe(0.7);
            expect(result.stopSequences).toEqual(["END"]);
            expect(result.prompt).toHaveLength(1);
        });
    });

    describe("tool ordering repair", () => {
        test("reorders consecutive assistant tool calls so each result follows its matching call", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [{ type: "tool-call", toolCallId: "call-1", toolName: "fs_read", input: {} }],
                },
                {
                    role: "assistant",
                    content: [{ type: "tool-call", toolCallId: "call-2", toolName: "fs_read", input: {} }],
                },
                {
                    role: "assistant",
                    content: [{ type: "tool-call", toolCallId: "call-3", toolName: "fs_read", input: {} }],
                },
                {
                    role: "tool",
                    content: [{ type: "tool-result", toolCallId: "call-1", toolName: "fs_read", output: { type: "text", value: "result-1" } }],
                },
                {
                    role: "tool",
                    content: [{ type: "tool-result", toolCallId: "call-2", toolName: "fs_read", output: { type: "text", value: "result-2" } }],
                },
                {
                    role: "tool",
                    content: [{ type: "tool-result", toolCallId: "call-3", toolName: "fs_read", output: { type: "text", value: "result-3" } }],
                },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const resultPrompt = result.prompt as LanguageModelV3Message[];
            expect(resultPrompt.map((message) => message.role)).toEqual([
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
            ]);
            expect(getToolCallIdsFromMsg(resultPrompt[1])).toEqual(["call-1"]);
            expect(getToolResultIdsFromMsg(resultPrompt[2])).toEqual(["call-1"]);
            expect(getToolCallIdsFromMsg(resultPrompt[3])).toEqual(["call-2"]);
            expect(getToolResultIdsFromMsg(resultPrompt[4])).toEqual(["call-2"]);
            expect(getToolCallIdsFromMsg(resultPrompt[5])).toEqual(["call-3"]);
            expect(getToolResultIdsFromMsg(resultPrompt[6])).toEqual(["call-3"]);
        });

        test("relocates misplaced tool results to the correct position", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [
                        { type: "tool-call", toolCallId: "call-J", toolName: "fs_read", input: {} },
                        { type: "tool-call", toolCallId: "call-L", toolName: "fs_read", input: {} },
                        { type: "tool-call", toolCallId: "call-H", toolName: "fs_read", input: {} },
                        { type: "tool-call", toolCallId: "call-12e", toolName: "fs_read", input: {} },
                    ],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-J", toolName: "fs_read", output: { type: "text", value: "result-J" } },
                    ],
                },
                {
                    role: "assistant",
                    content: [
                        { type: "tool-call", toolCallId: "call-Q", toolName: "fs_read", input: {} },
                        { type: "tool-call", toolCallId: "call-G", toolName: "fs_glob", input: {} },
                    ],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-12e", toolName: "fs_read", output: { type: "text", value: "result-12e" } },
                        { type: "tool-result", toolCallId: "call-H", toolName: "fs_read", output: { type: "text", value: "result-H" } },
                        { type: "tool-result", toolCallId: "call-L", toolName: "fs_read", output: { type: "text", value: "result-L" } },
                        { type: "tool-result", toolCallId: "call-G", toolName: "fs_glob", output: { type: "text", value: "result-G" } },
                        { type: "tool-result", toolCallId: "call-Q", toolName: "fs_read", output: { type: "text", value: "result-Q" } },
                    ],
                },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const resultPrompt = result.prompt as LanguageModelV3Message[];

            const firstAssistantIdx = resultPrompt.findIndex(
                (m) => m.role === "assistant" && getToolCallIdsFromMsg(m).includes("call-J")
            );
            expect(firstAssistantIdx).toBeGreaterThan(0);

            const toolResultsAfterFirst: string[] = [];
            for (let i = firstAssistantIdx + 1; i < resultPrompt.length; i++) {
                const msg = resultPrompt[i];
                if (msg.role === "tool") {
                    for (const part of msg.content as Array<{ type: string; toolCallId: string }>) {
                        if (part.type === "tool-result") toolResultsAfterFirst.push(part.toolCallId);
                    }
                } else {
                    break;
                }
            }

            expect(toolResultsAfterFirst.sort()).toEqual(["call-12e", "call-H", "call-J", "call-L"]);

            const secondAssistantIdx = resultPrompt.findIndex(
                (m, idx) => idx > firstAssistantIdx && m.role === "assistant" && getToolCallIdsFromMsg(m).includes("call-Q")
            );
            expect(secondAssistantIdx).toBeGreaterThan(firstAssistantIdx);

            const toolResultsAfterSecond: string[] = [];
            for (let i = secondAssistantIdx + 1; i < resultPrompt.length; i++) {
                const msg = resultPrompt[i];
                if (msg.role === "tool") {
                    for (const part of msg.content as Array<{ type: string; toolCallId: string }>) {
                        if (part.type === "tool-result") toolResultsAfterSecond.push(part.toolCallId);
                    }
                } else {
                    break;
                }
            }

            expect(toolResultsAfterSecond.sort()).toEqual(["call-G", "call-Q"]);
        });

        test("calls onFix with repair entry when tool results are relocated", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [
                        { type: "tool-call", toolCallId: "call-a", toolName: "search", input: {} },
                        { type: "tool-call", toolCallId: "call-b", toolName: "search", input: {} },
                    ],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-a", toolName: "search", output: { type: "text", value: "a" } },
                    ],
                },
                {
                    role: "assistant",
                    content: [{ type: "tool-call", toolCallId: "call-c", toolName: "search", input: {} }],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-b", toolName: "search", output: { type: "text", value: "b" } },
                        { type: "tool-result", toolCallId: "call-c", toolName: "search", output: { type: "text", value: "c" } },
                    ],
                },
            ];

            await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const repairEntry = fixEntries.find((e) => e.fix === "tool-ordering-repaired");
            expect(repairEntry).toBeDefined();
            expect((repairEntry!.repaired_tool_call_ids as string[]).includes("call-b")).toBe(true);

            const repairEvents = spanEvents.filter(
                (e) => e.name === "message-sanitizer.tool-ordering-repaired"
            );
            expect(repairEvents).toHaveLength(1);
        });

        test("does nothing when missing results are not in the prompt at all", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [
                        { type: "tool-call", toolCallId: "call-a", toolName: "search", input: {} },
                        { type: "tool-call", toolCallId: "call-b", toolName: "search", input: {} },
                    ],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-a", toolName: "search", output: { type: "text", value: "a" } },
                    ],
                },
                { role: "user", content: [{ type: "text", text: "Continue" }] },
            ];

            const params = makeParams(prompt);
            const result = await transformParams({
                params,
                type: "stream",
                model: fakeModel,
            });

            expect(result).toBe(params);

            expect(fixEntries.some((e) => e.fix === "invalid-tool-order-detected")).toBe(true);
            expect(fixEntries.some((e) => e.fix === "tool-ordering-repaired")).toBe(false);
        });

        test("removes empty tool messages after extracting parts", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [{ type: "text", text: "Start" }] },
                {
                    role: "assistant",
                    content: [
                        { type: "tool-call", toolCallId: "call-a", toolName: "t", input: {} },
                        { type: "tool-call", toolCallId: "call-b", toolName: "t", input: {} },
                    ],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-a", toolName: "t", output: { type: "text", value: "a" } },
                    ],
                },
                {
                    role: "assistant",
                    content: [{ type: "tool-call", toolCallId: "call-c", toolName: "t", input: {} }],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-b", toolName: "t", output: { type: "text", value: "b" } },
                    ],
                },
                {
                    role: "tool",
                    content: [
                        { type: "tool-result", toolCallId: "call-c", toolName: "t", output: { type: "text", value: "c" } },
                    ],
                },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            const resultPrompt = result.prompt as LanguageModelV3Message[];

            const emptyToolMessages = resultPrompt.filter(
                (m) => m.role === "tool" && Array.isArray(m.content) && m.content.length === 0
            );
            expect(emptyToolMessages).toHaveLength(0);

            const allToolResults = resultPrompt
                .filter((m) => m.role === "tool")
                .flatMap((m) => (m.content as Array<{ type: string; toolCallId: string }>))
                .filter((p) => p.type === "tool-result")
                .map((p) => p.toolCallId);

            expect(allToolResults).toContain("call-a");
            expect(allToolResults).toContain("call-b");
            expect(allToolResults).toContain("call-c");
        });
    });

    describe("combined sanitization", () => {
        test("handles empty content + trailing assistant in same pass", async () => {
            const prompt: LanguageModelV3Message[] = [
                { role: "user", content: [] },
                { role: "user", content: [{ type: "text", text: "Question" }] },
                { role: "assistant", content: [{ type: "text", text: "Answer" }] },
                { role: "assistant", content: [] },
            ];

            const result = await transformParams({
                params: makeParams(prompt),
                type: "stream",
                model: fakeModel,
            });

            expect(result.prompt).toHaveLength(1);
            expect(result.prompt[0].role).toBe("user");
        });
    });
});
