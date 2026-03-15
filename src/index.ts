import type { LanguageModelMiddleware } from "ai";
import type {
    LanguageModelV3CallOptions,
    LanguageModelV3Message,
} from "@ai-sdk/provider";

export interface MessageSanitizerFixEntry {
    ts: string;
    fix: string;
    model: string;
    callType: string;
    [key: string]: unknown;
}

export interface MessageSanitizerOptions {
    onFix?: (entry: MessageSanitizerFixEntry) => void;
}

interface SanitizationWarning {
    fix: string;
    removed: Array<{ index: number; role: string }>;
}

interface ToolOrderingIssue {
    assistantBlockStart: number;
    nextBlockStart: number | null;
    nextBlockRole: "assistant" | "system" | "user" | "none";
    toolCallIds: string[];
    resolvedToolCallIds: string[];
    missingToolCallIds: string[];
}

interface ToolOrderingRepair {
    toolCallId: string;
    fromMessageIndex: number;
    insertedAfterIndex: number;
}

function getMessageBlockRole(msg: LanguageModelV3Message): "assistant" | "system" | "user" {
    if (msg.role === "tool" || msg.role === "user") return "user";
    return msg.role;
}

function getToolCallIds(msg: LanguageModelV3Message): string[] {
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) return [];

    const toolCallIds: string[] = [];
    for (const part of msg.content) {
        if (
            typeof part === "object" &&
            part !== null &&
            "type" in part &&
            part.type === "tool-call" &&
            "toolCallId" in part &&
            typeof part.toolCallId === "string"
        ) {
            toolCallIds.push(part.toolCallId);
        }
    }

    return toolCallIds;
}

function getToolResultIds(msg: LanguageModelV3Message): string[] {
    if (msg.role !== "tool" || !Array.isArray(msg.content)) return [];

    const toolResultIds: string[] = [];
    for (const part of msg.content) {
        if (
            typeof part === "object" &&
            part !== null &&
            "type" in part &&
            part.type === "tool-result" &&
            "toolCallId" in part &&
            typeof part.toolCallId === "string"
        ) {
            toolResultIds.push(part.toolCallId);
        }
    }

    return toolResultIds;
}

function hasToolCallContent(msg: LanguageModelV3Message): boolean {
    return getToolCallIds(msg).length > 0;
}

/**
 * Check if a user or assistant message has empty content (content: []).
 * System messages use string content (always valid).
 * Tool messages may legitimately have minimal content for adjacency.
 */
function hasEmptyContent(msg: LanguageModelV3Message): boolean {
    if (msg.role === "system" || msg.role === "tool") return false;
    return Array.isArray(msg.content) && msg.content.length === 0;
}

function detectToolOrderingIssues(prompt: LanguageModelV3Message[]): ToolOrderingIssue[] {
    const issues: ToolOrderingIssue[] = [];

    for (let i = 0; i < prompt.length;) {
        const blockStart = i;
        const blockRole = getMessageBlockRole(prompt[i]);
        const blockMessages: LanguageModelV3Message[] = [];

        while (i < prompt.length && getMessageBlockRole(prompt[i]) === blockRole) {
            blockMessages.push(prompt[i]);
            i++;
        }

        if (blockRole !== "assistant") {
            continue;
        }

        const toolCallIds = Array.from(new Set(blockMessages.flatMap(getToolCallIds)));
        if (toolCallIds.length === 0) {
            continue;
        }

        const nextBlockStart = i < prompt.length ? i : null;
        const nextBlockRole = nextBlockStart !== null
            ? getMessageBlockRole(prompt[nextBlockStart])
            : "none";

        const resolvedToolCallIds: string[] = [];
        if (nextBlockRole === "user" && nextBlockStart !== null) {
            let nextIndex = nextBlockStart;
            while (nextIndex < prompt.length && getMessageBlockRole(prompt[nextIndex]) === "user") {
                resolvedToolCallIds.push(...getToolResultIds(prompt[nextIndex]));
                nextIndex++;
            }
        }

        const uniqueResolvedToolCallIds = Array.from(new Set(resolvedToolCallIds));
        const missingToolCallIds = toolCallIds.filter(
            (toolCallId) => !uniqueResolvedToolCallIds.includes(toolCallId)
        );
        if (missingToolCallIds.length === 0) {
            continue;
        }

        issues.push({
            assistantBlockStart: blockStart,
            nextBlockStart,
            nextBlockRole,
            toolCallIds,
            resolvedToolCallIds: uniqueResolvedToolCallIds,
            missingToolCallIds,
        });
    }

    return issues;
}

/**
 * Repair tool ordering issues by relocating misplaced tool results.
 *
 * When an assistant block has tool_use IDs whose tool_result parts appear
 * later in the prompt (not immediately after), this function:
 * 1. Extracts those result parts from their current positions
 * 2. Inserts them into the user/tool block immediately following the assistant block
 * 3. Removes empty messages left behind after extraction
 *
 * Processes issues from end-to-start to avoid index shift cascades.
 */
function repairToolOrdering(prompt: LanguageModelV3Message[]): {
    result: LanguageModelV3Message[];
    repairs: ToolOrderingRepair[];
} {
    const issues = detectToolOrderingIssues(prompt);
    if (issues.length === 0) return { result: prompt, repairs: [] };

    // Deep-clone prompt so mutations are safe
    const messages = prompt.map((msg) => ({
        ...msg,
        content: Array.isArray(msg.content)
            ? (msg.content as unknown[]).map((part) =>
                typeof part === "object" && part !== null ? { ...part } : part
            ) as typeof msg.content
            : msg.content,
    })) as LanguageModelV3Message[];

    // Build index: toolCallId → message index (for tool-result parts)
    const resultLocationByCallId = new Map<string, number>();
    for (let i = 0; i < messages.length; i++) {
        for (const id of getToolResultIds(messages[i])) {
            resultLocationByCallId.set(id, i);
        }
    }

    const repairs: ToolOrderingRepair[] = [];

    // Process issues from end to start to keep indices stable
    const sortedIssues = [...issues].sort(
        (a, b) => b.assistantBlockStart - a.assistantBlockStart
    );

    for (const issue of sortedIssues) {
        const collectedParts: unknown[] = [];
        const messageIndicesToClean = new Set<number>();

        for (const missingId of issue.missingToolCallIds) {
            const msgIndex = resultLocationByCallId.get(missingId);
            if (msgIndex === undefined) continue;

            const msg = messages[msgIndex];
            if (msg.role !== "tool" || !Array.isArray(msg.content)) continue;

            const partIndex = (msg.content as Array<{ type?: string; toolCallId?: string }>).findIndex(
                (part) => part.type === "tool-result" && part.toolCallId === missingId
            );
            if (partIndex === -1) continue;

            collectedParts.push(msg.content[partIndex]);
            (msg.content as unknown[]).splice(partIndex, 1);
            messageIndicesToClean.add(msgIndex);

            repairs.push({
                toolCallId: missingId,
                fromMessageIndex: msgIndex,
                insertedAfterIndex: issue.nextBlockStart ?? issue.assistantBlockStart + 1,
            });
        }

        if (collectedParts.length === 0) continue;

        if (issue.nextBlockRole === "user" && issue.nextBlockStart !== null) {
            let lastToolMsgIndex = issue.nextBlockStart;
            while (
                lastToolMsgIndex + 1 < messages.length &&
                getMessageBlockRole(messages[lastToolMsgIndex + 1]) === "user"
            ) {
                lastToolMsgIndex++;
            }

            const newToolMessage = {
                role: "tool" as const,
                content: collectedParts,
            } as LanguageModelV3Message;
            messages.splice(lastToolMsgIndex + 1, 0, newToolMessage);

            const adjusted = new Set<number>();
            for (const cleanIdx of messageIndicesToClean) {
                adjusted.add(cleanIdx > lastToolMsgIndex ? cleanIdx + 1 : cleanIdx);
            }
            messageIndicesToClean.clear();
            for (const idx of adjusted) messageIndicesToClean.add(idx);
        } else {
            const insertAt = issue.nextBlockStart ?? messages.length;
            const newToolMessage = {
                role: "tool" as const,
                content: collectedParts,
            } as LanguageModelV3Message;
            messages.splice(insertAt, 0, newToolMessage);

            const adjusted = new Set<number>();
            for (const cleanIdx of messageIndicesToClean) {
                adjusted.add(cleanIdx >= insertAt ? cleanIdx + 1 : cleanIdx);
            }
            messageIndicesToClean.clear();
            for (const idx of adjusted) messageIndicesToClean.add(idx);
        }
    }

    const result = messages.filter((msg) => {
        if (msg.role !== "tool" || !Array.isArray(msg.content)) return true;
        return msg.content.length > 0;
    });

    return { result, repairs };
}

/**
 * Run all sanitization passes on the original prompt, collecting warnings
 * with indices in the original coordinate space.
 *
 * Pass 1: Mark empty-content user/assistant messages for removal.
 * Pass 2: Mark trailing assistant messages for removal (from the end,
 *          skipping already-marked indices).
 */
function sanitize(
    prompt: LanguageModelV3Message[]
): { result: LanguageModelV3Message[]; warnings: SanitizationWarning[] } {
    const warnings: SanitizationWarning[] = [];
    const removeSet = new Set<number>();

    // Pass 1: empty content
    const emptyRemoved: Array<{ index: number; role: string }> = [];
    for (let i = 0; i < prompt.length; i++) {
        if (hasEmptyContent(prompt[i])) {
            emptyRemoved.push({ index: i, role: prompt[i].role });
            removeSet.add(i);
        }
    }
    if (emptyRemoved.length > 0) {
        warnings.push({ fix: "empty-content-stripped", removed: emptyRemoved });
    }

    // Pass 2: trailing assistants (walk backwards, skipping already-removed)
    const trailingRemoved: Array<{ index: number; role: string }> = [];
    for (let i = prompt.length - 1; i >= 0; i--) {
        if (removeSet.has(i)) continue;
        if (prompt[i].role !== "assistant") break;
        if (hasToolCallContent(prompt[i])) break;
        trailingRemoved.push({ index: i, role: "assistant" });
        removeSet.add(i);
    }
    if (trailingRemoved.length > 0) {
        warnings.push({ fix: "trailing-assistant-stripped", removed: trailingRemoved });
    }

    const result = prompt.filter((_, i) => !removeSet.has(i));
    return { result, warnings };
}

async function getActiveSpan() {
    try {
        const { trace } = await import("@opentelemetry/api");
        return trace.getActiveSpan() ?? null;
    } catch {
        return null;
    }
}

/**
 * Creates a message sanitizer middleware that runs before every LLM API call.
 *
 * Fixes message array problems that would cause API rejection:
 * - Trailing assistant messages without tool calls (Anthropic rejects these)
 * - Empty-content user/assistant messages
 * - Misplaced tool results (tool ordering repair)
 *
 * It intercepts `params.prompt` via `transformParams`, covering all call paths:
 * initial streamText, prepareStep-rebuilt messages, generateText, and generateObject.
 *
 * When fixes are applied, `options.onFix` is called with a structured entry describing
 * what was changed. Use this to log warnings, emit metrics, or write to a file.
 */
export function createMessageSanitizerMiddleware(options: MessageSanitizerOptions = {}): LanguageModelMiddleware {
    return {
        specificationVersion: "v3" as const,

        transformParams: async ({ params, type, model }) => {
            const originalPrompt = params.prompt as LanguageModelV3Message[];
            const originalCount = originalPrompt.length;

            const { result: sanitized, warnings } = sanitize(originalPrompt);

            const toolOrderingIssues = detectToolOrderingIssues(sanitized);

            const { result: repaired, repairs } = toolOrderingIssues.length > 0
                ? repairToolOrdering(sanitized)
                : { result: sanitized, repairs: [] as ToolOrderingRepair[] };

            if (warnings.length === 0 && toolOrderingIssues.length === 0) {
                return params;
            }

            const modelId = `${model.provider}:${model.modelId}`;
            const allRemoved = warnings.flatMap((w) => w.removed);
            const finalPrompt = repairs.length > 0 ? repaired : sanitized;

            for (const warning of warnings) {
                options.onFix?.({
                    ts: new Date().toISOString(),
                    fix: warning.fix,
                    model: modelId,
                    callType: type,
                    original_count: originalCount,
                    fixed_count: finalPrompt.length,
                    removed: warning.removed,
                });
            }

            for (const issue of toolOrderingIssues) {
                options.onFix?.({
                    ts: new Date().toISOString(),
                    fix: "invalid-tool-order-detected",
                    model: modelId,
                    callType: type,
                    assistant_block_start: issue.assistantBlockStart,
                    next_block_start: issue.nextBlockStart,
                    next_block_role: issue.nextBlockRole,
                    tool_call_ids: issue.toolCallIds,
                    resolved_tool_call_ids: issue.resolvedToolCallIds,
                    missing_tool_call_ids: issue.missingToolCallIds,
                });
            }

            if (repairs.length > 0) {
                options.onFix?.({
                    ts: new Date().toISOString(),
                    fix: "tool-ordering-repaired",
                    model: modelId,
                    callType: type,
                    original_count: originalCount,
                    fixed_count: finalPrompt.length,
                    repairs_count: repairs.length,
                    repaired_tool_call_ids: repairs.map((r) => r.toolCallId),
                });
            }

            const span = await getActiveSpan();
            if (span) {
                if (warnings.length > 0) {
                    span.addEvent("message-sanitizer.fix-applied", {
                        "sanitizer.fixes": warnings.map((w) => w.fix).join(","),
                        "sanitizer.original_count": originalCount,
                        "sanitizer.fixed_count": finalPrompt.length,
                        "sanitizer.removed_indices": allRemoved.map((r) => r.index).join(","),
                        "sanitizer.removed_roles": allRemoved.map((r) => r.role).join(","),
                        "sanitizer.model": modelId,
                        "sanitizer.call_type": type,
                    });
                }

                if (toolOrderingIssues.length > 0) {
                    span.addEvent("message-sanitizer.invalid-tool-order-detected", {
                        "sanitizer.issue_count": toolOrderingIssues.length,
                        "sanitizer.issue_block_starts": toolOrderingIssues
                            .map((issue) => issue.assistantBlockStart)
                            .join(","),
                        "sanitizer.missing_tool_call_ids": toolOrderingIssues
                            .flatMap((issue) => issue.missingToolCallIds)
                            .join(","),
                        "sanitizer.model": modelId,
                        "sanitizer.call_type": type,
                    });
                }

                if (repairs.length > 0) {
                    span.addEvent("message-sanitizer.tool-ordering-repaired", {
                        "sanitizer.repairs_count": repairs.length,
                        "sanitizer.repaired_tool_call_ids": repairs.map((r) => r.toolCallId).join(","),
                        "sanitizer.model": modelId,
                        "sanitizer.call_type": type,
                    });
                }
            }

            if (warnings.length === 0 && repairs.length === 0) {
                return params;
            }

            return {
                ...params,
                prompt: finalPrompt as LanguageModelV3CallOptions["prompt"],
            };
        },
    };
}
