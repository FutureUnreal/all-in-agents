import unittest

from all_in_agents import HistoryCompactor, HistoryManager, LLMResponse, ToolResponse


class RecordingSummaryLLM:
    def __init__(self):
        self.messages = []

    async def generate(self, messages, tools=None, system="", max_tokens=512, options=None):
        self.messages.append(messages)
        return LLMResponse(
            content='{"facts":[],"decisions":[],"open_threads":[]}',
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )


def tool_turn(tool_id: str, name: str, result_content: str, *, is_error: bool = False):
    result_block = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": result_content,
    }
    if is_error:
        result_block["is_error"] = True

    return [
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {"query": "x" * 300},
            }],
        },
        {"role": "user", "content": [result_block]},
    ]


class HistoryCompactorTests(unittest.IsolatedAsyncioTestCase):
    def test_micro_compact_preserves_head_and_tail(self):
        content = "BEGIN:" + ("x" * 300) + ":END"
        turns = [[{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": content}],
        }]]

        compacted = HistoryCompactor(micro_compact_max_chars=80).micro_compact_turns(turns)
        result = compacted[0][0]["content"][0]["content"]

        self.assertLessEqual(len(result), 80)
        self.assertTrue(result.startswith("BEGIN:"))
        self.assertTrue(result.endswith(":END"))
        self.assertIn("chars truncated", result)

    async def test_summarize_turns_compacts_old_tool_results(self):
        old_result = "OLD_BEGIN" + ("x" * 1000) + "OLD_END"
        recent_result = "RECENT_BEGIN keep this output RECENT_END"
        llm = RecordingSummaryLLM()
        compactor = HistoryCompactor(summary_keep_recent_tool_results=1)

        await compactor.summarize_turns(llm, [
            tool_turn("old_call", "read_file", old_result),
            tool_turn("recent_call", "bash", recent_result),
        ])

        prompt = llm.messages[0][0]["content"]
        self.assertNotIn("OLD_BEGIN", prompt)
        self.assertNotIn("OLD_END", prompt)
        self.assertIn("[tool_result: read_file -> success;", prompt)
        self.assertIn("RECENT_BEGIN keep this output RECENT_END", prompt)

    async def test_summarize_turns_marks_error_tool_results(self):
        llm = RecordingSummaryLLM()
        compactor = HistoryCompactor(summary_keep_recent_tool_results=0)

        await compactor.summarize_turns(llm, [
            tool_turn("call_1", "bash", "Traceback bottom error", is_error=True),
        ])

        prompt = llm.messages[0][0]["content"]
        self.assertIn("[tool_result: bash -> failed;", prompt)
        self.assertNotIn("Traceback bottom error", prompt)

    def test_history_manager_marks_error_tool_results(self):
        history = HistoryManager()

        history.add_tool_result("call_1", ToolResponse("error", "boom"))

        block = history.get_messages()[0]["content"][0]
        self.assertTrue(block["is_error"])


if __name__ == "__main__":
    unittest.main()
