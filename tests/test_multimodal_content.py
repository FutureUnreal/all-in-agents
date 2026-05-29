import tempfile
import unittest

from all_in_agents import (
    Agent,
    FileBase64Block,
    FileIdBlock,
    FileUrlBlock,
    ImageBase64Block,
    ImageUrlBlock,
    LLMAdapter,
    LLMResponse,
    TextBlock,
    ToolRegistry,
)
from all_in_agents.adapters.anthropic import AnthropicAdapter
from all_in_agents.adapters.openai_chat import convert_chat_messages
from all_in_agents.adapters.openai_responses import convert_responses_input
from all_in_agents.core.tokens import estimate_content_tokens


class RecordingLLM(LLMAdapter):
    model_id = "recording"
    max_context_tokens = 1000

    def __init__(self):
        self.messages = []

    async def generate(self, messages, tools=None, system="", max_tokens=2048, options=None):
        self.messages.append(messages)
        return LLMResponse(
            content="done",
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )


class MultimodalContentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_accepts_text_and_image_initial_messages(self):
        llm = RecordingLLM()

        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(llm, ToolRegistry(), run_dir=tmp)
            await agent.run(
                "describe it",
                initial_messages=[{
                    "role": "user",
                    "content": [
                        TextBlock("Look at this"),
                        ImageUrlBlock("https://example.com/image.png", detail="low"),
                    ],
                }],
            )

        self.assertEqual(llm.messages[0][0]["content"], [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "url": "https://example.com/image.png", "detail": "low"},
        ])

    def test_openai_chat_converts_image_and_file_blocks(self):
        converted = convert_chat_messages([{
            "role": "user",
            "content": [
                TextBlock("What is in this image?").to_dict(),
                ImageUrlBlock("https://example.com/cat.png", detail="high").to_dict(),
                ImageBase64Block("abc123", media_type="image/png").to_dict(),
                FileBase64Block("pdf123", filename="report.pdf").to_dict(),
                FileIdBlock("file_123", filename="uploaded.pdf").to_dict(),
            ],
        }])

        parts = converted[0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "What is in this image?"})
        self.assertEqual(parts[1]["image_url"]["url"], "https://example.com/cat.png")
        self.assertEqual(parts[1]["image_url"]["detail"], "high")
        self.assertEqual(parts[2]["image_url"]["url"], "data:image/png;base64,abc123")
        self.assertEqual(parts[3], {
            "type": "file",
            "file": {"file_data": "pdf123", "filename": "report.pdf"},
        })
        self.assertEqual(parts[4], {
            "type": "file",
            "file": {"file_id": "file_123", "filename": "uploaded.pdf"},
        })

    def test_openai_chat_rejects_file_url_blocks(self):
        with self.assertRaises(ValueError):
            convert_chat_messages([{
                "role": "user",
                "content": [FileUrlBlock("https://example.com/report.pdf").to_dict()],
            }])

    def test_openai_responses_converts_image_and_file_blocks(self):
        converted = convert_responses_input([{
            "role": "user",
            "content": [
                TextBlock("What is in this image?").to_dict(),
                ImageUrlBlock("https://example.com/cat.png", detail="low").to_dict(),
                FileUrlBlock("https://example.com/report.pdf", filename="report.pdf").to_dict(),
                FileBase64Block("pdf123", filename="local.pdf").to_dict(),
                FileIdBlock("file_123", filename="uploaded.pdf").to_dict(),
            ],
        }])

        content = converted[0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "What is in this image?"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], "https://example.com/cat.png")
        self.assertEqual(content[1]["detail"], "low")
        self.assertEqual(content[2], {
            "type": "input_file",
            "file_url": "https://example.com/report.pdf",
            "filename": "report.pdf",
        })
        self.assertEqual(content[3], {
            "type": "input_file",
            "file_data": "pdf123",
            "filename": "local.pdf",
        })
        self.assertEqual(content[4], {
            "type": "input_file",
            "file_id": "file_123",
            "filename": "uploaded.pdf",
        })

    def test_anthropic_converts_image_and_file_blocks(self):
        converted = AnthropicAdapter._convert_messages([{
            "role": "user",
            "content": [
                TextBlock("Describe").to_dict(),
                ImageUrlBlock("https://example.com/cat.png").to_dict(),
                ImageBase64Block("abc123", media_type="image/png").to_dict(),
                FileUrlBlock("https://example.com/report.pdf", filename="report.pdf").to_dict(),
                FileBase64Block("pdf123", filename="local.pdf").to_dict(),
                FileIdBlock("file_123", filename="uploaded.pdf").to_dict(),
            ],
        }])

        content = converted[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe"})
        self.assertEqual(content[1]["source"], {"type": "url", "url": "https://example.com/cat.png"})
        self.assertEqual(content[2]["source"], {
            "type": "base64",
            "media_type": "image/png",
            "data": "abc123",
        })
        self.assertEqual(content[3], {
            "type": "document",
            "source": {"type": "url", "url": "https://example.com/report.pdf"},
            "title": "report.pdf",
        })
        self.assertEqual(content[4], {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "pdf123"},
            "title": "local.pdf",
        })
        self.assertEqual(content[5], {
            "type": "document",
            "source": {"type": "file", "file_id": "file_123"},
            "title": "uploaded.pdf",
        })

    def test_media_blocks_use_fixed_token_estimate(self):
        content = [
            TextBlock("short").to_dict(),
            ImageBase64Block("x" * 40_000, detail="low").to_dict(),
            FileBase64Block("x" * 400_000, filename="report.pdf").to_dict(),
        ]

        self.assertLess(estimate_content_tokens(content), 2000)


if __name__ == "__main__":
    unittest.main()
