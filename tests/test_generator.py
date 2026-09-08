"""
Tests for Generator and PromptBuilder.

All tests patch ``src.generator.OpenAILike`` — Groq has been removed; the
NVIDIA endpoint is accessed exclusively via OpenAILike.
"""
import unittest
from unittest.mock import MagicMock, patch

from src.retriever import RetrievalResult, RetrievedChunk
from src.generator import Generator, PromptBuilder
from configs.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    DETAILED_THINKING_OFF,
    LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS,
)


def make_retrieval_result(num_chunks=2):
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            similarity_score=0.9,
            rank=i,
            page_number=str(i),
            source_file="doc.pdf",
            chunk_index=i,
            chunk_text=f"Chunk text number {i}.",
        )
        for i in range(1, num_chunks + 1)
    ]
    return RetrievalResult(
        question="What does the document say?",
        question_embedding_dimension=3,
        retrieved_chunks=chunks,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        similarity_scores=[c.similarity_score for c in chunks],
        retrieval_time=0.1,
        top_k=num_chunks,
        retrieval_metadata={},
    )


def make_chat_response(content):
    response = MagicMock()
    response.message.content = content
    return response


class TestGenerator(unittest.TestCase):
    @patch("src.generator.OpenAILike")
    def test_generate_returns_generation_result_fields(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_chat_response("This is the generated answer.")
        mock_llm_cls.return_value = mock_llm

        retrieval_result = make_retrieval_result()
        generator = Generator()
        result = generator.generate(retrieval_result)

        self.assertEqual(result.generated_answer, "This is the generated answer.")
        self.assertEqual(result.question, retrieval_result.question)
        self.assertEqual(result.retrieved_chunk_ids, retrieval_result.retrieved_chunk_ids)
        self.assertGreater(result.prompt_length, 0)

    @patch("src.generator.OpenAILike")
    def test_nvidia_uses_max_tokens_kwarg(self, mock_llm_cls):
        mock_llm_cls.return_value = MagicMock()
        Generator(max_tokens=2048)

        _, kwargs = mock_llm_cls.call_args
        self.assertEqual(kwargs.get("max_tokens"), 2048)

    @patch("src.generator.OpenAILike")
    def test_nvidia_passes_api_base(self, mock_llm_cls):
        mock_llm_cls.return_value = MagicMock()
        Generator()

        _, kwargs = mock_llm_cls.call_args
        self.assertIn("integrate.api.nvidia.com", kwargs.get("api_base", ""))

    @patch("src.generator.OpenAILike")
    def test_prompt_length_and_chunk_ids_passthrough(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_chat_response("Answer.")
        mock_llm_cls.return_value = mock_llm

        retrieval_result = make_retrieval_result(num_chunks=1)
        generator = Generator()
        result = generator.generate(retrieval_result)

        self.assertEqual(result.prompt_length, len(result.prompt))
        self.assertEqual(result.retrieved_chunk_ids, ["c1"])

    @patch("src.generator.OpenAILike")
    def test_llm_error_is_captured_in_generated_answer(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("boom")
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        result = generator.generate(make_retrieval_result())

        # The failure must be surfaced as an error, not silently become the
        # answer -- callers check .ok before persisting a turn.
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.error)
        self.assertIn("did not respond", result.generated_answer)

    @patch("src.generator.OpenAILike")
    def test_per_call_system_instructions_override(self, mock_llm_cls):
        """generate() should honour a per-call system_instructions kwarg."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_chat_response("Answer.")
        mock_llm_cls.return_value = mock_llm

        generator = Generator(system_instructions="Default instructions.")
        generator.generate(
            make_retrieval_result(),
            system_instructions="Override instructions.",
        )

        # The chat call should have seen the override, not the constructor value.
        # messages[0] is the fixed "detailed thinking off" toggle (see
        # PromptBuilder.build_messages); messages[1] is the instructions.
        call_args = mock_llm.chat.call_args[0][0]  # first positional: messages list
        system_content = call_args[1].content
        self.assertEqual(system_content, "Override instructions.")

    def test_default_instructions_are_domain_agnostic(self):
        self.assertNotIn("legal", DEFAULT_SYSTEM_INSTRUCTIONS.lower())

    def test_legal_preset_available(self):
        self.assertIn("legal assistant", LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS.lower())

    def test_build_messages_starts_with_thinking_off_toggle(self):
        """nemotron-3-super only honours "detailed thinking off" as its own
        standalone first system message -- see PromptBuilder.build_messages."""
        messages = PromptBuilder.build_messages("Q?", [])
        self.assertEqual(messages[0].content, DETAILED_THINKING_OFF)

    def test_build_messages_uses_default_instructions_by_default(self):
        messages = PromptBuilder.build_messages("Q?", [])
        self.assertEqual(messages[1].content, DEFAULT_SYSTEM_INSTRUCTIONS)

    def test_build_messages_override(self):
        custom = "You are a custom assistant."
        messages = PromptBuilder.build_messages("Q?", [], system_instructions=custom)
        self.assertEqual(messages[1].content, custom)

    def test_build_messages_override_legal_preset(self):
        messages = PromptBuilder.build_messages(
            "Q?", [], system_instructions=LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS
        )
        self.assertEqual(messages[1].content, LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS)

    def test_context_chunk_includes_chunk_id(self):
        chunk = RetrievedChunk(
            chunk_id="chunk-42",
            similarity_score=0.9,
            rank=1,
            page_number="1",
            source_file="doc.pdf",
            chunk_index=0,
            chunk_text="Some text.",
        )
        messages = PromptBuilder.build_messages("Q?", [chunk])
        self.assertIn("[Chunk-ID: chunk-42]", messages[2].content)

    def test_chunks_are_reversed_in_prompt(self):
        """Highest-ranked chunk (rank 1) should appear LAST in the context."""
        chunks = [
            RetrievedChunk(
                chunk_id=f"c{i}", similarity_score=0.9, rank=i,
                page_number=str(i), source_file="doc.pdf",
                chunk_index=i, chunk_text=f"Text rank {i}."
            )
            for i in range(1, 4)
        ]
        messages = PromptBuilder.build_messages("Q?", chunks)
        user_content = messages[2].content
        # rank-1 text should appear closer to the end than rank-3 text
        pos_rank1 = user_content.index("Text rank 1.")
        pos_rank3 = user_content.index("Text rank 3.")
        self.assertGreater(pos_rank1, pos_rank3)


def make_stream_chunks(deltas, finish_reasons=None):
    """finish_reasons, if given, must be the same length as deltas -- mirrors
    the real API, which sets finish_reason=None on every chunk but the last."""
    chunks = []
    for i, d in enumerate(deltas):
        c = MagicMock()
        c.delta = d
        reason = finish_reasons[i] if finish_reasons else None
        c.raw.choices = [MagicMock(finish_reason=reason)]
        chunks.append(c)
    return chunks


class TestGenerateStreamReasoning(unittest.TestCase):
    """generate_stream(reasoning=...) drives the DeepSeek-style thinking/
    answer split used by the chat SSE endpoint. reasoning=False must stay
    byte-for-byte the old plain-string behaviour (ui/app.py's Streamlit path
    still calls it that way)."""

    @patch("src.generator.OpenAILike")
    def test_plain_mode_yields_str_deltas_unchanged(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.stream_chat.return_value = make_stream_chunks(["Hello, ", "world."])
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        deltas = list(generator.generate_stream(make_retrieval_result()))

        self.assertEqual(deltas, ["Hello, ", "world."])
        self.assertEqual(generator.last_stream_result.generated_answer, "Hello, world.")
        self.assertEqual(generator.last_stream_result.reasoning, "")

    @patch("src.generator.OpenAILike")
    def test_reasoning_mode_splits_thinking_and_answer_across_chunk_boundaries(self, mock_llm_cls):
        mock_llm = MagicMock()
        # Split the tags themselves across deltas to exercise the
        # whole-buffer re-match, not just a same-delta happy path.
        full = "<thinking>Step one. Step two.</thinking><answer>Final answer.</answer>"
        mock_llm.stream_chat.return_value = make_stream_chunks(
            [full[i:i + 7] for i in range(0, len(full), 7)]
        )
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        items = list(generator.generate_stream(make_retrieval_result(), reasoning=True))

        reasoning_text = "".join(d for kind, d in items if kind == "reasoning")
        answer_text = "".join(d for kind, d in items if kind == "answer")
        self.assertEqual(reasoning_text, "Step one. Step two.")
        self.assertEqual(answer_text, "Final answer.")
        self.assertEqual(generator.last_stream_result.generated_answer, "Final answer.")
        self.assertEqual(generator.last_stream_result.reasoning, "Step one. Step two.")
        # The eval/citation pipeline parses generated_answer -- the tags and
        # the thinking text must never leak into it.
        self.assertNotIn("<thinking>", generator.last_stream_result.generated_answer)
        self.assertNotIn("Step one", generator.last_stream_result.generated_answer)

    @patch("src.generator.OpenAILike")
    def test_reasoning_mode_degrades_gracefully_without_tags(self, mock_llm_cls):
        """If the model ignores the think-aloud instruction and free-forms a
        plain reply, that reply must still surface as the answer, not be
        silently dropped because no <answer> tag ever matched."""
        mock_llm = MagicMock()
        mock_llm.stream_chat.return_value = make_stream_chunks(["Just ", "a ", "plain reply."])
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        list(generator.generate_stream(make_retrieval_result(), reasoning=True))

        self.assertEqual(generator.last_stream_result.generated_answer, "Just a plain reply.")
        self.assertEqual(generator.last_stream_result.reasoning, "")

    @patch("src.generator.OpenAILike")
    def test_finish_reason_length_appends_truncation_notice(self, mock_llm_cls):
        """A stream that ends with finish_reason="length" was cut off by
        max_tokens, not because the model was done -- that must be visible in
        the answer, not indistinguishable from a normal completion."""
        mock_llm = MagicMock()
        mock_llm.stream_chat.return_value = make_stream_chunks(
            ["Partial answer, still going"], finish_reasons=["length"]
        )
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        deltas = list(generator.generate_stream(make_retrieval_result()))

        full_text = "".join(deltas)
        self.assertIn("Partial answer, still going", full_text)
        self.assertIn("cut off", full_text.lower())
        self.assertIsNone(generator.last_stream_result.error)

    @patch("src.generator.OpenAILike")
    def test_finish_reason_stop_has_no_truncation_notice(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.stream_chat.return_value = make_stream_chunks(
            ["A complete answer."], finish_reasons=["stop"]
        )
        mock_llm_cls.return_value = mock_llm

        generator = Generator()
        deltas = list(generator.generate_stream(make_retrieval_result()))

        self.assertEqual("".join(deltas), "A complete answer.")


if __name__ == "__main__":
    unittest.main()
