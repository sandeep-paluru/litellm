"""
Regression test for the databricks streaming chunk parser.

OpenAI-compatible servers (e.g. Vertex AI Model Garden vLLM endpoints) send a final
usage-only chunk with an empty `choices` list when `stream_options.include_usage` is
set. `chunk_parser` previously did `choices[0]` unconditionally, raising
`IndexError` -> `MidStreamFallbackError` and crashing the stream.
"""

import json
from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.types.utils import ModelResponseStream


def test_chunk_parser_handles_empty_choices_usage_chunk():
    """A usage-only final chunk (empty choices) must not raise IndexError."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert result.choices == []
    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 8


def test_chunk_parser_empty_choices_without_usage():
    """An empty-choices chunk with no usage block returns usage=None, no error."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result.choices == []
    assert getattr(result, "usage", None) is None


def test_chunk_parser_normal_content_chunk_still_works():
    """A regular content chunk is unaffected by the empty-choices guard."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result.choices[0].delta.content == "hi"


def test_chunk_parser_keeps_cached_tokens_on_usage_only_chunk():
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {
            "prompt_tokens": 340,
            "completion_tokens": 1,
            "total_tokens": 341,
            "prompt_tokens_details": {"cached_tokens": 320},
        },
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 320


def _sse_event(body: dict[str, object]) -> bytes:
    return f"data: {json.dumps(body)}\n\n".encode()


def _visible_stream_fields(chunk: Mapping[str, object] | ModelResponseStream) -> tuple[str, str, int | None]:
    if isinstance(chunk, Mapping):
        usage: Final = chunk.get("usage")
        prompt_tokens: Final = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        text: Final = chunk.get("text")
        finish_reason: Final = chunk.get("finish_reason")
        return (
            text if isinstance(text, str) else "",
            finish_reason if isinstance(finish_reason, str) else "",
            prompt_tokens if isinstance(prompt_tokens, int) else None,
        )
    stream_usage: Final = getattr(chunk, "usage", None)
    if not chunk.choices:
        return ("", "", None if stream_usage is None else stream_usage.prompt_tokens)
    return (
        chunk.choices[0].delta.content or "",
        chunk.choices[0].finish_reason or "",
        None if stream_usage is None else stream_usage.prompt_tokens,
    )


@pytest.mark.respx(assert_all_called=True)
def test_stream_usage_after_finish_chunk_reaches_the_caller(respx_mock: respx.MockRouter) -> None:
    chunk_header: Final = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "sabia-3"}
    stream_body: Final = b"".join(
        (
            _sse_event({**chunk_header, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "144"}}]}),
            _sse_event({**chunk_header, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            _sse_event(
                {
                    **chunk_header,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 340,
                        "completion_tokens": 1,
                        "total_tokens": 341,
                        "prompt_tokens_details": {"cached_tokens": 320},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )
    respx_mock.post("http://upstream.test/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_body)
    )

    stream: Final = litellm.completion(
        model="maritalk/sabia-3",
        api_base="http://upstream.test",
        api_key="sk-test",
        messages=[{"role": "user", "content": "What is 12*12?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    usages: Final = [chunk.usage for chunk in stream if getattr(chunk, "usage", None)]

    assert [(usage.prompt_tokens, usage.prompt_tokens_details.cached_tokens) for usage in usages] == [(340, 320)]


def test_chunk_parser_keeps_usage_on_the_finish_chunk():
    iterator: Final = ModelResponseIterator(streaming_response=None, sync_stream=True)

    result: Final = iterator.chunk_parser(
        chunk={
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 340, "completion_tokens": 1, "total_tokens": 341},
        }
    )

    assert result.choices[0].finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 340
    assert result.usage.completion_tokens == 1


_SSE_LINES_WITH_BLANK_SEPARATORS: Final = (
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"144"},"finish_reason":null}]}',
    "",
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "",
    "data: {not json",
    "",
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],'
    '"usage":{"prompt_tokens":340,"completion_tokens":1,"total_tokens":341}}',
    "",
    "data: [DONE]",
    "",
)


@pytest.mark.asyncio
async def test_async_stream_skips_blank_sse_lines_and_keeps_the_usage_chunk():
    async def lines():
        for line in _SSE_LINES_WITH_BLANK_SEPARATORS:
            yield line

    chunks: Final = [chunk async for chunk in ModelResponseIterator(streaming_response=lines(), sync_stream=False)]

    assert [_visible_stream_fields(chunk) for chunk in chunks] == [
        ("144", "", None),
        ("", "stop", None),
        ("", "", None),
        ("", "", 340),
    ]


class _BrokenTransport(ValueError):
    pass


def test_sync_stream_transport_error_surfaces_as_runtime_error():
    def lines():
        yield _SSE_LINES_WITH_BLANK_SEPARATORS[0]
        raise _BrokenTransport("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        list(ModelResponseIterator(streaming_response=lines(), sync_stream=True))


@pytest.mark.asyncio
async def test_async_stream_transport_error_surfaces_as_runtime_error():
    async def lines():
        yield _SSE_LINES_WITH_BLANK_SEPARATORS[0]
        raise _BrokenTransport("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        [chunk async for chunk in ModelResponseIterator(streaming_response=lines(), sync_stream=False)]


@pytest.mark.asyncio
async def test_async_stream_without_done_marker_ends_after_the_last_chunk():
    async def lines():
        for line in _SSE_LINES_WITH_BLANK_SEPARATORS[:4]:
            yield line

    chunks: Final = [chunk async for chunk in ModelResponseIterator(streaming_response=lines(), sync_stream=False)]

    assert [_visible_stream_fields(chunk)[:2] for chunk in chunks] == [("144", ""), ("", "stop")]


_CHUNK_HEADER: Final = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "glm"}


def _openai_chunk(**body: object) -> dict[str, object]:
    return {**_CHUNK_HEADER, **body}


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning_content": "7919 is prime"},
        {"reasoning": "7919 is prime"},
    ],
)
def test_chunk_parser_keeps_reasoning_content_deltas(delta: dict[str, str]) -> None:
    iterator: Final = ModelResponseIterator(streaming_response=None, sync_stream=True)

    result: Final = iterator.chunk_parser(
        chunk=_openai_chunk(choices=[{"index": 0, "delta": delta, "finish_reason": None}])
    )

    assert result.choices[0].delta.reasoning_content == "7919 is prime"
    assert not hasattr(result.choices[0].delta, "reasoning")


def test_chunk_parser_keeps_every_tool_call_delta() -> None:
    iterator: Final = ModelResponseIterator(streaming_response=None, sync_stream=True)
    opened: Final = iterator.chunk_parser(
        chunk=_openai_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "check_prime", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": ""},
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        )
    )
    followed: Final = iterator.chunk_parser(
        chunk=_openai_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"n":7919}'}},
                            {"index": 1, "function": {"arguments": '{"q":"7919"}'}},
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        )
    )

    assert [
        (call.index, call.id, call.function.name, call.function.arguments)
        for call in opened.choices[0].delta.tool_calls
    ] == [
        (0, "call_a", "check_prime", ""),
        (1, "call_b", "lookup", ""),
    ]
    assert [(call.index, call.function.arguments) for call in followed.choices[0].delta.tool_calls] == [
        (0, '{"n":7919}'),
        (1, '{"q":"7919"}'),
    ]


def test_chunk_parser_keeps_usage_details_on_the_final_chunk() -> None:
    iterator: Final = ModelResponseIterator(streaming_response=None, sync_stream=True)

    result: Final = iterator.chunk_parser(
        chunk=_openai_chunk(
            choices=[],
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 64,
                "total_tokens": 84,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        )
    )

    assert result.choices == []
    assert result.usage is not None
    assert result.usage.completion_tokens == 64
    assert result.usage.prompt_tokens == 20
    assert result.usage.total_tokens == 84
    assert result.usage.completion_tokens_details is not None
    assert result.usage.completion_tokens_details.reasoning_tokens == 50


def _sse_lines(bodies: tuple[Mapping[str, object], ...]) -> Iterator[str]:
    for body in bodies:
        yield f"data: {json.dumps(body)}"
    yield "data: [DONE]"


def test_vertex_openai_stream_keeps_reasoning_tool_calls_and_usage() -> None:
    bodies: Final = (
        _openai_chunk(
            choices=[
                {"index": 0, "delta": {"role": "assistant", "reasoning_content": "7919 is"}, "finish_reason": None}
            ]
        ),
        _openai_chunk(choices=[{"index": 0, "delta": {"reasoning_content": " prime."}, "finish_reason": None}]),
        _openai_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "check_prime", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": ""},
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        ),
        _openai_chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"n":7919}'}},
                            {"index": 1, "function": {"arguments": '{"q":"7919"}'}},
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        ),
        _openai_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        _openai_chunk(
            choices=[],
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 64,
                "total_tokens": 84,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        ),
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=ModelResponseIterator(streaming_response=_sse_lines(bodies), sync_stream=True),
        model="endpoint",
        custom_llm_provider="vertex_ai",
        stream_options={"include_usage": True},
        logging_obj=Logging(
            model="vertex_ai/openai/endpoint",
            messages=[{"role": "user", "content": "Is 7919 a prime number?"}],
            stream=True,
            call_type="completion",
            start_time=1.0,
            litellm_call_id="call-1",
            function_id="fn-1",
        ),
    )

    received: Final = list(wrapper)
    reasoning: Final = [
        chunk.choices[0].delta.reasoning_content
        for chunk in received
        if chunk.choices and getattr(chunk.choices[0].delta, "reasoning_content", None)
    ]
    tool_deltas: Final = [
        [(call.index, call.function.name, call.function.arguments) for call in chunk.choices[0].delta.tool_calls]
        for chunk in received
        if chunk.choices and chunk.choices[0].delta.tool_calls
    ]
    usages: Final = [chunk.usage for chunk in received if getattr(chunk, "usage", None) is not None]

    assert reasoning == ["7919 is", " prime."]
    assert tool_deltas == [
        [(0, "check_prime", ""), (1, "lookup", "")],
        [(0, None, '{"n":7919}'), (1, None, '{"q":"7919"}')],
    ]
    assert [(usage.prompt_tokens, usage.completion_tokens) for usage in usages] == [(20, 64)]
    assert usages[0].completion_tokens_details is not None
    assert usages[0].completion_tokens_details.reasoning_tokens == 50
    assembled: Final = litellm.stream_chunk_builder(
        chunks=received,
        messages=[{"role": "user", "content": "Is 7919 a prime number?"}],
    )
    assert assembled is not None
    assert assembled.choices[0].message.reasoning_content == "7919 is prime."
    assert [call.function.name for call in assembled.choices[0].message.tool_calls] == ["check_prime", "lookup"]
    assert assembled.usage.completion_tokens == 64
