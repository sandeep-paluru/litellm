import json
from typing import Final

import litellm
from litellm import verbose_logger
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            return litellm.ModelResponseStream(**chunk)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        self.response_iterator = self.streaming_response
        return self

    def __next__(self):
        if not hasattr(self, "response_iterator"):
            self.response_iterator = self.streaming_response
        chunk: Final = self._next_sse_data()
        try:
            json_chunk: Final = json.loads(chunk)
            return self.chunk_parser(chunk=json_chunk)
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return _empty_generic_chunk()

    def _next_sse_data(self) -> str:
        try:
            for raw_chunk in self.response_iterator:
                if sse_data := _sse_data(raw_chunk):
                    return sse_data
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")
        raise StopIteration

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        chunk: Final = await self._anext_sse_data()
        if chunk == "[DONE]":
            raise StopAsyncIteration
        try:
            json_chunk: Final = json.loads(chunk)
            return self.chunk_parser(chunk=json_chunk)
        except ValueError as e:
            verbose_logger.debug(
                "Error parsing chunk: %s,\nReceived chunk: %s. Defaulting to empty chunk here.", e, chunk
            )
            return _empty_generic_chunk()

    async def _anext_sse_data(self) -> str:
        try:
            async for raw_chunk in self.async_response_iterator:
                if sse_data := _sse_data(raw_chunk):
                    return sse_data
        except Exception as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")
        raise StopAsyncIteration


def _sse_data(raw_chunk: str) -> str:
    return (litellm.CustomStreamWrapper.strip_sse_data_from_chunk(raw_chunk) or "").strip()


def _empty_generic_chunk() -> GenericStreamingChunk:
    return GenericStreamingChunk(
        text="",
        is_finished=False,
        finish_reason="",
        usage=None,
        index=0,
        tool_use=None,
    )
