# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import httpx2 as httpx
import openai
from openai._models import FinalRequestOptions

from livekit.agents.llm import ChatContext, Tool, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM
from livekit.plugins.openai.llm import LLMStream

from ._common import client_headers, openai_base_url, resolve_api_key
from .models import SUPPORTED_SAMPLING, ChatModels

# The openai client refuses to start without a key. Anonymous clients get this placeholder,
# and _GatewayClient makes sure it never reaches the wire.
_ANONYMOUS_KEY = "vlmrun-anonymous"


class _GatewayClient(openai.AsyncClient):
    """AsyncClient that omits the Authorization header in anonymous mode.

    The gateway's anonymous tier accepts requests with no Authorization header, but answers
    any bearer token it doesn't recognize with 401, so a placeholder key can't be sent.
    """

    def __init__(self, *, anonymous: bool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._anonymous = anonymous

    def _build_request(
        self, options: FinalRequestOptions, *, retries_taken: int = 0
    ) -> httpx.Request:
        if self._anonymous:
            existing = (
                dict(options.headers) if is_given(options.headers) and options.headers else {}
            )
            options.headers = {**existing, "Authorization": openai.Omit()}
        return super()._build_request(options, retries_taken=retries_taken)


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "qwen/qwen3.8-27b",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of a VLM Run gateway LLM.

        The VLM Run gateway serves open-weight and hosted vision-language models through an
        OpenAI-compatible Chat Completions API. Images in the chat context (``ImageContent``,
        including frames from a video track) are sent to the model.

        ``api_key`` defaults to the ``VLMRUN_API_KEY`` environment variable. Without a key the
        gateway's anonymous tier is used: free models only (such as ``qwen/qwen3.5-0.8b``) and
        low rate limits. ``base_url`` defaults to ``VLMRUN_GATEWAY_BASE_URL`` or
        ``https://gateway.vlm.run``.

        Tool calling is supported by ``qwen/qwen3.8-27b`` and the hosted models (Gemini, Kimi,
        MiniMax, Muse), not by ``qwen/qwen3.5-0.8b`` or the Gemma models.

        Only parameters you set are sent. ``temperature`` and ``top_p`` are checked against what
        the model accepts, since the gateway rejects unsupported sampling fields.

        Args:
            model: Gateway model id. Defaults to ``qwen/qwen3.8-27b``.
            api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``; omit for anonymous use.
            base_url: Gateway root URL, without the ``/v1/openai`` suffix.
            client: A preconfigured ``openai.AsyncClient``; overrides key, URL and timeout.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            parallel_tool_calls: Allow parallel tool calls.
            tool_choice: Tool choice strategy.
            max_completion_tokens: Maximum tokens per reply.
            extra_body: Gateway-specific fields, such as ``method`` or ``method_params``.
            timeout: HTTP timeout.
            max_retries: Retries made by the openai client.
        """
        _check_sampling(model, temperature=temperature, top_p=top_p)

        key = resolve_api_key(api_key if is_given(api_key) else None)
        created_client = False
        if client is None:
            client = _GatewayClient(
                anonymous=key is None,
                api_key=key or _ANONYMOUS_KEY,
                base_url=openai_base_url(base_url if is_given(base_url) else None),
                default_headers=client_headers(None),
                max_retries=max_retries if is_given(max_retries) else 0,
                http_client=httpx.AsyncClient(
                    timeout=timeout
                    if timeout
                    else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=120,
                    ),
                ),
            )
            created_client = True

        super().__init__(
            model=model,
            client=client,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
            _strict_tool_schema=False,
        )

        if created_client:
            self._owns_client = True

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        # The gateway rejects a request without a user message ("No user query found in
        # messages"), which is what generate_reply(instructions=...) sends for a greeting.
        # Same placeholder the anthropic, aws and mistralai formats insert.
        if not any(item.type == "message" and item.role == "user" for item in chat_ctx.items):
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(role="user", content="(empty)")
        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_kwargs=extra_kwargs,
        )


def _check_sampling(
    model: str, *, temperature: NotGivenOr[float], top_p: NotGivenOr[float]
) -> None:
    supported = SUPPORTED_SAMPLING.get(model)
    if supported is None:
        return
    for name, value in (("temperature", temperature), ("top_p", top_p)):
        if is_given(value) and name not in supported:
            raise ValueError(f"VLM Run model {model!r} does not accept {name!r}")
