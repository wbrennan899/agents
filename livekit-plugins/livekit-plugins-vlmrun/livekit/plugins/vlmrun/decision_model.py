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

import asyncio
import base64
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    llm,
    utils,
)
from livekit.agents.llm import _systemone as systemone
from livekit.agents.llm.decisions import Decision, DecisionCapabilities, DecisionResponse
from livekit.agents.llm.utils import serialize_image
from livekit.agents.types import APIConnectOptions

from ._common import client_headers, resolve_api_key, systemone_url

DEFAULT_DECISION_MODEL = "google/diffusiongemma-26b-a4b-it"
MAX_IMAGES = 8
"""The gateway reads at most 8 images per request."""

ImageDetail = Literal["auto", "low", "high"]


class DecisionModel(llm.DecisionModel):
    """Typed decisions over text and images, read by a VLM Run gateway model.

    Declares ``image_input``, so every ``ImageContent`` in a user message (a URL, data
    URL, or video frame) and in ``evaluate(images=...)`` goes with the transcript, oldest
    first. Only the newest 8 are sent, the gateway's limit.

    Args:
        model: A model served on the gateway's System One route.
        api_key: Defaults to VLMRUN_API_KEY. Without a key, requests use the gateway's
            anonymous tier, which serves free models at lower rate limits.
        base_url: Gateway root. Defaults to VLMRUN_GATEWAY_BASE_URL or
            ``https://gateway.vlm.run``.
        detail: Image detail for every image. ``low`` costs a quarter of the tokens.
            Defaults to each image's ``inference_detail``.
        http_session: Optional caller-owned session. Otherwise uses the job's HTTP
            session. Outside a job, use ``async with utils.http_context.open()``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_DECISION_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        detail: ImageDetail | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(
                decision_kinds=["probability", "choice", "score"], image_input=True
            )
        )
        self._model = model
        self._api_key = resolve_api_key(api_key)
        self._endpoint = systemone_url(base_url)
        self._detail = detail
        self._session = http_session

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "vlmrun"

    def _image_part(self, image: llm.ImageContent) -> dict[str, Any]:
        serialized = serialize_image(image)
        if serialized.external_url:
            url = serialized.external_url
        else:
            assert serialized.data_bytes is not None
            encoded = base64.b64encode(serialized.data_bytes).decode("utf-8")
            url = f"data:{serialized.mime_type or 'image/jpeg'};base64,{encoded}"
        detail = self._detail or serialized.inference_detail
        return {"type": "image_url", "image_url": {"url": url, "detail": detail}}

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        return await self._evaluate_images_impl(
            input=input, images=(), decisions=decisions, conn_options=conn_options
        )

    async def _evaluate_images_impl(
        self,
        *,
        input: str,
        images: Sequence[llm.ImageContent],
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "state": input,
            "questions": {name: systemone.encode_question(d) for name, d in decisions.items()},
        }
        if images:
            payload["content"] = [self._image_part(image) for image in images[-MAX_IMAGES:]]

        session = self._session or utils.http_context.http_session()
        try:
            async with session.post(
                self._endpoint,
                headers=client_headers(self._api_key),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                if response.status >= 400:
                    raise APIStatusError(
                        "VLM Run decision request failed",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                    )
                parsed = systemone.Response.model_validate(await response.json())
                request_id = parsed.id or response.headers.get("x-request-id", "")
            results, errors = systemone.decode_answers(
                parsed.answers, invalid="invalid VLM Run decision answer"
            )
            return DecisionResponse(
                results=results,
                errors=errors,
                model=parsed.model,
                provider=self.provider,
                request_id=request_id,
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
            )
        except asyncio.TimeoutError as exc:
            raise APITimeoutError("VLM Run decision request timed out") from exc
        except aiohttp.ContentTypeError as exc:
            raise APIError("invalid VLM Run decision response", retryable=False) from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("VLM Run connection failed") from exc
        except ValueError as exc:
            raise APIError("invalid VLM Run decision response", retryable=False) from exc
