from __future__ import annotations

import asyncio
import base64
import dataclasses
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIError, APIStatusError
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent, decisions
from livekit.plugins import vlmrun

pytestmark = pytest.mark.unit

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}
NO_RETRY = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()


def response_body():
    return {
        "id": "s1-123",
        "model": "google/diffusiongemma-26b-a4b-it",
        "answers": {
            "human": {"type": "noul", "noul": 0.9},
            "intent": {
                "type": "choice",
                "choice": "booking",
                "probabilities": {"booking": 0.8, "other": 0.2},
                "confidence": 0.5,
            },
            "frustration": {
                "type": "score",
                "score": 1.4,
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                "legend": ["Calm", "Annoyed", "Angry"],
                "confidence": 0.1,
            },
        },
        "usage": {"input_tokens": 417, "output_tokens": 0, "reads": 1, "cost": 0.001},
    }


def model_for(body, status=200, **kwargs):
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.status = status
    response.headers = {"x-request-id": "request-1"}
    response.json = AsyncMock(return_value=body)
    http_session = MagicMock(spec=aiohttp.ClientSession)
    http_session.post.return_value = response
    return vlmrun.DecisionModel(http_session=http_session, **kwargs), http_session


def sent(http) -> dict:
    return http.post.call_args.kwargs["json"]


async def test_posts_systemone_request_and_decodes_all_kinds(monkeypatch) -> None:
    monkeypatch.delenv("VLMRUN_GATEWAY_BASE_URL", raising=False)
    model, http = model_for(response_body(), api_key="key-1")
    response = await model.evaluate(input="user: Book a table.", decisions=QUESTIONS)
    assert http.post.call_args.args[0] == "https://gateway.vlm.run/typesafe/v1/systemone"
    payload = sent(http)
    assert payload["model"] == "google/diffusiongemma-26b-a4b-it"
    assert payload["state"] == "user: Book a table."
    assert payload["questions"]["human"] == {
        "instructions": "The caller wants a human.",
        "type": "noul",
    }
    assert payload["questions"]["intent"]["criteria"] == QUESTIONS["intent"].options
    assert payload["questions"]["frustration"]["criteria"] == QUESTIONS["frustration"].levels
    assert "content" not in payload
    assert response.results["human"].value == 0.9
    assert response.results["intent"].value == "booking"
    assert response.results["frustration"].levels == ["Calm", "Annoyed", "Angry"]
    assert (response.provider, response.request_id, response.input_tokens) == (
        "vlmrun",
        "s1-123",
        417,
    )
    assert response.errors == {}


async def test_api_key_and_client_header(monkeypatch) -> None:
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    model, http = model_for(response_body())
    await model.evaluate(input="hi", decisions=QUESTIONS)
    headers = http.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer env-key"
    assert headers["X-VLMRun-Client"].startswith("livekit-agents/")


async def test_anonymous_requests_omit_authorization(monkeypatch) -> None:
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    model, http = model_for(response_body())
    await model.evaluate(input="hi", decisions=QUESTIONS)
    assert "Authorization" not in http.post.call_args.kwargs["headers"]


async def test_base_url_override(monkeypatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://gw.internal/")
    model, http = model_for(response_body())
    await model.evaluate(input="hi", decisions=QUESTIONS)
    assert http.post.call_args.args[0] == "https://gw.internal/typesafe/v1/systemone"


async def test_conversation_images_are_sent_as_content_parts() -> None:
    context = ChatContext(
        [
            ChatMessage(role="system", content=["Be terse."]),
            ChatMessage(
                role="user",
                content=[
                    "Is the gripper holding the cube?",
                    ImageContent(image="https://example.com/frame.jpg", inference_detail="low"),
                ],
            ),
            ChatMessage(role="assistant", content=["Looking."]),
            ChatMessage(role="user", content=[ImageContent(image=PNG_DATA_URL)]),
        ]
    )
    model, http = model_for(response_body())
    await model.evaluate(input=context, decisions=QUESTIONS)
    payload = sent(http)
    assert payload["state"] == "user: Is the gripper holding the cube?\nassistant: Looking."
    assert payload["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/frame.jpg", "detail": "low"},
        },
        {"type": "image_url", "image_url": {"url": PNG_DATA_URL, "detail": "auto"}},
    ]


async def test_video_frames_are_encoded_as_jpeg() -> None:
    frame = rtc.VideoFrame(4, 4, rtc.VideoBufferType.RGBA, bytes([255, 0, 0, 255] * 16))
    model, http = model_for(response_body(), detail="high")
    await model.evaluate(input="Is it red?", decisions=QUESTIONS, images=[frame])
    (part,) = sent(http)["content"]
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert part["image_url"]["detail"] == "high"


async def test_only_the_newest_eight_images_are_sent() -> None:
    urls = [f"https://example.com/{i}.jpg" for i in range(10)]
    model, http = model_for(response_body())
    await model.evaluate(input="", decisions=QUESTIONS, images=urls)
    assert [p["image_url"]["url"] for p in sent(http)["content"]] == urls[2:]


async def test_concurrent_evaluations_keep_their_own_images() -> None:
    model, http = model_for(response_body())
    await asyncio.gather(
        model.evaluate(input="a", decisions=QUESTIONS, images=["https://example.com/a.jpg"]),
        model.evaluate(input="b", decisions=QUESTIONS),
    )
    by_state = {c.kwargs["json"]["state"]: c.kwargs["json"] for c in http.post.call_args_list}
    assert by_state["a"]["content"][0]["image_url"]["url"] == "https://example.com/a.jpg"
    assert "content" not in by_state["b"]


async def test_client_errors_are_not_retried() -> None:
    model, http = model_for({}, status=422)
    with pytest.raises(APIStatusError) as raised:
        await model.evaluate(input="hi", decisions=QUESTIONS)
    assert raised.value.status_code == 422
    assert raised.value.request_id == "request-1"
    http.post.assert_called_once()


async def test_server_errors_are_retryable() -> None:
    model, http = model_for({}, status=503)
    with pytest.raises(APIStatusError) as raised:
        await model.evaluate(input="hi", decisions=QUESTIONS, conn_options=NO_RETRY)
    assert raised.value.retryable


async def test_refusal_is_a_per_decision_error() -> None:
    body = response_body()
    body["answers"]["human"] = {"type": "refusal"}
    model, _ = model_for(body)
    response = await model.evaluate(input="hi", decisions=QUESTIONS)
    assert "human" in response.errors
    assert response.results["intent"].value == "booking"


async def test_malformed_body_is_not_retried() -> None:
    model, http = model_for({"answers": "nope"})
    with pytest.raises(APIError):
        await model.evaluate(input="hi", decisions=QUESTIONS)
    http.post.assert_called_once()
