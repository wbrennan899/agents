from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import APIConnectOptions
from livekit.plugins.vlmrun import vision_stream as vs
from livekit.plugins.vlmrun.vision_stream import (
    FrameDecision,
    FrameDropped,
    VisionStream,
    VisionStreamError,
)

pytestmark = pytest.mark.unit

QUESTIONS = {
    "person": llm.Probability("A person is in frame."),
    "scene": llm.Choice("What is shown?", options={"door": "A door", "other": "Anything else"}),
    "crowd": llm.Score("How crowded is it?", levels=["Empty", "Some", "Packed"]),
}

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

Reply = Callable[[web.WebSocketResponse, dict[str, Any]], Awaitable[None]]


def answers(person: float) -> dict[str, Any]:
    return {
        "person": {"type": "noul", "noul": person},
        "scene": {
            "type": "choice",
            "choice": "door",
            "probabilities": {"door": 0.7, "other": 0.3},
            "confidence": 0.4,
        },
        "crowd": {
            "type": "score",
            "score": 1.2,
            "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
            "legend": ["Empty", "Some", "Packed"],
        },
    }


async def answer(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
    await ws.send_json(
        {
            "type": "decision",
            "id": event["id"],
            "frame": int(event["id"]),
            "answers": answers(0.9),
            "usage": {"input_tokens": 147, "output_tokens": 0, "cost": 0.001},
            "latency_ms": 42.0,
        }
    )


class FakeGateway:
    def __init__(self, reply: Reply = answer, *, expires_in: float = 600.0) -> None:
        self.reply = reply
        self.expires_in = expires_in
        self.headers: list[dict[str, str]] = []
        self.creates: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.sessions = 0
        self.server = TestServer(self._app())

    def _app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/typesafe/ws", self._handle)
        return app

    @property
    def base_url(self) -> str:
        return str(self.server.make_url("")).rstrip("/")

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sessions += 1
        session_id = f"s{self.sessions}"
        self.headers.append(dict(request.headers))
        self.creates.append(await ws.receive_json())
        await ws.send_json(
            {
                "type": "session.created",
                "session_id": session_id,
                "max_inflight": self.creates[-1]["max_inflight"],
                "expires_in": self.expires_in,
                "max_frame_bytes": 2 * 1024 * 1024,
            }
        )
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                break
            event = json.loads(msg.data)
            event["_session"] = session_id
            self.messages.append(event)
            if event["type"] == "decide":
                await self.reply(ws, event)
            elif event["type"] == "session.close":
                await ws.send_json(
                    {"type": "session.stats", "session_id": session_id, "p50_ms": 40.0}
                )
                await ws.close()
        return ws

    def decides(self) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == "decide"]


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_GATEWAY_BASE_URL", raising=False)


@pytest.fixture
async def gateway():
    gateway = FakeGateway()
    await gateway.server.start_server()
    yield gateway
    await gateway.server.close()


@pytest.fixture
async def http():
    async with aiohttp.ClientSession() as session:
        yield session


def make_stream(gateway: FakeGateway, http: aiohttp.ClientSession, **kwargs: Any) -> VisionStream:
    kwargs.setdefault("questions", QUESTIONS)
    return VisionStream(
        base_url=gateway.base_url,
        http_session=http,
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.05, timeout=2.0),
        **kwargs,
    )


def collect(stream: VisionStream) -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = {
        "decision": [],
        "frame_dropped": [],
        "error": [],
        "session_stats": [],
    }
    for name, bucket in events.items():
        stream.on(name, bucket.append)
    return events


async def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def test_session_create_carries_questions_and_options(gateway, http):
    stream = make_stream(gateway, http, model="m", state="A lobby camera", max_inflight=4)
    events = collect(stream)
    stream.push_frame(JPEG)
    await wait_for(lambda: len(events["decision"]) == 1)
    await stream.aclose()

    assert gateway.creates[0] == {
        "type": "session.create",
        "model": "m",
        "questions": {
            "person": {"type": "noul", "instructions": "A person is in frame."},
            "scene": {
                "type": "choice",
                "instructions": "What is shown?",
                "criteria": {"door": "A door", "other": "Anything else"},
            },
            "crowd": {
                "type": "score",
                "instructions": "How crowded is it?",
                "criteria": ["Empty", "Some", "Packed"],
            },
        },
        "max_inflight": 4,
        "state": "A lobby camera",
    }
    assert "Authorization" not in gateway.headers[0], "anonymous mode sends no bearer"
    assert gateway.headers[0]["X-VLMRun-Client"].startswith("livekit-agents/")


async def test_api_key_is_sent_as_bearer(gateway, http):
    stream = make_stream(gateway, http, api_key="vlm-key")
    events = collect(stream)
    stream.push_frame(JPEG)
    await wait_for(lambda: len(events["decision"]) == 1)
    await stream.aclose()
    assert gateway.headers[0]["Authorization"] == "Bearer vlm-key"


async def test_frame_is_sent_as_image_part_and_decision_is_decoded(gateway, http):
    stream = make_stream(gateway, http, detail="high")
    events = collect(stream)
    index = stream.push_frame(PNG, timestamp=123.5)
    await wait_for(lambda: len(events["decision"]) == 1)
    await stream.aclose()

    part = gateway.decides()[0]["content"][0]
    assert part["type"] == "image_url"
    assert part["image_url"]["detail"] == "high"
    assert part["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()

    decision: FrameDecision = events["decision"][0]
    assert decision.frame == index
    assert decision.timestamp == 123.5
    assert decision.session_id == "s1"
    assert decision.latency_ms == 42.0
    assert decision.usage["cost"] == 0.001
    assert decision.errors == {}
    assert decision.values == {"person": 0.9, "scene": "door", "crowd": 1.2}
    crowd = decision.results["crowd"]
    assert isinstance(crowd, llm.ScoreResult)
    assert crowd.levels == ["Empty", "Some", "Packed"]
    assert crowd.probabilities == {0: 0.2, 1: 0.4, 2: 0.4}
    assert events["session_stats"][0]["p50_ms"] == 40.0, "stats collected on close"


async def test_video_frame_is_encoded_as_resized_jpeg(gateway, http):
    stream = make_stream(gateway, http, width=64, height=64)
    events = collect(stream)
    frame = rtc.VideoFrame(320, 240, rtc.VideoBufferType.RGBA, bytes(320 * 240 * 4))
    stream.push_frame(frame)
    await wait_for(lambda: len(events["decision"]) == 1)
    await stream.aclose()

    url = gateway.decides()[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == (64, 48)


async def test_latest_frame_wins_when_slots_are_busy(http):
    release = asyncio.Event()
    tasks: set[asyncio.Future[None]] = set()

    async def slow(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        async def later() -> None:
            await release.wait()
            await answer(ws, event)

        tasks.add(asyncio.ensure_future(later()))

    gateway = FakeGateway(slow)
    await gateway.server.start_server()
    try:
        stream = make_stream(gateway, http, max_inflight=1)
        events = collect(stream)
        first = stream.push_frame(JPEG)
        await wait_for(lambda: len(gateway.decides()) == 1)
        second = stream.push_frame(JPEG)
        third = stream.push_frame(JPEG)
        release.set()
        await wait_for(lambda: len(events["decision"]) == 2)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert [d.frame for d in events["decision"]] == [first, third]
    dropped: list[FrameDropped] = events["frame_dropped"]
    assert [(d.frame, d.reason) for d in dropped] == [(second, "busy")]


async def test_server_frame_dropped_is_reported(http):
    async def drop(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        # The live gateway's frame.dropped carries its own frame count, not the decide id.
        await ws.send_json({"type": "frame.dropped", "frame": 0, "inflight": 1, "reason": "busy"})

    gateway = FakeGateway(drop)
    await gateway.server.start_server()
    try:
        stream = make_stream(gateway, http, max_inflight=1)
        events = collect(stream)
        index = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["frame_dropped"]) == 1)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(gateway.decides()) == 2)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert events["frame_dropped"][0].frame == index
    assert events["frame_dropped"][0].reason == "server"
    assert gateway.sessions == 1, "a dropped frame frees its slot without a reconnect"


def error_reply(error_type: str, retry_after: float | None = None) -> Reply:
    """Fails the first read, then answers. Errors match the live gateway: no decide id."""
    calls = 0

    async def reply(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await ws.send_json(
                {
                    "type": "error",
                    "error_type": error_type,
                    "message": "failed",
                    "frame": 0,
                    "retry_after": retry_after,
                }
            )
        else:
            await answer(ws, event)

    return reply


async def test_rate_limit_error_frees_the_slot_and_waits_retry_after(http):
    gateway = FakeGateway(error_reply("rate_limit_error", retry_after=0.3))
    await gateway.server.start_server()
    loop = asyncio.get_running_loop()
    try:
        stream = make_stream(gateway, http, max_inflight=1)
        events = collect(stream)
        first = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["error"]) == 1)
        errored_at = loop.time()
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["decision"]) == 1)
        answered_after = loop.time() - errored_at
        await stream.aclose()
    finally:
        await gateway.server.close()

    error: VisionStreamError = events["error"][0]
    assert error.error_type == "rate_limit_error"
    assert error.frame == first, "an id-less error is matched to the oldest read"
    assert error.recoverable
    assert answered_after >= 0.25, f"next read waited for retry_after: {answered_after:.2f}s"
    assert gateway.sessions == 1


async def test_invalid_frame_is_reported_and_streaming_continues(http):
    gateway = FakeGateway(error_reply("invalid_request_error"))
    await gateway.server.start_server()
    try:
        stream = make_stream(gateway, http, max_inflight=1)
        events = collect(stream)
        bad = stream.push_frame(b"not an image")
        await wait_for(lambda: len(events["error"]) == 1)
        good = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["decision"]) == 1)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert events["error"][0].frame == bad
    assert events["error"][0].recoverable
    assert events["decision"][0].frame == good
    assert gateway.sessions == 1, "one bad frame doesn't end the session"


async def test_aclose_after_fatal_error_releases_owned_session():
    gateway = FakeGateway(error_reply("authentication_error"))
    await gateway.server.start_server()
    try:
        stream = VisionStream(
            questions=QUESTIONS,
            base_url=gateway.base_url,
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.05, timeout=2.0),
        )
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: any(not e.recoverable for e in events["error"]))
        owned = stream._http_session
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert owned is not None and owned.closed, "aclose() still cleans up after the stream stopped"


async def test_fatal_error_stops_the_stream(http):
    async def deny(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        await ws.send_json(
            {"type": "error", "error_type": "authentication_error", "message": "bad key"}
        )

    gateway = FakeGateway(deny)
    await gateway.server.start_server()
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: any(not e.recoverable for e in events["error"]))
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.sessions == 1, "no reconnect after an auth error"
    with pytest.raises(RuntimeError):
        stream.push_frame(JPEG)


async def test_unexpected_close_reconnects_and_resends_questions(http):
    calls = 0

    async def close_once(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await ws.close(code=1013)
        else:
            await answer(ws, event)

    gateway = FakeGateway(close_once)
    await gateway.server.start_server()
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        lost = stream.push_frame(JPEG)
        await wait_for(lambda: gateway.sessions == 1 and calls == 1)
        await wait_for(lambda: any(d.reason == "lost" for d in events["frame_dropped"]))
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["decision"]) == 1)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.sessions == 2
    assert gateway.creates[0]["questions"] == gateway.creates[1]["questions"]
    assert events["frame_dropped"][0].frame == lost
    assert events["error"][0].recoverable


async def test_session_renews_before_expiry(http, monkeypatch):
    monkeypatch.setattr(vs, "_MIN_SESSION_S", 0.2)
    monkeypatch.setattr(vs, "_RENEW_MARGIN_S", 0.0)
    gateway = FakeGateway(expires_in=0.2)
    await gateway.server.start_server()
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["decision"]) == 1)
        await asyncio.sleep(0.4)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["decision"]) == 2)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.sessions == 2
    closes = [m for m in gateway.messages if m["type"] == "session.close"]
    assert closes and closes[0]["_session"] == "s1", "the old session was closed politely"
    assert [d.session_id for d in events["decision"]] == ["s1", "s2"]
    assert events["error"] == [], "renewal is not an error"


async def test_idle_session_closes_and_reopens_on_next_frame(gateway, http, monkeypatch):
    monkeypatch.setattr(vs, "_IDLE_CLOSE_S", 0.2)
    stream = make_stream(gateway, http)
    events = collect(stream)
    stream.push_frame(JPEG)
    await wait_for(lambda: len(events["session_stats"]) == 1)
    assert gateway.sessions == 1
    stream.push_frame(JPEG)
    await wait_for(lambda: len(events["decision"]) == 2)
    await stream.aclose()
    assert gateway.sessions == 2


async def test_update_state_is_sent_before_next_read(gateway, http):
    stream = make_stream(gateway, http)
    events = collect(stream)
    stream.push_frame(JPEG)
    await wait_for(lambda: len(events["decision"]) == 1)
    stream.update_state("Night shift")
    stream.push_frame(JPEG)
    await wait_for(lambda: len(events["decision"]) == 2)
    await stream.aclose()
    kinds = [m["type"] for m in gateway.messages]
    assert kinds[:3] == ["decide", "state", "decide"]
    assert gateway.messages[1]["state"] == "Night shift"


async def test_handshake_rejection_without_key_is_fatal(http):
    app = web.Application()

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401)

    app.router.add_get("/typesafe/ws", reject)
    server = TestServer(app)
    await server.start_server()
    try:
        stream = VisionStream(
            questions=QUESTIONS, base_url=str(server.make_url("")).rstrip("/"), http_session=http
        )
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["error"]) == 1)
        await stream.aclose()
    finally:
        await server.close()

    error = events["error"][0]
    assert not error.recoverable
    assert "VLMRUN_API_KEY" in str(error.error)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        VisionStream(questions={})
    with pytest.raises(ValueError):
        VisionStream(questions=QUESTIONS, max_inflight=9)
    with pytest.raises(ValueError):
        VisionStream(questions=QUESTIONS, fps=0)
