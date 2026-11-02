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
import contextlib
import itertools
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    llm,
    utils,
)
from livekit.agents.llm import _systemone as systemone
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.images import EncodeOptions, ResizeOptions, encode

from ._common import client_headers, resolve_api_key, stream_ws_url
from .log import logger

DEFAULT_MODEL = "google/diffusiongemma-26b-a4b-it"

EventTypes = Literal["decision", "frame_dropped", "error", "session_stats"]

FrameInput = rtc.VideoFrame | bytes

# The session route holds at most this many reads in flight; asking for more is a
# validation error on session.create.
_MAX_INFLIGHT = 8
# The gateway closes a session after 30 s without a frame. Closing it ourselves a
# little earlier turns that into a clean end instead of an error.
_IDLE_CLOSE_S = 20.0
# Open the next session this long before the current one expires.
_RENEW_MARGIN_S = 15.0
_MIN_SESSION_S = 5.0
_CLOSE_TIMEOUT_S = 5.0
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
# Rejections that a reconnect cannot fix.
_FATAL_ERROR_TYPES = frozenset(
    {"authentication_error", "permission_error", "invalid_request_error", "not_found_error"}
)
# An invalid frame is that frame's problem; only these end the session no matter what.
_SESSION_FATAL_ERROR_TYPES = frozenset({"authentication_error", "permission_error"})


@dataclass
class FrameDecision:
    """Answers to every question for one frame."""

    frame: int
    """Sequence number assigned by :meth:`VisionStream.push_frame`."""
    timestamp: float
    """Wall-clock time (seconds since the epoch) when the frame was captured or pushed."""
    results: dict[str, llm.DecisionResult]
    """Decoded answers keyed by question name."""
    errors: dict[str, str]
    """Questions the model could not answer, keyed by name."""
    latency_ms: float | None
    """Server-side read latency reported by the gateway."""
    usage: dict[str, Any]
    """Token counts and ``cost`` in USD for this read."""
    session_id: str

    @property
    def values(self) -> dict[str, float | str]:
        """The bare value of each answer: a probability, an option name, or a level index."""
        return {name: result.value for name, result in self.results.items()}


@dataclass
class FrameDropped:
    """A frame that was never answered."""

    frame: int
    timestamp: float
    reason: Literal["busy", "server", "lost", "too_large"]
    """``busy``: replaced by a newer frame while every read slot was in use.
    ``server``: the gateway dropped it. ``lost``: the session ended before it was
    answered. ``too_large``: the encoded image exceeded the gateway's 2 MiB limit."""


@dataclass
class VisionStreamError:
    """An error reported by the gateway or the connection."""

    error: Exception
    error_type: str
    frame: int | None
    recoverable: bool
    """False when the stream has stopped and will not reconnect."""


@dataclass
class _Pending:
    frame: int
    timestamp: float
    data: FrameInput


@dataclass
class _SessionEnd:
    reason: Literal["renew", "idle", "closed"]


@dataclass
class _Session:
    session_id: str
    slots: asyncio.Semaphore
    inflight: dict[str, tuple[int, float]] = field(default_factory=dict)
    stats_received: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    resume_at: float = 0.0
    """Loop time before which no read is sent, set by a rate limit's ``retry_after``."""


class VisionStream(rtc.EventEmitter[EventTypes]):
    """Typed per-frame decisions about live video, over the VLM Run gateway's WebSocket.

    Frames come from a room's video track (:meth:`start`) or from :meth:`push_frame`.
    Each sampled frame is read against the same questions a ``DecisionModel`` takes
    (``llm.Probability``, ``llm.Choice``, ``llm.Score``), and every answer is emitted as
    a ``"decision"`` event carrying a :class:`FrameDecision`.

    Latest frame wins: when every read slot is busy, a waiting frame is replaced by the
    newer one and reported as a ``"frame_dropped"`` event, so answers never fall behind
    the video. Sessions are renewed before the gateway's lifetime cap, closed after
    20 s without frames, and reopened when the next frame arrives.

    Events:
        decision: :class:`FrameDecision`
        frame_dropped: :class:`FrameDropped`
        error: :class:`VisionStreamError`
        session_stats: ``dict`` with the gateway's ``session.stats`` frame
            (frames, decisions, cost, ``p50_ms``, ``p95_ms`` and more)
    """

    def __init__(
        self,
        *,
        questions: Mapping[str, llm.Decision],
        model: str = DEFAULT_MODEL,
        state: str | None = None,
        fps: float = 2.0,
        width: int = 512,
        height: int = 512,
        detail: Literal["low", "high", "auto"] = "low",
        max_inflight: int = 2,
        publish_topic: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a VisionStream.

        Args:
            questions: Decisions to evaluate on every frame, keyed by name.
            model: A model served on the gateway's ``/typesafe`` route.
            state: Context the frames are read against, e.g. "A warehouse camera".
            fps: Frames sampled per second from a room track. :meth:`push_frame` is not
                throttled.
            width: Maximum width of the encoded frame, aspect ratio preserved.
            height: Maximum height of the encoded frame, aspect ratio preserved.
            detail: Vision budget per frame. ``low`` is fastest and cheapest.
            max_inflight: Reads in flight at once (1 to 8). The server may grant fewer.
            publish_topic: When set and the stream is attached to a room, each decision is
                also published to the room as JSON data on this topic.
            api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``.
            base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or
                ``https://gateway.vlm.run``.
            http_session: Session to open the WebSocket with. Defaults to the job's.
            conn_options: Connection timeout, retry count and retry interval.
        """
        super().__init__()
        if not questions:
            raise ValueError("VisionStream requires at least one question")
        if fps <= 0:
            raise ValueError("fps must be greater than 0")
        if not 1 <= max_inflight <= _MAX_INFLIGHT:
            raise ValueError(f"max_inflight must be between 1 and {_MAX_INFLIGHT}")

        self._questions = {name: systemone.encode_question(d) for name, d in questions.items()}
        self._model = model
        self._state = state
        self._state_dirty = False
        self._fps = fps
        self._resize = ResizeOptions(width=width, height=height, strategy="scale_aspect_fit")
        self._detail = detail
        self._max_inflight = max_inflight
        self._publish_topic = publish_topic
        self._api_key = resolve_api_key(api_key)
        self._url = stream_ws_url(base_url)
        self._conn_options = conn_options

        self._http_session = http_session
        self._owns_http_session = False

        self._seq = itertools.count()
        self._pending: _Pending | None = None
        self._frame_ready = asyncio.Event()
        self._closed = False
        # Set when the stream stops on its own after a fatal error; aclose() still runs.
        self._stopped = False
        self._main_task: asyncio.Task[None] | None = None
        self._stats: dict[str, Any] = {}

        self._room: rtc.Room | None = None
        self._participant_identity: str | None = None
        self._track_sid: str | None = None
        self._video_stream: rtc.VideoStream | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def model(self) -> str:
        return self._model

    @property
    def stats(self) -> dict[str, Any]:
        """The last ``session.stats`` frame the gateway sent, or an empty dict."""
        return dict(self._stats)

    async def start(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant | str | None = None,
    ) -> None:
        """Read video from a room.

        Args:
            room: A connected room.
            participant: The participant, or its identity, whose camera or screen share
                to read. Defaults to the first remote participant that publishes video.
        """
        if self._closed:
            raise RuntimeError("VisionStream is closed")
        if self._room is not None:
            raise RuntimeError("VisionStream is already attached to a room")

        self._room = room
        if isinstance(participant, rtc.RemoteParticipant):
            self._participant_identity = participant.identity
        else:
            self._participant_identity = participant

        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)

        for remote in room.remote_participants.values():
            for publication in remote.track_publications.values():
                if publication.track is not None:
                    self._on_track_subscribed(publication.track, publication, remote)

    def push_frame(self, frame: FrameInput, *, timestamp: float | None = None) -> int:
        """Queue one frame for a read.

        Use this to feed frames that don't come from a room track, such as a robot's
        camera. Every pushed frame is a candidate; throttle in the caller.

        Args:
            frame: A video frame, or already-encoded JPEG, PNG, WebP or GIF bytes.
            timestamp: Capture time in seconds since the epoch. Defaults to now.

        Returns:
            The frame's sequence number, matching :attr:`FrameDecision.frame`.
        """
        if self._closed:
            raise RuntimeError("VisionStream is closed")
        if self._stopped:
            raise RuntimeError("VisionStream stopped after a fatal error")

        index = next(self._seq)
        previous = self._pending
        self._pending = _Pending(
            frame=index, timestamp=time.time() if timestamp is None else timestamp, data=frame
        )
        if previous is not None:
            self.emit("frame_dropped", FrameDropped(previous.frame, previous.timestamp, "busy"))
        self._frame_ready.set()

        if self._main_task is None:
            self._main_task = asyncio.create_task(self._run(), name="vlmrun.VisionStream")
        return index

    def update_state(self, state: str | None) -> None:
        """Change the context frames are read against. Applies from the next read."""
        self._state = state
        self._state_dirty = True

    async def aclose(self) -> None:
        """Stop reading, close the session after in-flight reads finish, and release the
        room."""
        if self._closed:
            return
        self._closed = True
        self._frame_ready.set()

        if self._room is not None:
            self._room.off("track_subscribed", self._on_track_subscribed)
            self._room.off("track_unsubscribed", self._on_track_unsubscribed)
        await self._stop_reader()

        if self._main_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._main_task), self._conn_options.timeout + _CLOSE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                await utils.aio.cancel_and_wait(self._main_task)

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._owns_http_session and self._http_session is not None:
            await self._http_session.close()

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        if self._participant_identity is None:
            if self._track_sid is not None:
                return
        elif participant.identity != self._participant_identity:
            return
        if track.sid == self._track_sid:
            return

        old_task, old_stream = self._reader_task, self._video_stream
        self._track_sid = track.sid
        self._video_stream = rtc.VideoStream(track)
        self._reader_task = asyncio.create_task(
            self._read_track(self._video_stream, old_task, old_stream),
            name="vlmrun.VisionStream.read_track",
        )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.sid == self._track_sid:
            self._track_sid = None
            self._track_task(asyncio.create_task(self._stop_reader()))

    async def _stop_reader(self) -> None:
        task, stream = self._reader_task, self._video_stream
        self._reader_task, self._video_stream = None, None
        if task is not None:
            await utils.aio.cancel_and_wait(task)
        if stream is not None:
            await stream.aclose()

    @utils.log_exceptions(logger=logger)
    async def _read_track(
        self,
        stream: rtc.VideoStream,
        previous_task: asyncio.Task[None] | None,
        previous_stream: rtc.VideoStream | None,
    ) -> None:
        if previous_task is not None:
            await utils.aio.cancel_and_wait(previous_task)
        if previous_stream is not None:
            await previous_stream.aclose()

        loop = asyncio.get_running_loop()
        interval = 1.0 / self._fps
        next_due = 0.0
        async for event in stream:
            now = loop.time()
            if self._closed or self._stopped:
                return
            if now < next_due:
                continue
            next_due = now + interval
            self.push_frame(event.frame)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._owns_http_session = True
        return self._http_session

    def _encode(self, data: FrameInput) -> str:
        if isinstance(data, rtc.VideoFrame):
            raw = encode(data, EncodeOptions(format="JPEG", resize_options=self._resize))
            mime = "image/jpeg"
        else:
            raw = data
            mime = _sniff_mime(raw)
        if len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError("encoded frame exceeds the gateway's 2 MiB limit")
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    async def _run(self) -> None:
        failures = 0
        while not self._closed:
            if self._pending is None:
                self._frame_ready.clear()
                await self._frame_ready.wait()
                continue

            try:
                ws, created = await self._connect()
            except APIError as e:
                failures += 1
                fatal = not e.retryable or failures > self._conn_options.max_retry
                self._emit_error(e, frame=None, recoverable=not fatal)
                if fatal:
                    break
                await asyncio.sleep(self._conn_options._interval_for_retry(failures - 1))
                continue

            failures = 0
            try:
                end = await self._run_session(ws, created)
                logger.debug("vlmrun vision session ended", extra={"reason": end.reason})
            except APIError as e:
                failures += 1
                fatal = not e.retryable or failures > self._conn_options.max_retry
                self._emit_error(e, frame=None, recoverable=not fatal)
                if fatal:
                    break
                await asyncio.sleep(self._conn_options._interval_for_retry(failures - 1))
            finally:
                await ws.close()

        if not self._closed:
            self._stopped = True
        if self._pending is not None:
            dropped, self._pending = self._pending, None
            self.emit("frame_dropped", FrameDropped(dropped.frame, dropped.timestamp, "lost"))

    async def _connect(self) -> tuple[aiohttp.ClientWebSocketResponse, dict[str, Any]]:
        session = self._ensure_http_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._url, headers=client_headers(self._api_key), autoping=False
                ),
                self._conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "VLM Run gateway refused the WebSocket"
                + (" (set VLMRUN_API_KEY)" if e.status == 401 and not self._api_key else ""),
                status_code=e.status,
                retryable=e.status not in (400, 401, 403, 404),
            ) from None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to the VLM Run gateway") from e

        create: dict[str, Any] = {
            "type": "session.create",
            "model": self._model,
            "questions": self._questions,
            "max_inflight": self._max_inflight,
        }
        if self._state:
            create["state"] = self._state
        self._state_dirty = False

        try:
            await ws.send_str(json.dumps(create))
            msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            await ws.close()
            raise APIConnectionError("VLM Run gateway did not acknowledge the session") from e

        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise _closed_error(ws)
        created = json.loads(msg.data)
        if created.get("type") != "session.created":
            await ws.close()
            raise _event_error(created)
        return ws, created

    async def _run_session(
        self, ws: aiohttp.ClientWebSocketResponse, created: dict[str, Any]
    ) -> _SessionEnd:
        granted = int(created.get("max_inflight") or self._max_inflight)
        session = _Session(
            session_id=str(created.get("session_id", "")),
            slots=asyncio.Semaphore(max(1, min(self._max_inflight, granted))),
        )
        loop = asyncio.get_running_loop()
        expires_in = float(created.get("expires_in") or 300.0)
        renew_at = loop.time() + max(_MIN_SESSION_S, expires_in - _RENEW_MARGIN_S)

        send_task = asyncio.create_task(self._send_loop(ws, session, renew_at))
        recv_task = asyncio.create_task(self._recv_loop(ws, session))
        try:
            done, _ = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                # The server ended the session; reconnect unless we were closing anyway.
                recv_task.result()
                await utils.aio.cancel_and_wait(send_task)
                raise APIConnectionError("VLM Run gateway closed the session")

            end = send_task.result()
            await self._close_session(ws, session)
            return end
        finally:
            await utils.aio.cancel_and_wait(send_task, recv_task)
            for frame, ts in session.inflight.values():
                self.emit("frame_dropped", FrameDropped(frame, ts, "lost"))
            session.inflight.clear()

    async def _send_loop(
        self, ws: aiohttp.ClientWebSocketResponse, session: _Session, renew_at: float
    ) -> _SessionEnd:
        loop = asyncio.get_running_loop()
        last_frame_at = loop.time()
        while True:
            if self._closed:
                return _SessionEnd("closed")
            now = loop.time()
            if now >= renew_at:
                return _SessionEnd("renew")
            if now >= last_frame_at + _IDLE_CLOSE_S and self._pending is None:
                return _SessionEnd("idle")

            if now < session.resume_at:
                # Rate limited: keep only the newest frame until the gateway's retry_after.
                await asyncio.sleep(min(session.resume_at - now, 0.25))
                continue

            if self._pending is None:
                self._frame_ready.clear()
                deadline = min(renew_at, last_frame_at + _IDLE_CLOSE_S)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._frame_ready.wait(), max(0.0, deadline - now))
                continue

            # While waiting for a slot, newer frames replace the pending one.
            await session.slots.acquire()
            pending, self._pending = self._pending, None
            if pending is None or self._closed:
                session.slots.release()
                if pending is not None:
                    self._pending = pending
                continue

            try:
                url = await asyncio.to_thread(self._encode, pending.data)
            except ValueError:
                session.slots.release()
                self.emit(
                    "frame_dropped", FrameDropped(pending.frame, pending.timestamp, "too_large")
                )
                continue

            if self._state_dirty:
                self._state_dirty = False
                await ws.send_str(json.dumps({"type": "state", "state": self._state or ""}))

            request_id = str(pending.frame)
            session.inflight[request_id] = (pending.frame, pending.timestamp)
            await ws.send_str(
                json.dumps(
                    {
                        "type": "decide",
                        "id": request_id,
                        "content": [
                            {"type": "image_url", "image_url": {"url": url, "detail": self._detail}}
                        ],
                    }
                )
            )
            last_frame_at = loop.time()

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse, session: _Session) -> None:
        monitor = utils.aio.WebSocketMonitor(
            ws, liveness_timeout=self._conn_options.liveness_timeout
        )
        while True:
            try:
                msg = await monitor.receive()
            except aiohttp.ServerTimeoutError as e:
                raise APIConnectionError("VLM Run gateway stopped responding") from e

            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                if session.closing:
                    return
                raise _closed_error(ws)
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("VLM Run gateway connection lost") from ws.exception()
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("invalid JSON from the VLM Run gateway")
                continue
            self._handle_event(event, session)

    def _handle_event(self, event: dict[str, Any], session: _Session) -> None:
        kind = event.get("type")
        if kind == "decision":
            entry = self._settle(event.get("id"), session)
            if entry is None:
                return
            frame, ts = entry
            results, errors = systemone.decode_answers(
                {name: _coerce_answer(raw) for name, raw in (event.get("answers") or {}).items()}
            )
            decision = FrameDecision(
                frame=frame,
                timestamp=ts,
                results=results,
                errors=errors,
                latency_ms=event.get("latency_ms"),
                usage=event.get("usage") or {},
                session_id=session.session_id,
            )
            self.emit("decision", decision)
            self._publish(decision)
        elif kind == "frame.dropped":
            entry = self._settle(event.get("id"), session) or self._settle_oldest(session)
            frame, ts = entry if entry is not None else (int(event.get("frame", -1)), time.time())
            self.emit("frame_dropped", FrameDropped(frame, ts, "server"))
        elif kind == "error":
            error = _event_error(event)
            error_type = str(event.get("error_type") or "")
            if error_type in _SESSION_FATAL_ERROR_TYPES:
                raise error
            # The gateway doesn't echo the decide id on errors. Reads are answered in
            # order, so the error belongs to the oldest read in flight.
            entry = self._settle(event.get("id"), session) or self._settle_oldest(session)
            if entry is None and not error.retryable:
                raise error
            if retry_after := event.get("retry_after"):
                loop = asyncio.get_running_loop()
                session.resume_at = max(session.resume_at, loop.time() + float(retry_after))
            self._emit_error(error, frame=entry[0] if entry else None, recoverable=True)
        elif kind == "session.stats":
            self._stats = event
            session.stats_received.set()
            self.emit("session_stats", dict(event))

    def _settle_oldest(self, session: _Session) -> tuple[int, float] | None:
        if not session.inflight:
            return None
        return self._settle(next(iter(session.inflight)), session)

    def _settle(self, request_id: Any, session: _Session) -> tuple[int, float] | None:
        if request_id is None:
            return None
        entry = session.inflight.pop(str(request_id), None)
        if entry is not None:
            session.slots.release()
        return entry

    async def _close_session(self, ws: aiohttp.ClientWebSocketResponse, session: _Session) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CLOSE_TIMEOUT_S
        while session.inflight and loop.time() < deadline:
            await asyncio.sleep(0.02)
        session.closing = True
        with contextlib.suppress(aiohttp.ClientError, ConnectionError):
            await ws.send_str(json.dumps({"type": "session.close"}))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.stats_received.wait(), max(0.0, deadline - loop.time()) or 0.5
            )

    def _publish(self, decision: FrameDecision) -> None:
        if self._publish_topic is None or self._room is None or not self._room.isconnected():
            return
        payload = json.dumps(
            {
                "frame": decision.frame,
                "timestamp": decision.timestamp,
                "values": decision.values,
                "errors": decision.errors,
                "latency_ms": decision.latency_ms,
            }
        )
        self._track_task(
            asyncio.create_task(
                self._room.local_participant.publish_data(payload, topic=self._publish_topic)
            )
        )

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _emit_error(self, error: APIError, *, frame: int | None, recoverable: bool) -> None:
        error_type = (
            error.body.get("error_type", "api_error")
            if isinstance(error.body, dict)
            else type(error).__name__
        )
        logger.warning(
            "vlmrun vision stream error",
            extra={"error_type": error_type, "recoverable": recoverable},
        )
        self.emit(
            "error",
            VisionStreamError(
                error=error, error_type=str(error_type), frame=frame, recoverable=recoverable
            ),
        )


def _event_error(event: dict[str, Any]) -> APIStatusError:
    error_type = str(event.get("error_type") or "api_error")
    return APIStatusError(
        f"VLM Run gateway error ({error_type}): {event.get('message', '')}",
        status_code=-1,
        body=event,
        retryable=error_type not in _FATAL_ERROR_TYPES,
    )


def _closed_error(ws: aiohttp.ClientWebSocketResponse) -> APIStatusError:
    code = ws.close_code or -1
    # 1013 means the gateway is at capacity (too many sessions); worth retrying.
    return APIStatusError(
        f"VLM Run gateway closed the WebSocket (code {code})",
        status_code=code,
        retryable=code not in (1008,),
    )


def _coerce_answer(answer: Any) -> Any:
    # JSON object keys are strings, but score legends and distributions are keyed by
    # level index.
    if not isinstance(answer, dict) or answer.get("type") != "score":
        return answer
    coerced = dict(answer)
    for key in ("legend", "probabilities"):
        value = coerced.get(key)
        if isinstance(value, dict):
            coerced[key] = {
                (int(k) if str(k).lstrip("-").isdigit() else k): v for k, v in value.items()
            }
    return coerced


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "image/jpeg"
