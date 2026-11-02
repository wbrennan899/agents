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

"""Vision tools backed by VLM Run gateway models.

Two layers:

- ``detect``, ``segment``, ``pose`` and ``ocr`` are plain async functions that run one
  gateway model on one image and return typed results.
- ``VisionTools`` is a ``Toolset`` that exposes the same operations to an agent's LLM as
  function tools, run on the latest frame of the room's video.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    get_job_context,
    utils,
)
from livekit.agents.llm import ToolError, Toolset, function_tool
from livekit.agents.utils.images import EncodeOptions, ResizeOptions, encode

from ._common import client_headers, openai_base_url, resolve_api_key
from .log import logger

ImageInput = str | bytes | rtc.VideoFrame
"""An image URL, a ``data:`` URI, raw JPEG/PNG/WebP bytes, or a video frame."""

FrameSource = Callable[[], ImageInput | None]

DETECT_MODEL = "microsoft/florence-2-base-ft"
SEGMENT_MODEL = "facebook/sam3.1"
POSE_MODEL = "usyd-community/vitpose-plus-large"
HAND_POSE_MODEL = "geopavlakos/hamer"
OCR_MODEL = "paddleocr/pp-ocrv6"

_DEFAULT_MAX_SIDE = 1024


@dataclass(frozen=True)
class BoundingBox:
    """A box in normalized image coordinates (0-1), top-left origin."""

    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x * width),
            round(self.y * height),
            round(self.w * width),
            round(self.h * height),
        )


@dataclass
class Detection:
    box: BoundingBox
    label: str | None = None
    score: float | None = None


@dataclass
class DetectionResult:
    items: list[Detection]
    model: str
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class SegmentInstance:
    box: BoundingBox
    label: str | None
    instance_id: int
    area: float | None = None
    score: float | None = None


@dataclass
class SegmentationResult:
    items: list[SegmentInstance]
    model: str
    mask_png: bytes | None = None
    """8-bit PNG label map: pixel 0 is background, any other value is an ``instance_id``."""
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class Keypoint:
    name: str
    x: float
    """Normalized; may fall outside 0-1 for joints out of frame."""
    y: float
    score: float | None = None
    xyz: tuple[float, float, float] | None = None
    """Metric 3D position, only returned by hand-pose models."""


@dataclass
class PoseInstance:
    box: BoundingBox
    keypoints: list[Keypoint]
    label: str | None = None
    score: float | None = None


@dataclass
class PoseResult:
    items: list[PoseInstance]
    model: str
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class TextLine:
    text: str
    box: BoundingBox | None = None
    score: float | None = None


@dataclass
class OCRResult:
    text: str
    model: str
    lines: list[TextLine] = field(default_factory=list)


async def detect(
    image: ImageInput,
    *,
    query: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_session: aiohttp.ClientSession | None = None,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
) -> DetectionResult:
    """Find objects in an image.

    Args:
        image: The image to analyze.
        query: What to look for, e.g. ``"red cube"``. With a query, detection is
            open-vocabulary (SAM 3.1 boxes); without one, a fixed-class detector
            (Florence-2) returns everything it recognizes.
        model: Override the gateway model.
        api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``; anonymous if unset.
        base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or the public gateway.
        http_session: Session to reuse. Defaults to the agent's shared session.
        conn_options: Timeout options.
    """
    if query:
        seg = await segment(
            image,
            query,
            model=model or SEGMENT_MODEL,
            include_mask=False,
            api_key=api_key,
            base_url=base_url,
            http_session=http_session,
            conn_options=conn_options,
        )
        return DetectionResult(
            items=[Detection(box=i.box, label=i.label, score=i.score) for i in seg.items],
            model=seg.model,
            image_width=seg.image_width,
            image_height=seg.image_height,
        )

    model = model or DETECT_MODEL
    envelope = await _complete(
        model=model,
        method="od",
        image=image,
        api_key=api_key,
        base_url=base_url,
        http_session=http_session,
        conn_options=conn_options,
    )
    payload = _payload(envelope)
    return DetectionResult(
        items=[
            Detection(box=_box(i["bbox_xywh"]), label=i.get("label"), score=i.get("score"))
            for i in payload.get("items", [])
        ],
        model=model,
        image_width=envelope.get("image_width"),
        image_height=envelope.get("image_height"),
    )


async def segment(
    image: ImageInput,
    prompt: str,
    *,
    include_mask: bool = True,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_session: aiohttp.ClientSession | None = None,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
) -> SegmentationResult:
    """Segment every instance of ``prompt`` in an image.

    Args:
        image: The image to analyze.
        prompt: What to segment, e.g. ``"cup"``.
        include_mask: Return the PNG label map. Skipping it makes the response much smaller.
        model: Override the gateway model.
        api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``; anonymous if unset.
        base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or the public gateway.
        http_session: Session to reuse. Defaults to the agent's shared session.
        conn_options: Timeout options.
    """
    model = model or SEGMENT_MODEL
    envelope = await _complete(
        model=model,
        method="segment",
        image=image,
        text=prompt,
        method_params=None if include_mask else {"mask_format": "none"},
        api_key=api_key,
        base_url=base_url,
        http_session=http_session,
        conn_options=conn_options,
    )
    payload = _payload(envelope)
    mask_png: bytes | None = None
    mask = payload.get("mask") or envelope.get("mask")
    if include_mask and isinstance(mask, dict) and isinstance(mask.get("data"), str):
        mask_png = _decode_data_uri(mask["data"])
    return SegmentationResult(
        items=[
            SegmentInstance(
                box=_box(i["bbox_xywh"]),
                label=i.get("label"),
                instance_id=int(i.get("instance_id", 0)),
                area=i.get("area"),
                score=i.get("score"),
            )
            for i in payload.get("items", [])
        ],
        model=model,
        mask_png=mask_png,
        image_width=envelope.get("image_width"),
        image_height=envelope.get("image_height"),
    )


async def pose(
    image: ImageInput,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_session: aiohttp.ClientSession | None = None,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
) -> PoseResult:
    """Estimate body keypoints (17 COCO joints per person).

    Pass ``model=HAND_POSE_MODEL`` for 21 hand joints per hand, with metric 3D positions.

    Args:
        image: The image to analyze.
        model: Override the gateway model.
        api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``; anonymous if unset.
        base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or the public gateway.
        http_session: Session to reuse. Defaults to the agent's shared session.
        conn_options: Timeout options.
    """
    model = model or POSE_MODEL
    envelope = await _complete(
        model=model,
        method="pose",
        image=image,
        api_key=api_key,
        base_url=base_url,
        http_session=http_session,
        conn_options=conn_options,
    )
    payload = _payload(envelope)
    names: list[str] = payload.get("kpts_labels") or []
    items: list[PoseInstance] = []
    for i in payload.get("items", []):
        xy = i.get("kpts_xy") or []
        scores = i.get("kpts_score") or []
        xyz = i.get("kpts_xyz") or []
        keypoints = [
            Keypoint(
                name=names[k] if k < len(names) else str(k),
                x=float(p[0]),
                y=float(p[1]),
                score=float(scores[k]) if k < len(scores) else None,
                xyz=(float(xyz[k][0]), float(xyz[k][1]), float(xyz[k][2]))
                if k < len(xyz)
                else None,
            )
            for k, p in enumerate(xy)
        ]
        items.append(
            PoseInstance(
                box=_box(i["bbox_xywh"]),
                keypoints=keypoints,
                label=i.get("label"),
                score=i.get("score"),
            )
        )
    return PoseResult(
        items=items,
        model=model,
        image_width=envelope.get("image_width"),
        image_height=envelope.get("image_height"),
    )


async def ocr(
    image: ImageInput,
    *,
    model: str | None = None,
    method: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_session: aiohttp.ClientSession | None = None,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
) -> OCRResult:
    """Read the text in an image.

    The default model returns one line per text region with its box. Markdown OCR models
    (e.g. ``zai-org/glm-ocr``) return the page as text with no boxes.

    Args:
        image: The image to read.
        model: Override the gateway model.
        method: Override the model's OCR method. Defaults to ``"ocr"`` for the default model
            and to the model's own default otherwise.
        api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``; anonymous if unset.
        base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or the public gateway.
        http_session: Session to reuse. Defaults to the agent's shared session.
        conn_options: Timeout options.
    """
    if model is None:
        model = OCR_MODEL
        method = method or "ocr"
    envelope = await _complete(
        model=model,
        method=method,
        image=image,
        api_key=api_key,
        base_url=base_url,
        http_session=http_session,
        conn_options=conn_options,
    )
    content = envelope.get("content")
    if isinstance(content, str):
        return OCRResult(text=content, model=model)
    payload = _payload(envelope)
    lines = [
        TextLine(
            text=str(i.get("text", "")),
            box=_box(i["bbox_xywh"]) if "bbox_xywh" in i else None,
            score=i.get("score"),
        )
        for i in payload.get("items", [])
    ]
    return OCRResult(
        text="\n".join(line.text for line in lines if line.text), model=model, lines=lines
    )


class VisionTools(Toolset):
    """VLM Run vision tools for an agent's LLM, run on the latest video frame.

    By default the toolset follows the most recently published remote video track in the
    job's room, the same track an agent would see with live video input. Pass
    ``frame_source`` to supply frames yourself (a robot camera, a screenshot, a test image).

    Usage::

        agent = Agent(instructions="...", tools=[vlmrun.VisionTools()])
    """

    def __init__(
        self,
        *,
        frame_source: FrameSource | None = None,
        max_image_side: int = _DEFAULT_MAX_SIDE,
        detect_model: str | None = None,
        segment_model: str | None = None,
        pose_model: str | None = None,
        ocr_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            frame_source: Returns the image to analyze, or None when there is none yet.
                Defaults to the latest frame of the room's video track.
            max_image_side: Frames are downscaled so the longer side is at most this many
                pixels before upload.
            detect_model: Model for ``detect_objects`` without a query.
            segment_model: Model for open-vocabulary detection and segmentation.
            pose_model: Model for ``detect_poses``.
            ocr_model: Model for ``read_text``.
            api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``; anonymous if unset.
            base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or the public
                gateway.
            http_session: Session to reuse. Defaults to the agent's shared session.
        """
        super().__init__(id="vlmrun_vision")
        self._frame_source = frame_source
        self._max_side = max_image_side
        self._detect_model = detect_model
        self._segment_model = segment_model
        self._pose_model = pose_model
        self._ocr_model = ocr_model
        self._opts: dict[str, Any] = {
            "api_key": resolve_api_key(api_key),
            "base_url": base_url,
            "http_session": http_session,
        }
        self._watcher: _LatestVideoFrame | None = None

    async def setup(self) -> VisionTools:
        await super().setup()
        if self._frame_source is None and self._watcher is None:
            job_ctx = get_job_context(required=False)
            if job_ctx is None:
                logger.warning(
                    "VisionTools has no frame_source and no job context; tools will fail"
                )
            else:
                self._watcher = _LatestVideoFrame(job_ctx.room)
                self._frame_source = self._watcher.get
        return self

    async def aclose(self) -> None:
        if self._watcher is not None:
            await self._watcher.aclose()
            self._watcher = None
        await super().aclose()

    @function_tool
    async def detect_objects(self, query: str | None = None) -> list[dict[str, Any]]:
        """Find objects in what the camera currently sees.

        Boxes are [x, y, width, height], normalized to 0-1 from the top-left corner.

        Args:
            query: What to look for, e.g. "red cube" or "person". Leave empty to list
                every recognizable object.
        """
        model = self._segment_model if query else self._detect_model
        result = await self._run(detect, query=query, model=model)
        return [_detection_dict(d.label, d.box, d.score) for d in result.items]

    @function_tool
    async def segment_objects(self, query: str) -> list[dict[str, Any]]:
        """Outline every instance of something in what the camera currently sees.

        Returns each instance's box (normalized [x, y, width, height]), confidence and the
        fraction of the frame it covers.

        Args:
            query: What to segment, e.g. "cup".
        """
        result = await self._run(segment, query, include_mask=False, model=self._segment_model)
        return [
            {**_detection_dict(i.label, i.box, i.score), "area": _round(i.area)}
            for i in result.items
        ]

    @function_tool
    async def detect_poses(self) -> list[dict[str, Any]]:
        """Estimate the body pose of each person the camera currently sees.

        Returns each person's box and named joints as normalized [x, y, confidence].
        """
        result = await self._run(pose, model=self._pose_model)
        return [
            {
                "box": _box_list(p.box),
                "keypoints": {
                    k.name: [_round(k.x), _round(k.y), _round(k.score)] for k in p.keypoints
                },
            }
            for p in result.items
        ]

    @function_tool
    async def read_text(self) -> str:
        """Read any text the camera currently sees (signs, labels, documents, screens)."""
        result = await self._run(ocr, model=self._ocr_model)
        return result.text or "No text found."

    async def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        image = self._current_image()
        try:
            return await fn(image, *args, **kwargs, **self._opts)
        except APIStatusError as e:
            raise ToolError(f"vision model request failed with status {e.status_code}") from e
        except APIError as e:
            raise ToolError(f"vision model request failed: {e.message}") from e

    def _current_image(self) -> ImageInput:
        frame = self._frame_source() if self._frame_source else None
        if frame is None:
            raise ToolError("no camera frame is available yet")
        if isinstance(frame, rtc.VideoFrame):
            return _encode_frame(frame, self._max_side)
        return frame


class _LatestVideoFrame:
    """Keeps the newest frame of the most recently subscribed remote video track."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._frame: rtc.VideoFrame | None = None
        self._task: asyncio.Task[None] | None = None
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None and publication.kind == rtc.TrackKind.KIND_VIDEO:
                    self._follow(publication.track)
        room.on("track_subscribed", self._on_track_subscribed)

    def get(self) -> rtc.VideoFrame | None:
        return self._frame

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self._follow(track)

    def _follow(self, track: rtc.Track) -> None:
        # the cancelled reader closes its own stream, so the sync event handler never awaits
        if self._task is not None:
            self._task.cancel()
        self._task = asyncio.create_task(self._read(rtc.VideoStream(track)))

    async def _read(self, stream: rtc.VideoStream) -> None:
        try:
            async for event in stream:
                self._frame = event.frame
        finally:
            await stream.aclose()

    async def aclose(self) -> None:
        self._room.off("track_subscribed", self._on_track_subscribed)
        if self._task is not None:
            await utils.aio.cancel_and_wait(self._task)
            self._task = None


async def _complete(
    *,
    model: str,
    method: str | None,
    image: ImageInput,
    api_key: str | None,
    base_url: str | None,
    http_session: aiohttp.ClientSession | None,
    conn_options: APIConnectOptions,
    text: str | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": _image_url(image)}}]
    if text:
        content.append({"type": "text", "text": text})
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }
    if method:
        body["method"] = method
    if method_params:
        body["method_params"] = method_params

    session = http_session or utils.http_context.http_session()
    try:
        async with session.post(
            f"{openai_base_url(base_url)}/chat/completions",
            headers=client_headers(resolve_api_key(api_key)),
            json=body,
            timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
        ) as resp:
            if resp.status >= 400:
                raise APIStatusError(
                    f"VLM Run {model} request failed: {await resp.text()}",
                    status_code=resp.status,
                    request_id=resp.headers.get("x-request-id"),
                )
            data = await resp.json()
    except asyncio.TimeoutError as e:
        raise APITimeoutError(f"VLM Run {model} request timed out") from e
    except aiohttp.ContentTypeError as e:
        raise APIError(f"invalid VLM Run {model} response", retryable=False) from e
    except aiohttp.ClientError as e:
        raise APIConnectionError(f"VLM Run {model} connection failed") from e

    return _parse_envelope(data, model)


def _parse_envelope(data: dict[str, Any], model: str) -> dict[str, Any]:
    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise APIError(f"invalid VLM Run {model} response", retryable=False) from e
    if isinstance(message, dict):
        return message
    if not isinstance(message, str):
        raise APIError(f"invalid VLM Run {model} response", retryable=False)
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        # markdown-kind methods answer with plain text, not a JSON envelope
        return {"content": message}
    if not isinstance(parsed, dict):
        raise APIError(f"invalid VLM Run {model} response", retryable=False)
    return parsed


def _payload(envelope: dict[str, Any]) -> dict[str, Any]:
    content = envelope.get("content")
    if isinstance(content, dict):
        return content
    # some region models return the typed payload at the top level instead of under content
    return envelope


def _image_url(image: ImageInput) -> str:
    if isinstance(image, rtc.VideoFrame):
        return _encode_frame(image, _DEFAULT_MAX_SIDE)
    if isinstance(image, (bytes, bytearray)):
        return f"data:{_sniff_mime(bytes(image))};base64,{base64.b64encode(image).decode()}"
    return image


def _encode_frame(frame: rtc.VideoFrame, max_side: int) -> str:
    resize = None
    if max(frame.width, frame.height) > max_side:
        resize = ResizeOptions(width=max_side, height=max_side, strategy="scale_aspect_fit")
    data = encode(frame, EncodeOptions(format="JPEG", resize_options=resize))
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def _decode_data_uri(uri: str) -> bytes:
    return base64.b64decode(uri.split(",", 1)[1] if uri.startswith("data:") else uri)


def _box(xywh: list[float]) -> BoundingBox:
    x, y, w, h = (float(v) for v in xywh)
    return BoundingBox(x=x, y=y, w=w, h=h)


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def _box_list(box: BoundingBox) -> list[float | None]:
    return [_round(box.x), _round(box.y), _round(box.w), _round(box.h)]


def _detection_dict(label: str | None, box: BoundingBox, score: float | None) -> dict[str, Any]:
    return {"label": label, "box": _box_list(box), "score": _round(score)}
