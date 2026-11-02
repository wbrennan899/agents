"""Live tests for the VLM Run vision tools against the public gateway.

They need ``VLMRUN_API_KEY``. A 429 (rate limit) or 403 (model needs a paid account)
skips instead of failing.
"""

from __future__ import annotations

import io
import os

import aiohttp
import pytest

from livekit.agents import APIStatusError
from livekit.plugins.vlmrun import tools

pytestmark = [
    pytest.mark.plugin("vlmrun"),
    pytest.mark.skipif(not os.environ.get("VLMRUN_API_KEY"), reason="VLMRUN_API_KEY not set"),
]

PERSON_IMAGE = "https://ultralytics.com/images/bus.jpg"


def _png(draw) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), "white")
    draw(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _shapes() -> bytes:
    def draw(d):
        d.ellipse((60, 60, 260, 260), fill="red")
        d.rectangle((360, 200, 580, 420), fill="blue")

    return _png(draw)


def _text() -> bytes:
    from PIL import ImageFont

    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()

    return _png(lambda d: d.text((40, 180), "HELLO 42", fill="black", font=font))


async def _call(fn, *args, **kwargs):
    async with aiohttp.ClientSession() as session:
        try:
            return await fn(*args, http_session=session, **kwargs)
        except APIStatusError as e:
            if e.status_code == 429:
                pytest.skip("VLM Run anonymous rate limit reached")
            if e.status_code == 403:
                pytest.skip(f"{fn.__name__} needs a paid VLM Run key: {e.message}")
            raise


async def test_detect_open_vocabulary_finds_red_circle():
    result = await _call(tools.detect, _shapes(), query="red circle")

    assert result.items, "expected at least one red circle"
    box = result.items[0].box
    # the circle is drawn at x 60-260, y 60-260 on a 640x480 canvas
    assert box.x < 0.2 and box.y < 0.25 and box.x + box.w < 0.5


async def _person_photo() -> bytes:
    # The host serves this photo as application/octet-stream, which the gateway rejects as a
    # URL, so it is sent as bytes.
    async with aiohttp.ClientSession() as session, session.get(PERSON_IMAGE) as resp:
        resp.raise_for_status()
        return await resp.read()


async def test_detect_fixed_classes_returns_boxes():
    result = await _call(tools.detect, await _person_photo())

    assert result.items, "bus.jpg shows a bus and people"
    assert any(i.label for i in result.items)
    assert all(0 <= i.box.x <= 1 and 0 <= i.box.y <= 1 for i in result.items)


async def test_segment_returns_png_label_map():
    result = await _call(tools.segment, _shapes(), "blue square")

    assert result.items, "expected the blue square"
    assert result.mask_png is not None and result.mask_png.startswith(b"\x89PNG")


async def test_pose_finds_people():
    result = await _call(tools.pose, await _person_photo())

    assert result.items, "bus.jpg shows several people"
    person = result.items[0]
    assert len(person.keypoints) == 17
    assert person.keypoints[0].name


async def test_ocr_reads_text():
    result = await _call(tools.ocr, _text())

    assert "HELLO" in result.text.upper()
