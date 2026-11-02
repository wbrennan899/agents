from __future__ import annotations

import base64
import io

import pytest

from livekit.agents import llm
from livekit.plugins import vlmrun

pytestmark = pytest.mark.plugin("vlmrun")


def _stop_sign_jpeg() -> str:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 40, 260, 200), fill="red")
    draw.text((140, 110), "STOP", fill="white")
    buf = io.BytesIO()
    image.save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def test_streams_answer_about_image() -> None:
    # qwen/qwen3.5-0.8b is served on the anonymous tier, so this runs without VLMRUN_API_KEY.
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b", max_completion_tokens=60)
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(
        role="user",
        content=[
            "Describe this image in one sentence.",
            llm.ImageContent(image=_stop_sign_jpeg()),
        ],
    )

    text = ""
    try:
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
    finally:
        await model.aclose()

    lowered = text.lower()
    assert "stop" in lowered or "red" in lowered, text
