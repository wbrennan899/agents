from __future__ import annotations

import asyncio
import io
import os

import aiohttp
import pytest

from livekit.agents import llm
from livekit.plugins.vlmrun.vision_stream import FrameDecision, VisionStream

pytestmark = [
    pytest.mark.plugin("vlmrun"),
    # The gateway's WebSocket route refuses anonymous sessions (401), unlike its HTTP routes.
    pytest.mark.skipif(not os.environ.get("VLMRUN_API_KEY"), reason="VLMRUN_API_KEY not set"),
]


def jpeg(red_circle: bool) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "white")
    if red_circle:
        ImageDraw.Draw(image).ellipse((60, 40, 260, 200), fill="red")
    buf = io.BytesIO()
    image.save(buf, "JPEG")
    return buf.getvalue()


async def test_live_decisions_track_frame_content():
    frames = [jpeg(i % 2 == 0) for i in range(6)]
    async with aiohttp.ClientSession() as http:
        stream = VisionStream(
            questions={"red": llm.Probability("A red circle is visible.")},
            state="A test camera.",
            max_inflight=8,
            http_session=http,
        )
        decisions: list[FrameDecision] = []
        stats: list[dict] = []
        stream.on("decision", decisions.append)
        stream.on("session_stats", stats.append)
        stream.on("error", lambda e: pytest.fail(f"gateway error: {e.error}"))

        for frame in frames:
            stream.push_frame(frame)
            await asyncio.sleep(0.5)

        async def all_answered() -> None:
            while len(decisions) < len(frames):
                await asyncio.sleep(0.05)

        await asyncio.wait_for(all_answered(), 60)
        await stream.aclose()

    by_frame = {d.frame: d.values["red"] for d in decisions}
    for index in range(len(frames)):
        if index % 2 == 0:
            assert by_frame[index] > 0.5, f"frame {index} has a red circle: {by_frame[index]}"
        else:
            assert by_frame[index] < 0.5, f"frame {index} is blank: {by_frame[index]}"
    assert stats and stats[-1].get("decisions") == len(frames)
    print("session.stats", stats[-1])
