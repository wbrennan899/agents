from __future__ import annotations

import asyncio
import io
import os

import aiohttp
import pytest

from livekit.agents import llm
from livekit.plugins.vlmrun.vision_stream import FrameDecision, FrameDropped, VisionStream

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
        dropped: list[FrameDropped] = []
        stats: list[dict] = []
        stream.on("decision", decisions.append)
        stream.on("frame_dropped", dropped.append)
        stream.on("session_stats", stats.append)
        stream.on("error", lambda e: pytest.fail(f"gateway error: {e.error}"))

        # Opening the session takes over a second; frames pushed before then are replaced
        # by newer ones, by design. Warm up so the measured frames meet an open session.
        stream.push_frame(jpeg(False))

        async def settled(count: int) -> None:
            while len(decisions) + len(dropped) < count:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(settled(1), 30)
        first = len(decisions) + len(dropped)

        indices = []
        for frame in frames:
            indices.append(stream.push_frame(frame))
            await asyncio.sleep(0.5)

        await asyncio.wait_for(settled(first + len(frames)), 60)
        await stream.aclose()

    by_frame = {d.frame: d.values["red"] for d in decisions}
    answered = [i for i in indices if i in by_frame]
    assert len(answered) >= len(frames) - 1, f"too many frames dropped: {dropped}"
    for position, index in enumerate(indices):
        if index not in by_frame:
            continue
        if position % 2 == 0:
            assert by_frame[index] > 0.5, f"frame {index} has a red circle: {by_frame[index]}"
        else:
            assert by_frame[index] < 0.5, f"frame {index} is blank: {by_frame[index]}"
    assert stats and stats[-1].get("decisions") == len(decisions)
    print("session.stats", stats[-1])
