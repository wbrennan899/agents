from __future__ import annotations

import base64
import io

import pytest

from livekit.agents import utils
from livekit.agents.llm import ChatContext, ImageContent, decisions
from livekit.plugins import vlmrun

# Calls the live gateway. Without VLMRUN_API_KEY it uses the anonymous tier.
pytestmark = pytest.mark.plugin("vlmrun")


def stop_sign_data_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 40, 260, 200), fill="red")
    draw.text((140, 110), "STOP", fill="white")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


async def test_reads_the_image_not_just_the_text() -> None:
    context = ChatContext.empty()
    context.add_message(
        role="user",
        content=["What do you see?", ImageContent(image=stop_sign_data_url())],
    )
    async with utils.http_context.open():
        response = await vlmrun.DecisionModel().evaluate(
            input=context,
            decisions={
                "red": decisions.Probability("The image shows a red circle."),
                "cat": decisions.Probability("The image shows a cat."),
                "color": decisions.Choice(
                    "What color is the main shape?",
                    options={"red": "Red", "blue": "Blue", "green": "Green"},
                ),
            },
        )
    assert response.errors == {}, response.errors
    assert response.results["red"].value > 0.8, response.results
    assert response.results["cat"].value < 0.2, response.results
    assert response.results["color"].value == "red", response.results
