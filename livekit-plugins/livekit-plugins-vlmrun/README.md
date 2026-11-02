# VLM Run plugin for LiveKit Agents

Vision for LiveKit agents through the [VLM Run gateway](https://docs.vlm.run/gateway/introduction):
an LLM that reads camera frames, typed decisions over images, per-frame decisions on live video,
and detection, segmentation, pose and OCR tools.

## Installation

```bash
pip install "livekit-agents[vlmrun]"
```

## Pre-requisites

Set `VLMRUN_API_KEY` (from [app.vlm.run](https://app.vlm.run)). Without a key, the chat and
decision routes use the gateway's anonymous tier, which serves free models at low rate limits.
Live video (`VisionStream`) always needs a key. Set `VLMRUN_GATEWAY_BASE_URL` to use another
gateway.

## LLM

The gateway's OpenAI-compatible chat route. Images in the chat context, including video frames,
are sent to the model.

```python
from livekit.agents import AgentSession
from livekit.plugins import vlmrun

session = AgentSession(llm=vlmrun.LLM(model="qwen/qwen3.8-27b"), stt=..., tts=...)
```

Tool calling works on `qwen/qwen3.8-27b` and the provider models (`google/gemini-*`,
`moonshotai/kimi-k3`, `minimax/minimax-m3`). Sampling settings are sent only when set, because
the gateway rejects settings a model doesn't support.

## Decisions over images

A `DecisionModel` that declares `image_input`: images in user messages, and any passed as
`images=`, are read with the transcript (the newest 8).

```python
from livekit.agents import llm
from livekit.plugins import vlmrun

model = vlmrun.DecisionModel()  # google/diffusiongemma-26b-a4b-it
response = await model.evaluate(
    input=chat_ctx,
    decisions={"holding": llm.Probability("The gripper is holding the cube.")},
    images=[llm.ImageContent(image=latest_frame)],
)
```

## Live video

`VisionStream` reads frames from a room's video track over the gateway's WebSocket and emits one
set of answers per frame. It reuses the decision types above.

```python
stream = vlmrun.VisionStream(
    questions={"person": llm.Probability("A person is in the frame.")},
    fps=2,
)


@stream.on("decision")
def on_decision(decision: vlmrun.FrameDecision) -> None:
    if decision.results["person"].value > 0.8:
        ...


await stream.start(ctx.room)
```

Frames that arrive while every read is busy replace the waiting frame, so answers never fall
behind the video. Sessions are renewed before the gateway's lifetime limit. Use `push_frame()` to
feed frames that don't come from a room, such as a robot camera.

## Vision tools

`VisionTools` gives the LLM `detect_objects`, `segment_objects`, `detect_poses` and `read_text`,
run on the latest frame of the room's video track.

```python
agent = Agent(instructions="...", tools=[vlmrun.VisionTools()])
```

The same operations are available as functions: `vlmrun.detect`, `vlmrun.segment`, `vlmrun.pose`
and `vlmrun.ocr`.

## Examples

- `examples/voice_agents/vlmrun_camera_agent.py`: answers questions about what the camera sees.
- `examples/voice_agents/vlmrun_watcher_agent.py`: speaks up when something appears on camera.
