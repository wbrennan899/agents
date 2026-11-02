"""Voice agent that answers questions about what the user's camera sees, using VLM Run.

The agent keeps the latest frame from the user's camera (or screen share) and attaches it to
each user turn, so the VLM Run model answers with the current view in front of it.

Requires VLMRUN_API_KEY for the default model (qwen/qwen3.8-27b). Without a key the gateway's
anonymous tier serves qwen/qwen3.5-0.8b only.

Run with:
    python vlmrun_camera_agent.py console   # local mic and speaker, no camera
    python vlmrun_camera_agent.py dev       # join a LiveKit room and publish a camera
"""

import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    get_job_context,
    inference,
    llm,
)
from livekit.plugins import vlmrun

logger = logging.getLogger("vlmrun-camera-agent")

load_dotenv()


class CameraAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant that can see through the user's camera. "
            "Each user message may include the current camera frame. "
            "Answer questions about what you see in one or two short spoken sentences. "
            "Do not use markdown, emojis or special characters. "
            "If no image is attached, say you can't see anything yet.",
        )
        self._latest_frame: rtc.VideoFrame | None = None
        self._video_stream: rtc.VideoStream | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def on_enter(self) -> None:
        room = get_job_context().room

        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._watch(publication.track)
                    break

        @room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._watch(track)

        self.session.generate_reply(
            instructions="Greet the user and offer to describe what you see."
        )

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if self._latest_frame is not None:
            # 768px keeps detail for small text while bounding image tokens per turn.
            new_message.content.append(
                llm.ImageContent(
                    image=self._latest_frame, inference_width=768, inference_height=768
                )
            )

    def _watch(self, track: rtc.Track) -> None:
        if self._video_stream is not None:
            old = self._video_stream
            task = asyncio.create_task(old.aclose())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        stream = rtc.VideoStream(track)
        self._video_stream = stream

        async def _read() -> None:
            async for event in stream:
                self._latest_frame = event.frame

        task = asyncio.create_task(_read())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=vlmrun.LLM(model="qwen/qwen3.8-27b"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    await session.start(agent=CameraAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
