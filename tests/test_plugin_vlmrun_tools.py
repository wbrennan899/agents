from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import APIError, APIStatusError
from livekit.agents.llm import ToolError
from livekit.plugins.vlmrun import tools

pytestmark = pytest.mark.unit

PNG = b"\x89PNG\r\n\x1a\nfake"


def completion(content: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content)
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def session_for(body: Any, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.status = status
    response.headers = {"x-request-id": "req-1"}
    response.json = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=json.dumps(body))
    session = MagicMock()
    session.post = MagicMock(return_value=response)
    return session


def sent_body(session: MagicMock) -> dict[str, Any]:
    return session.post.call_args.kwargs["json"]


def sent_headers(session: MagicMock) -> dict[str, str]:
    return session.post.call_args.kwargs["headers"]


async def test_detect_without_query_uses_florence_od():
    session = session_for(
        completion(
            {
                "model": "microsoft/florence-2-base-ft",
                "method": "od",
                "image_width": 640,
                "image_height": 480,
                "content": {
                    "object": "img.detect.bboxes",
                    "items": [{"bbox_xywh": [0.1, 0.2, 0.3, 0.4], "label": "ball"}],
                },
            }
        )
    )
    result = await tools.detect("https://example.com/a.jpg", api_key="k", http_session=session)

    body = sent_body(session)
    assert body["model"] == tools.DETECT_MODEL
    assert body["method"] == "od"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}
    ]
    assert session.post.call_args.args[0] == "https://gateway.vlm.run/v1/openai/chat/completions"
    assert result.items[0].label == "ball"
    assert result.items[0].box == tools.BoundingBox(0.1, 0.2, 0.3, 0.4)
    assert result.items[0].box.to_pixels(640, 480) == (64, 96, 192, 192)
    assert (result.image_width, result.image_height) == (640, 480)


async def test_detect_with_query_uses_sam_without_mask():
    session = session_for(
        completion(
            {
                "model": "facebook/sam3.1",
                "method": "segment",
                "content": {
                    "object": "img.segment.masks",
                    "items": [
                        {
                            "bbox_xywh": [0.1, 0.1, 0.3, 0.4],
                            "label": "red cube",
                            "score": 0.91,
                            "area": 0.08,
                            "instance_id": 1,
                        }
                    ],
                },
            }
        )
    )
    result = await tools.detect(PNG, query="red cube", http_session=session)

    body = sent_body(session)
    assert body["model"] == tools.SEGMENT_MODEL
    assert body["method"] == "segment"
    assert body["method_params"] == {"mask_format": "none"}
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "red cube"}
    assert result.items[0].label == "red cube"
    assert result.items[0].score == 0.91


async def test_segment_decodes_label_map():
    mask = base64.b64encode(PNG).decode()
    session = session_for(
        completion(
            {
                "model": "facebook/sam3.1",
                "method": "segment",
                "content": {
                    "object": "img.segment.masks",
                    "items": [
                        {"bbox_xywh": [0, 0, 1, 1], "label": "cup", "area": 0.5, "instance_id": 2}
                    ],
                    "mask": {
                        "format": "png",
                        "height": 4,
                        "width": 4,
                        "data": f"data:image/png;base64,{mask}",
                    },
                },
            }
        )
    )
    result = await tools.segment(PNG, "cup", http_session=session)

    assert "method_params" not in sent_body(session)
    assert result.mask_png == PNG
    assert result.items[0].instance_id == 2
    assert result.items[0].area == 0.5


async def test_pose_pairs_keypoint_names():
    session = session_for(
        completion(
            {
                "model": "usyd-community/vitpose-plus-large",
                "method": "pose",
                "content": {
                    "object": "img.pose.kpts",
                    "kpts_labels": ["nose", "left_eye"],
                    "items": [
                        {
                            "bbox_xywh": [0.2, 0.1, 0.3, 0.8],
                            "label": "person",
                            "kpts_xy": [[0.35, 0.15], [0.36, 0.14]],
                            "kpts_score": [0.9, 0.8],
                        }
                    ],
                },
            }
        )
    )
    result = await tools.pose(PNG, http_session=session)

    assert sent_body(session)["method"] == "pose"
    person = result.items[0]
    assert [k.name for k in person.keypoints] == ["nose", "left_eye"]
    assert person.keypoints[0].x == 0.35
    assert person.keypoints[1].score == 0.8
    assert person.keypoints[0].xyz is None


async def test_hand_pose_keeps_3d_joints():
    session = session_for(
        completion(
            {
                "content": {
                    "object": "world.pose.kpts",
                    "items": [
                        {
                            "bbox_xywh": [0.4, 0.4, 0.1, 0.1],
                            "label": "right_hand",
                            "score": 0.97,
                            "kpts_xy": [[0.45, 0.45]],
                            "kpts_xyz": [[0.01, -0.02, 0.5]],
                        }
                    ],
                }
            }
        )
    )
    result = await tools.pose(PNG, model=tools.HAND_POSE_MODEL, http_session=session)

    assert sent_body(session)["model"] == tools.HAND_POSE_MODEL
    hand = result.items[0]
    assert hand.label == "right_hand"
    assert hand.keypoints[0].name == "0"
    assert hand.keypoints[0].xyz == (0.01, -0.02, 0.5)


async def test_ocr_lines_join_into_text():
    session = session_for(
        completion(
            {
                "model": "paddleocr/pp-ocrv6",
                "method": "ocr",
                "content": {
                    "object": "doc.ocr.lines",
                    "items": [
                        {"bbox_xywh": [0.1, 0.1, 0.5, 0.1], "text": "HELLO", "score": 0.99},
                        {"bbox_xywh": [0.1, 0.3, 0.5, 0.1], "text": "WORLD", "score": 0.98},
                    ],
                },
            }
        )
    )
    result = await tools.ocr(PNG, http_session=session)

    body = sent_body(session)
    assert (body["model"], body["method"]) == (tools.OCR_MODEL, "ocr")
    assert result.text == "HELLO\nWORLD"
    assert result.lines[1].score == 0.98


async def test_ocr_markdown_model_returns_plain_text():
    session = session_for(completion("# Receipt\n\nTotal: $4.20"))
    result = await tools.ocr(PNG, model="zai-org/glm-ocr", http_session=session)

    assert "method" not in sent_body(session)
    assert result.text == "# Receipt\n\nTotal: $4.20"
    assert result.lines == []


async def test_ocr_markdown_inside_json_envelope():
    session = session_for(completion({"model": "zai-org/glm-ocr", "content": "Total: $4.20"}))
    result = await tools.ocr(PNG, model="zai-org/glm-ocr", http_session=session)

    assert result.text == "Total: $4.20"


async def test_payload_at_top_level_is_accepted():
    session = session_for(
        completion({"object": "img.detect.bboxes", "items": [{"bbox_xywh": [0, 0, 1, 1]}]})
    )
    result = await tools.detect(PNG, http_session=session)

    assert result.items[0].label is None


async def test_anonymous_requests_omit_authorization(monkeypatch):
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    session = session_for(completion({"content": {"items": []}}))
    await tools.detect(PNG, http_session=session)

    headers = sent_headers(session)
    assert "Authorization" not in headers
    assert headers["X-VLMRun-Client"].startswith("livekit-agents/")


async def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VLMRUN_API_KEY", "env-key")
    session = session_for(completion({"content": {"items": []}}))
    await tools.detect(PNG, http_session=session)

    assert sent_headers(session)["Authorization"] == "Bearer env-key"


async def test_base_url_override():
    session = session_for(completion({"content": {"items": []}}))
    await tools.detect(PNG, base_url="https://gw.internal/", http_session=session)

    assert session.post.call_args.args[0] == "https://gw.internal/v1/openai/chat/completions"


async def test_http_error_raises_status_error():
    session = session_for({"detail": "model_not_entitled"}, status=403)
    with pytest.raises(APIStatusError) as exc:
        await tools.detect(PNG, http_session=session)
    assert exc.value.status_code == 403
    assert exc.value.request_id == "req-1"


async def test_malformed_response_raises():
    session = session_for({"choices": []})
    with pytest.raises(APIError):
        await tools.detect(PNG, http_session=session)


def test_bytes_mime_sniffing():
    assert tools._image_url(PNG).startswith("data:image/png;base64,")
    assert tools._image_url(b"\xff\xd8\xff\xe0jpeg").startswith("data:image/jpeg;base64,")
    assert tools._image_url(b"RIFF\x00\x00\x00\x00WEBPVP8 ").startswith("data:image/webp;base64,")
    assert tools._image_url("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


def test_video_frame_is_encoded_and_downscaled():
    from livekit import rtc

    frame = rtc.VideoFrame(2048, 1024, rtc.VideoBufferType.RGBA, bytes(2048 * 1024 * 4))
    url = tools._encode_frame(frame, 512)
    assert url.startswith("data:image/jpeg;base64,")

    import io

    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(img.size) == 512


def test_vision_tools_exposes_function_tools():
    vt = tools.VisionTools(frame_source=lambda: None)
    assert sorted(t.info.name for t in vt.tools) == [
        "detect_objects",
        "detect_poses",
        "read_text",
        "segment_objects",
    ]


async def test_vision_tools_without_frame_raises_tool_error():
    vt = tools.VisionTools(frame_source=lambda: None)
    with pytest.raises(ToolError, match="no camera frame"):
        await vt.read_text()


async def test_vision_tools_detect_returns_llm_friendly_dicts():
    session = session_for(
        completion(
            {
                "content": {
                    "items": [
                        {"bbox_xywh": [0.12345, 0.2, 0.3, 0.4], "label": "cup", "score": 0.87654}
                    ]
                }
            }
        )
    )
    vt = tools.VisionTools(frame_source=lambda: PNG, http_session=session)
    out = await vt.detect_objects(query="cup")

    assert out == [{"label": "cup", "box": [0.123, 0.2, 0.3, 0.4], "score": 0.877}]
    assert sent_body(session)["model"] == tools.SEGMENT_MODEL


async def test_vision_tools_pose_output():
    session = session_for(
        completion(
            {
                "content": {
                    "kpts_labels": ["nose"],
                    "items": [
                        {"bbox_xywh": [0, 0, 1, 1], "kpts_xy": [[0.5, 0.25]], "kpts_score": [0.9]}
                    ],
                }
            }
        )
    )
    vt = tools.VisionTools(frame_source=lambda: PNG, http_session=session)
    out = await vt.detect_poses()

    assert out == [{"box": [0.0, 0.0, 1.0, 1.0], "keypoints": {"nose": [0.5, 0.25, 0.9]}}]


async def test_vision_tools_read_text_empty():
    session = session_for(completion({"content": {"object": "doc.ocr.lines", "items": []}}))
    vt = tools.VisionTools(frame_source=lambda: PNG, http_session=session)

    assert await vt.read_text() == "No text found."


async def test_vision_tools_maps_api_errors_to_tool_error():
    session = session_for({"error": "rate_limit_exceeded"}, status=429)
    vt = tools.VisionTools(frame_source=lambda: PNG, http_session=session)

    with pytest.raises(ToolError, match="429"):
        await vt.segment_objects(query="cup")


async def test_vision_tools_model_overrides():
    session = session_for(completion({"content": {"items": []}}))
    vt = tools.VisionTools(
        frame_source=lambda: PNG, pose_model=tools.HAND_POSE_MODEL, http_session=session
    )
    await vt.detect_poses()

    assert sent_body(session)["model"] == tools.HAND_POSE_MODEL
