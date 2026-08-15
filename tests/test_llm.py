"""OpenAI-compatible extraction tests.

Run against a local stub server rather than a real provider, so they need no
API key and no network. The point is the fallback ladder: free providers vary
wildly in how much of the structured-output spec they implement, so the
extractor has to degrade from strict json_schema, to JSON mode, to a
validation-feedback retry - and these assert each rung actually works.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from earnings_agent.llm import (
    Brief,
    ExtractionError,
    OpenAICompatExtractor,
    RefusalError,
    _strict_schema,
)

VALID = {
    "guidance_status": "written_guidance_found",
    "guidance_status_note": "Company guided Q3 revenue.",
    "guidance": [{
        "metric": "Total revenue", "period": "Q3 2026", "value": "$61-64 billion",
        "quote": "We expect third quarter 2026 total revenue to be in the range of $61-64 billion.",
        "change_vs_prior": "not stated in this release",
    }],
    "key_points": [{"point": "Revenue grew.", "quote": "Revenue was up 26%."}],
    "watch_items": ["Capex guidance raised"],
    "tone_shift": "nothing notable",
}


class Stub(BaseHTTPRequestHandler):
    """Scripted OpenAI-compatible endpoint. Behaviour set per-test on the class."""

    replies: list = []          # each: (status, body) - consumed in order
    seen: list = []             # request payloads, for assertions

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        status, payload = type(self).replies.pop(0)
        if status != 200:
            out = json.dumps({"error": {"message": payload, "type": "invalid_request_error"}})
        else:
            content, finish = payload
            out = json.dumps({
                "id": "stub", "object": "chat.completion", "created": 0, "model": "stub-model",
                "choices": [{
                    "index": 0, "finish_reason": finish,
                    "message": {"role": "assistant", "content": content},
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        raw = out.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def stub():
    Stub.replies, Stub.seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield Stub, f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def extractor(base_url: str) -> OpenAICompatExtractor:
    # max_retries=0 so a scripted 400 is not silently retried past its script
    e = OpenAICompatExtractor("stub-key", base_url, "stub-model")
    e.client = e.client.with_options(max_retries=0)
    return e


def run(base_url: str) -> tuple[Brief, str]:
    return extractor(base_url).extract("META", "2026-07-29", "<press release text>")


class TestSchemaGeneration:
    def test_no_refs_survive_flattening(self):
        s = json.dumps(_strict_schema())
        assert "$ref" not in s and "$defs" not in s

    def test_objects_are_closed_and_fully_required(self):
        s = _strict_schema()
        assert s["additionalProperties"] is False
        assert set(s["required"]) == set(s["properties"].keys())
        item = s["properties"]["guidance"]["items"]
        assert item["additionalProperties"] is False
        assert "quote" in item["required"], "citations must not be optional"


class TestFallbackLadder:
    def test_strict_json_schema_when_provider_supports_it(self, stub):
        Stub_, url = stub
        Stub_.replies = [(200, (json.dumps(VALID), "stop"))]
        brief, model = run(url)
        assert brief.guidance[0].value == "$61-64 billion"
        assert model == "stub-model"
        assert len(Stub_.seen) == 1
        assert Stub_.seen[0]["response_format"]["type"] == "json_schema"

    def test_falls_back_to_json_mode_when_schema_rejected(self, stub):
        Stub_, url = stub
        Stub_.replies = [
            (400, "response_format.json_schema is not supported"),
            (200, (json.dumps(VALID), "stop")),
        ]
        brief, _ = run(url)
        assert brief.guidance_status == "written_guidance_found"
        assert [s["response_format"]["type"] for s in Stub_.seen] == ["json_schema", "json_object"]
        # the schema must reach a provider that cannot enforce it
        assert "guidance_status" in Stub_.seen[1]["messages"][0]["content"]

    def test_falls_back_again_when_json_mode_rejected(self, stub):
        Stub_, url = stub
        Stub_.replies = [
            (400, "response_format is not supported"),
            (400, "response_format is not supported"),
            (200, (json.dumps(VALID), "stop")),
        ]
        brief, _ = run(url)
        assert brief.guidance_status == "written_guidance_found"
        assert "response_format" not in Stub_.seen[2]

    def test_strips_markdown_fence(self, stub):
        Stub_, url = stub
        Stub_.replies = [(200, ("```json\n" + json.dumps(VALID) + "\n```", "stop"))]
        brief, _ = run(url)
        assert brief.guidance_status == "written_guidance_found"


class TestRetryOnInvalidOutput:
    def test_retries_with_the_validation_error(self, stub):
        Stub_, url = stub
        Stub_.replies = [
            (400, "json_schema unsupported"),
            (200, (json.dumps({"guidance_status": "nonsense_value"}), "stop")),
            (200, (json.dumps(VALID), "stop")),
        ]
        brief, _ = run(url)
        assert brief.guidance_status == "written_guidance_found"
        # the retry has to show the model what was wrong, not just ask again
        last = Stub_.seen[-1]["messages"][-1]["content"]
        assert "did not validate" in last and "guidance_status" in last

    def test_gives_up_after_one_retry(self, stub):
        Stub_, url = stub
        Stub_.replies = [
            (400, "json_schema unsupported"),
            (200, ("not json at all", "stop")),
            (200, ("still not json", "stop")),
        ]
        with pytest.raises(ExtractionError, match="invalid output twice"):
            run(url)


class TestErrorSurfaces:
    def test_content_filter_is_a_refusal(self, stub):
        Stub_, url = stub
        Stub_.replies = [(200, ("", "content_filter"))]
        with pytest.raises(RefusalError):
            run(url)

    def test_truncation_names_the_setting_to_change(self, stub):
        Stub_, url = stub
        Stub_.replies = [(200, ('{"guidance_status":', "length"))]
        with pytest.raises(ExtractionError, match="max_tokens"):
            run(url)


class TestEmptyGuidanceIsValid:
    """Apple/Microsoft/Alphabet publish no written guidance. That must round-trip
    as a real answer, not fail validation or get coerced into something else."""

    def test_no_written_guidance_round_trips(self, stub):
        Stub_, url = stub
        payload = dict(VALID, guidance_status="no_written_guidance", guidance=[],
                       guidance_status_note="Apple does not provide written guidance; "
                                            "it guides verbally on the call.")
        Stub_.replies = [(200, (json.dumps(payload), "stop"))]
        brief, _ = run(url)
        assert brief.guidance == []
        assert brief.guidance_status == "no_written_guidance"
