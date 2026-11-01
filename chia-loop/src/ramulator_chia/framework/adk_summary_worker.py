"""Run ADK's summarizer, not an ADK agent or runner.

CHIA and ADK pin incompatible web-server dependencies. This small stdio bridge
runs in ADK's own environment. It receives selected history, asks the parent to
make ONE model request, and returns ADK's summary event. No tools or credentials
are passed to this process. The parent retains provider settings and accounting.
"""

import asyncio
import json
import sys
from importlib.metadata import version

from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part


class ParentModel(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        print(
            json.dumps({"request": llm_request.model_dump(mode="json", exclude_none=True)}),
            flush=True,
        )
        yield LlmResponse.model_validate(json.loads(sys.stdin.readline()))


async def main():
    if version("google-adk") != "2.3.0":
        raise RuntimeError("unqualified ADK summarizer version")
    incoming = json.loads(sys.stdin.readline())
    # Tool payloads are text records here so ADK's 2,000-character tool preview
    # cannot silently omit source or measurement fields. Native history itself
    # is never rewritten into these records: this is only the summary input.
    events = [
        Event(
            author="user", timestamp=float(i), content=Content(role="user", parts=[Part(text=text)])
        )
        for i, text in enumerate(incoming["records"])
    ]
    summary = await LlmEventSummarizer(
        llm=ParentModel(model=incoming["model"]),
        prompt_template=incoming["prompt_template"],
    ).maybe_summarize_events(events=events)
    if summary is None:
        raise RuntimeError("ADK returned no summary")
    print(json.dumps({"summary": summary.model_dump(mode="json", exclude_none=True)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
