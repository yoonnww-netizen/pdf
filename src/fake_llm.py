"""테스트용 가짜 OpenAI 클라이언트.

환경변수 PDF_ANALYZER_FAKE_LLM=1 이면 llm.get_client()가 이 클라이언트를 돌려준다.
API를 호출하지 않고, 프롬프트 종류(요약/부분 메모/질의응답/비교)에 맞는 형식의 응답을 만든다.
질문에 "force-error"가 들어 있으면 인증 오류를 흉내 낸다.
"""

import re
import time
from types import SimpleNamespace

import httpx2
import openai

from src import prompts

STREAM_DELAY = 0.01
NOTE_DELAY = 0.3  # 긴 문서 진행 표시를 E2E에서 관찰할 수 있도록


def _language(text: str) -> str:
    return "English" if re.search(r"\bin English\b", text) else "한국어"


def _sources(text: str) -> list[tuple[str, str]]:
    return re.findall(r"\[([^\[\]\n]+?) p\.(\d+)\]", text)


def _doc_name(text: str) -> str:
    m = re.search(r'document "([^"]+)"', text)
    return m.group(1) if m else "document"


def _summary(text: str) -> str:
    one, detail, points, keywords = prompts.SUMMARY_HEADINGS[_language(text)]
    pages = [n for _, n in _sources(text)] or ["1"]
    return (
        f"## {one}\n[FAKE] {_doc_name(text)} 문서의 가짜 요약입니다.\n\n"
        f"## {detail}\n가짜 상세 요약 첫 문단입니다.\n\n가짜 상세 요약 둘째 문단입니다.\n\n"
        f"## {points}\n- 첫 번째 핵심 포인트 (p.{pages[0]})\n- 마지막 핵심 포인트 (p.{pages[-1]})\n\n"
        f"## {keywords}\n가짜, 테스트, 요약\n"
    )


def _compare(text: str) -> str:
    common, diff, overall = prompts.COMPARE_HEADINGS[_language(text)]
    names = re.findall(r'<document name="([^"]+)">', text)
    header = "| 항목 | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)
    row = "| 내용 | " + " | ".join(f"[{n} p.1]" for n in names) + " |"
    return (
        f"## {common}\n- [FAKE] 모든 문서가 테스트용입니다.\n\n"
        f"## {diff}\n{header}\n{sep}\n{row}\n\n"
        f"## {overall}\n[FAKE] 가짜 종합 의견입니다.\n"
    )


def _answer(text: str) -> str:
    question = text.rsplit("Question:", 1)[-1].strip()
    if "force-error" in question:
        request = httpx2.Request("POST", "https://fake.local/v1/responses")
        raise openai.AuthenticationError(
            "fake auth error", response=httpx2.Response(401, request=request), body=None
        )
    sources = _sources(text)
    if not sources or "CEO" in question:
        return prompts.NOT_FOUND[_language(text)]
    name, page = sources[0]
    return f"[FAKE] '{question}'에 대한 가짜 답변입니다. [{name} p.{page}]"


def _respond(input) -> str:
    text = input[-1]["content"] if isinstance(input, list) else input
    if "Question:" in text:
        return _answer(text)
    if "Compare the following documents" in text:
        return _compare(text)
    if "<document_part>" in text:
        time.sleep(NOTE_DELAY)
        pages = [n for _, n in _sources(text)] or ["1"]
        return f"[FAKE notes] 부분 메모 (p.{pages[0]}-{pages[-1]})"
    return _summary(text)


def _events(text: str):
    for piece in re.findall(r"\S+\s*", text):
        time.sleep(STREAM_DELAY)
        yield SimpleNamespace(type="response.output_text.delta", delta=piece)
    yield SimpleNamespace(type="response.completed")


class _FakeResponses:
    def create(self, *, model, input, instructions=None, stream=False, **_):
        text = _respond(input)
        return _events(text) if stream else SimpleNamespace(output_text=text)


class FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()
