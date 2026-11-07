"""OpenAI 호출 래퍼 (Responses API): 재시도, 오류 변환, 스트리밍, 요약/질의응답/비교."""

import os
import time
from collections.abc import Callable, Iterator

import openai
from openai import OpenAI

from src import prompts
from src.pdf_utils import (
    MAX_CONTEXT_CHARS,
    PdfDocument,
    chunk_pages,
    format_pages,
    format_selected,
    select_relevant_pages,
)

MAX_RETRIES = 3  # 일시적 오류 시 재시도 횟수 (최초 시도 제외)
BACKOFF_SECONDS = 1.0  # 1s, 2s, 4s
REQUEST_TIMEOUT = 180.0
HISTORY_MESSAGES = 6  # 질의응답 시 함께 보낼 직전 대화 메시지 수

_sleep = time.sleep  # 테스트에서 교체 가능

ProgressCallback = Callable[[str], None]


class LLMError(Exception):
    """사용자에게 그대로 보여줄 수 있는 메시지를 담은 OpenAI 호출 오류."""


FAKE_ENV = "PDF_ANALYZER_FAKE_LLM"


def is_fake_mode() -> bool:
    return os.environ.get(FAKE_ENV) == "1"


def get_client(api_key: str) -> OpenAI:
    if is_fake_mode():
        from src.fake_llm import FakeClient

        return FakeClient()
    # 재시도는 직접 관리하므로 SDK 자체 재시도는 끈다.
    return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)


def _is_transient(e: Exception) -> bool:
    if isinstance(e, openai.RateLimitError):
        return getattr(e, "code", None) != "insufficient_quota"
    return isinstance(e, (openai.APIConnectionError, openai.InternalServerError))


def _to_user_error(e: Exception) -> LLMError:
    if isinstance(e, openai.AuthenticationError):
        msg = "OpenAI API 키가 올바르지 않습니다. OPENAI_API_KEY 설정을 확인하세요."
    elif isinstance(e, openai.PermissionDeniedError):
        msg = "이 API 키로는 해당 모델을 사용할 권한이 없습니다. 계정 권한이나 OPENAI_MODEL 설정을 확인하세요."
    elif isinstance(e, openai.NotFoundError):
        msg = "모델을 찾을 수 없습니다. OPENAI_MODEL 설정의 모델 이름을 확인하세요."
    elif isinstance(e, openai.RateLimitError):
        if getattr(e, "code", None) == "insufficient_quota":
            msg = "OpenAI 사용 한도(크레딧)가 소진되었습니다. OpenAI 계정의 결제 설정을 확인하세요."
        else:
            msg = "요청이 너무 많아 처리하지 못했습니다. 잠시 후 다시 시도하세요."
    elif isinstance(e, openai.APITimeoutError):
        msg = "OpenAI 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요."
    elif isinstance(e, openai.APIConnectionError):
        msg = "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하세요."
    elif isinstance(e, openai.BadRequestError):
        msg = f"요청이 거부되었습니다. 문서가 너무 길거나 내용에 문제가 있을 수 있습니다. ({e.message})"
    elif isinstance(e, openai.APIStatusError):
        msg = f"OpenAI 서버 오류가 발생했습니다 (HTTP {e.status_code}). 잠시 후 다시 시도하세요."
    else:
        msg = f"분석 중 알 수 없는 오류가 발생했습니다: {e}"
    return LLMError(msg)


def _backoff(attempt: int) -> None:
    _sleep(BACKOFF_SECONDS * (2**attempt))


def ask(client: OpenAI, model: str, input, instructions: str | None = prompts.SYSTEM_PROMPT) -> str:
    """요청을 보내고 전체 응답 텍스트를 반환한다. 일시적 오류는 재시도한다."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.responses.create(model=model, instructions=instructions, input=input)
            return response.output_text
        except Exception as e:
            if _is_transient(e) and attempt < MAX_RETRIES:
                _backoff(attempt)
                continue
            raise _to_user_error(e) from e
    raise AssertionError("unreachable")


def stream(client: OpenAI, model: str, input, instructions: str | None = prompts.SYSTEM_PROMPT) -> Iterator[str]:
    """응답 텍스트를 조각 단위로 내보낸다. 첫 조각이 나오기 전의 일시적 오류만 재시도한다."""
    attempt = 0
    while True:
        started = False
        try:
            events = client.responses.create(
                model=model, instructions=instructions, input=input, stream=True
            )
            for event in events:
                if event.type == "response.output_text.delta":
                    started = True
                    yield event.delta
                elif event.type == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise LLMError(f"모델이 응답 생성에 실패했습니다: {getattr(error, 'message', '알 수 없는 오류')}")
                elif event.type == "error":
                    raise LLMError(f"모델 응답 중 오류가 발생했습니다: {getattr(event, 'message', '')}")
            return
        except LLMError:
            raise
        except Exception as e:
            if not started and _is_transient(e) and attempt < MAX_RETRIES:
                _backoff(attempt)
                attempt += 1
                continue
            raise _to_user_error(e) from e


def _group_texts(texts: list[str], max_chars: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    size = 0
    for t in texts:
        if current and size + len(t) > max_chars:
            groups.append("\n\n".join(current))
            current, size = [], 0
        current.append(t)
        size += len(t)
    if current:
        groups.append("\n\n".join(current))
    return groups


def _notify(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress:
        on_progress(message)


def summarize(
    client: OpenAI,
    model: str,
    doc: PdfDocument,
    language: str,
    on_progress: ProgressCallback | None = None,
) -> Iterator[str]:
    """문서 요약을 스트리밍한다. 길면 map-reduce(부분 요약 → 종합)로 처리한다."""
    chunks = chunk_pages(doc.pages)
    if not chunks:
        raise LLMError("문서에서 추출된 텍스트가 없어 요약할 수 없습니다.")

    full_text = format_pages(doc.name, [p for c in chunks for p in c])
    if len(full_text) <= MAX_CONTEXT_CHARS:
        yield from stream(client, model, prompts.summary_prompt(doc.name, full_text, language))
        return

    notes: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        _notify(on_progress, f"긴 문서를 나누어 분석하는 중... ({i}/{len(chunks)})")
        content = format_pages(doc.name, chunk)
        notes.append(ask(client, model, prompts.chunk_summary_prompt(doc.name, content, language)))

    # 부분 메모를 합쳐도 한도를 넘으면 한 번 더 압축한다.
    while len("\n\n".join(notes)) > MAX_CONTEXT_CHARS and len(notes) > 1:
        groups = _group_texts(notes, MAX_CONTEXT_CHARS)
        if len(groups) == len(notes):
            break
        _notify(on_progress, f"부분 요약을 압축하는 중... ({len(groups)}개 묶음)")
        notes = [
            ask(client, model, prompts.chunk_summary_prompt(doc.name, g, language)) for g in groups
        ]

    _notify(on_progress, "부분 요약을 종합하는 중...")
    yield from stream(
        client, model, prompts.reduce_summary_prompt(doc.name, "\n\n".join(notes), language)
    )


def answer(
    client: OpenAI,
    model: str,
    docs: list[PdfDocument],
    question: str,
    history: list[dict],
    language: str,
) -> Iterator[str]:
    """질문과 관련된 페이지를 골라 근거 기반 답변을 스트리밍한다."""
    selected = select_relevant_pages(docs, question)
    if not selected:
        raise LLMError("선택한 문서에서 추출된 텍스트가 없습니다.")
    context = format_selected(selected)
    messages = [
        {"role": m["role"], "content": m["content"]} for m in history[-HISTORY_MESSAGES:]
    ]
    messages.append({"role": "user", "content": prompts.qa_prompt(context, question, language)})
    yield from stream(client, model, messages)


def compare(
    client: OpenAI,
    model: str,
    docs: list[PdfDocument],
    language: str,
    on_progress: ProgressCallback | None = None,
) -> Iterator[str]:
    """여러 문서를 비교한다. 문서가 길면 먼저 요약본을 만들어 비교에 사용한다."""
    if len(docs) < 2:
        raise LLMError("비교하려면 문서가 2개 이상 필요합니다.")
    budget = MAX_CONTEXT_CHARS // len(docs)
    contents: list[tuple[str, str]] = []
    for doc in docs:
        text = format_pages(doc.name, [p for c in chunk_pages(doc.pages) for p in c])
        if len(text) > budget:
            _notify(on_progress, f"'{doc.name}'이(가) 길어 먼저 요약하는 중...")
            text = "".join(summarize(client, model, doc, language, on_progress))
        contents.append((doc.name, text))
    _notify(on_progress, "문서를 비교하는 중...")
    yield from stream(client, model, prompts.compare_prompt(contents, language))
