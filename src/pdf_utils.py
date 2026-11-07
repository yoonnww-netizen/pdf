"""PDF 텍스트 추출, 청크 분할, 질문 관련 페이지 선택.

Streamlit에 의존하지 않는다 (캐싱은 app.py에서 담당).
"""

import hashlib
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

# 한 번의 요약 요청(map 단계)에 넣을 최대 글자 수
CHUNK_CHARS = 20_000
# 한 번의 요청에 넣을 문서 컨텍스트 최대 글자 수 (질의응답, 비교, 단일 요약)
MAX_CONTEXT_CHARS = 60_000
# 페이지당 평균 글자 수가 이보다 적으면 스캔(이미지) PDF로 간주
MIN_CHARS_PER_PAGE = 20


class PdfExtractionError(Exception):
    """사용자에게 그대로 보여줄 수 있는 메시지를 담은 PDF 처리 오류."""


@dataclass(frozen=True)
class PdfDocument:
    name: str
    file_hash: str
    pages: tuple[str, ...]  # 인덱스 0 = 1페이지

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def is_scanned(self) -> bool:
        return self.char_count < MIN_CHARS_PER_PAGE * max(self.page_count, 1)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(data: bytes) -> list[str]:
    """PDF 바이트에서 페이지별 텍스트 리스트를 반환한다.

    읽을 수 없으면 PdfExtractionError를 던진다.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")  # 열기 암호 없이 권한만 걸린 PDF는 빈 암호로 열린다
            except Exception:
                decrypted = 0
            if not decrypted:
                raise PdfExtractionError("암호로 보호된 PDF입니다. 암호를 해제한 뒤 다시 업로드하세요.")
        pages = [_clean(page.extract_text() or "") for page in reader.pages]
    except PdfExtractionError:
        raise
    except Exception as e:
        raise PdfExtractionError(
            "PDF 파일을 읽을 수 없습니다. 파일이 손상되었거나 올바른 PDF 형식이 아닙니다."
        ) from e
    if not pages:
        raise PdfExtractionError("페이지가 없는 PDF입니다.")
    return pages


def load_document(name: str, data: bytes) -> PdfDocument:
    return PdfDocument(name=name, file_hash=file_hash(data), pages=tuple(extract_pages(data)))


def chunk_pages(pages: list[str] | tuple[str, ...], max_chars: int = CHUNK_CHARS) -> list[list[tuple[int, str]]]:
    """페이지를 순서대로 묶어 각 묶음이 max_chars 이하가 되도록 나눈다.

    반환: [[(페이지번호(1부터), 텍스트), ...], ...]. 한 페이지가 max_chars보다 길면 여러 조각으로 자른다.
    빈 페이지는 제외한다.
    """
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = 0
    for page_no, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        pieces = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        for piece in pieces:
            if current and size + len(piece) > max_chars:
                chunks.append(current)
                current, size = [], 0
            current.append((page_no, piece))
            size += len(piece)
    if current:
        chunks.append(current)
    return chunks


def format_pages(name: str, pages: list[tuple[int, str]]) -> str:
    """모델 입력용으로 각 페이지 앞에 [파일명 p.N] 머리말을 붙인다."""
    return "\n\n".join(f"[{name} p.{n}]\n{text}" for n, text in pages if text.strip())


def _bigrams(text: str) -> set[str]:
    # 한국어는 조사가 붙어 단어가 정확히 일치하지 않으므로 글자 2-gram으로 비교한다.
    grams: set[str] = set()
    for word in re.findall(r"\w+", text.lower()):
        if len(word) == 1:
            continue
        grams.update(word[i : i + 2] for i in range(len(word) - 1))
    return grams


def select_relevant_pages(
    docs: list[PdfDocument], question: str, max_chars: int = MAX_CONTEXT_CHARS
) -> list[tuple[str, int, str]]:
    """질문과 관련 있는 페이지를 max_chars 안에서 고른다.

    전체가 한도 안에 들어가면 모든 페이지를 그대로 반환한다.
    반환: [(파일명, 페이지번호, 텍스트), ...] (문서/페이지 원래 순서)
    """
    all_pages = [
        (doc.name, n, text)
        for doc in docs
        for n, text in enumerate(doc.pages, start=1)
        if text.strip()
    ]
    if sum(len(t) for _, _, t in all_pages) <= max_chars:
        return all_pages

    q = _bigrams(question)
    scored = sorted(
        range(len(all_pages)),
        key=lambda i: (-len(q & _bigrams(all_pages[i][2])), i),
    )
    chosen: list[int] = []
    remaining = max_chars
    truncated: dict[int, str] = {}
    for i in scored:
        text = all_pages[i][2]
        if len(text) <= remaining:
            chosen.append(i)
            remaining -= len(text)
        elif not chosen:
            # 가장 관련 있는 페이지 하나가 한도보다 길면 잘라서라도 넣는다.
            chosen.append(i)
            truncated[i] = text[:remaining]
            remaining = 0
        if remaining <= 0:
            break
    chosen.sort()
    return [
        (all_pages[i][0], all_pages[i][1], truncated.get(i, all_pages[i][2]))
        for i in chosen
    ]


def format_selected(selected: list[tuple[str, int, str]]) -> str:
    return "\n\n".join(f"[{name} p.{n}]\n{text}" for name, n, text in selected)
