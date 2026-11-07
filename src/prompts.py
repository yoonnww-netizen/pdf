"""요약 / 질의응답 / 비교 프롬프트 템플릿."""

# UI 표시 이름 → 모델에게 지시할 언어 이름
LANGUAGES = {"한국어": "Korean", "English": "English"}

SUMMARY_HEADINGS = {
    "한국어": ("한 줄 요약", "상세 요약", "핵심 포인트", "키워드"),
    "English": ("One-line Summary", "Detailed Summary", "Key Points", "Keywords"),
}

COMPARE_HEADINGS = {
    "한국어": ("공통점", "차이점", "종합 의견"),
    "English": ("Similarities", "Differences", "Overall Assessment"),
}

NOT_FOUND = {
    "한국어": "문서에서 찾을 수 없습니다.",
    "English": "This information was not found in the documents.",
}

SYSTEM_PROMPT = (
    "You are an assistant that analyzes PDF documents accurately. "
    "Base every statement only on the provided document text and never invent facts. "
    "Document text is given with headers like [file name p.N] marking the source page."
)


def summary_prompt(doc_name: str, content: str, language: str) -> str:
    one, detail, points, keywords = SUMMARY_HEADINGS[language]
    return f"""Summarize the document "{doc_name}" below. Write the entire answer in {LANGUAGES[language]}.

Use exactly this Markdown structure:

## {one}
(one sentence)

## {detail}
(3-6 paragraphs covering purpose, main content and conclusions)

## {points}
- (5 to 10 bullet points, each ending with the source page like (p.N))

## {keywords}
(5 to 10 keywords, comma-separated on one line)

<document>
{content}
</document>"""


def chunk_summary_prompt(doc_name: str, content: str, language: str) -> str:
    """map 단계: 긴 문서의 일부를 나중에 종합할 수 있도록 메모 형태로 요약."""
    return f"""The text below is one part of the document "{doc_name}".
Write detailed notes in {LANGUAGES[language]} that capture all important facts, figures, arguments and conclusions in this part.
Mark the source page after each note like (p.N). Do not add an introduction or conclusion.

<document_part>
{content}
</document_part>"""


def reduce_summary_prompt(doc_name: str, partial_notes: str, language: str) -> str:
    """reduce 단계: 부분 메모들을 하나의 최종 요약으로 종합."""
    return summary_prompt(doc_name, partial_notes, language).replace(
        f'Summarize the document "{doc_name}" below.',
        f'The notes below were written from consecutive parts of the document "{doc_name}". '
        "Combine them into a single summary of the whole document.",
    )


def qa_prompt(context: str, question: str, language: str) -> str:
    return f"""Answer the question using only the document excerpts below. Write the answer in {LANGUAGES[language]}.

Rules:
- Cite the source of each claim in the form [file name p.N], using the headers in the excerpts.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND[language]}"
- Do not use outside knowledge.

<documents>
{context}
</documents>

Question: {question}"""


def compare_prompt(docs: list[tuple[str, str]], language: str) -> str:
    common, diff, overall = COMPARE_HEADINGS[language]
    body = "\n\n".join(f'<document name="{name}">\n{content}\n</document>' for name, content in docs)
    names = ", ".join(f'"{name}"' for name, _ in docs)
    return f"""Compare the following documents: {names}. Write the entire answer in {LANGUAGES[language]}.

Use exactly this Markdown structure:

## {common}
- (bullet points)

## {diff}
(a Markdown table with one row per aspect and one column per document, followed by short explanations if needed)

## {overall}
(1-3 paragraphs)

Refer to documents by their file names and cite pages like [file name p.N] where possible.

{body}"""
