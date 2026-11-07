"""분석 결과(Markdown)를 보기 좋게 렌더링하는 도우미."""

import re

import streamlit as st

from src import prompts

# [파일명.pdf p.3], [파일명.pdf p.3-4], [파일명.pdf p.1, p.2]
_DOC_CITATION = re.compile(r"\[([^\[\]\n]+?\.pdf)\s+(p\.\s?[\d\s,\-–p.]*\d)\]", re.IGNORECASE)
# (p.3), (p.3-4)
_PAGE_CITATION = re.compile(r"\((p\.\s?\d+(?:\s*[-–,]\s*(?:p\.\s?)?\d+)*)\)")
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def format_citations(text: str) -> str:
    """출처 표기를 Streamlit 배지로 바꿔 눈에 띄게 한다."""
    text = _DOC_CITATION.sub(lambda m: f":violet-badge[📄 {m.group(1)} {m.group(2)}]", text)
    return _PAGE_CITATION.sub(lambda m: f":violet-badge[{m.group(1)}]", text)


def parse_sections(text: str) -> dict[str, str]:
    """'## 제목' 단위로 나눠 {제목: 본문}을 반환한다."""
    parts = _HEADING.split(text)
    # parts = [머리말, 제목1, 본문1, 제목2, 본문2, ...]
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def split_keywords(text: str) -> list[str]:
    items = re.split(r"[,，、\n]", text)
    cleaned = [re.sub(r"^[\s\-*#•\d.]+|[\[\]]", "", k).strip() for k in items]
    return [k for k in cleaned if k]


def _smaller_headings(text: str) -> str:
    return re.sub(r"^#{1,3}\s", "#### ", text, flags=re.MULTILINE)


def render_summary(text: str, language: str) -> None:
    one, detail, points, keywords = prompts.SUMMARY_HEADINGS[language]
    sections = parse_sections(text)
    if not all(h in sections for h in (one, detail, points, keywords)):
        st.markdown(format_citations(_smaller_headings(text)))
        return

    st.info(f"**{one}** · {format_citations(sections[one])}", icon="💡")
    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        st.markdown(f"##### 📖 {detail}")
        st.markdown(format_citations(sections[detail]))
    with right.container(border=True):
        st.markdown(f"##### ✅ {points}")
        st.markdown(format_citations(sections[points]))
    tags = " ".join(f":blue-badge[{k}]" for k in split_keywords(sections[keywords]))
    st.markdown(f"**🏷️ {keywords}**&nbsp;&nbsp;{tags}")


def render_comparison(text: str, language: str) -> None:
    icons = ("🤝", "🔀", "🧭")
    headings = prompts.COMPARE_HEADINGS[language]
    sections = parse_sections(text)
    if not all(h in sections for h in headings):
        st.markdown(format_citations(_smaller_headings(text)))
        return
    for icon, heading in zip(icons, headings):
        with st.container(border=True):
            st.markdown(f"##### {icon} {heading}")
            st.markdown(format_citations(sections[heading]))


def render_answer(text: str) -> None:
    st.markdown(format_citations(_smaller_headings(text)))
