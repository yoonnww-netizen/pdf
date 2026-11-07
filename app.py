from collections.abc import Callable, Iterator
from pathlib import Path

import streamlit as st

from src import llm, prompts, ui
from src.config import get_api_key, get_model
from src.pdf_utils import PdfDocument, PdfExtractionError, file_hash, load_document

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stMainBlockContainer { max-width: 1100px; padding-top: 2.5rem; }
    [data-testid="stSidebarContent"] [data-testid="stCaptionContainer"] p { line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "이 문서의 핵심 내용은 무엇인가요?",
    "중요한 수치나 날짜를 정리해 주세요.",
    "결론이나 권고 사항은 무엇인가요?",
]

state = st.session_state
state.setdefault("summaries", {})  # (file_hash, language) -> markdown
state.setdefault("comparisons", {})  # (sorted file_hashes, language) -> markdown
state.setdefault("messages", [])  # [{"role": "user"|"assistant", "content": str}]
state.setdefault("uploader_key", 0)  # 바꾸면 업로더가 비워진다 (전체 초기화)

model = get_model()
api_key = get_api_key()


@st.cache_resource(show_spinner=False)
def _client(key: str):
    return llm.get_client(key)


@st.cache_data(show_spinner=False, max_entries=50)
def _load(digest: str, name: str, _data: bytes) -> tuple[PdfDocument | None, str | None]:
    # digest(파일 해시)와 name으로 캐시한다. _data는 해시 계산에서 제외된다.
    try:
        return load_document(name, _data), None
    except PdfExtractionError as e:
        return None, str(e)


def _reset_all() -> None:
    state.summaries = {}
    state.comparisons = {}
    state.messages = []
    state.uploader_key += 1


# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 PDF 분석기")
    st.caption("PDF를 올리면 AI가 요약하고, 질문에 답하고, 여러 문서를 비교해 드립니다.")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{state.uploader_key}",
        help="여러 파일을 한 번에 올릴 수 있습니다. 파일은 서버에 저장되지 않습니다.",
    )
    doc_list = st.container()
    st.divider()
    st.markdown("**⚙️ 설정**")
    language = st.radio("결과 언어", list(prompts.LANGUAGES), horizontal=True)
    st.caption(f"사용 모델: `{model}`")
    if llm.is_fake_mode():
        st.caption("🧪 테스트 모드: API를 호출하지 않고 가짜 응답을 사용합니다.")
    st.button(
        "🔄 전체 초기화", on_click=_reset_all, width="stretch",
        help="업로드한 파일, 분석 결과, 대화 기록을 모두 지웁니다.",
    )

# ── 헤더 ────────────────────────────────────────────────────────
st.title("📄 PDF 분석기")
st.caption("업로드한 PDF를 근거로 요약 · 질의응답 · 비교 분석을 합니다. 답변에는 출처 페이지가 표시됩니다.")

if not api_key:
    st.error(
        "**OpenAI API 키가 설정되지 않았습니다.**\n\n"
        "- 로컬 실행: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...`를 추가한 뒤 앱을 다시 실행하세요.\n"
        "- Streamlit Cloud: App settings → **Secrets**에 `OPENAI_API_KEY = \"sk-...\"`를 추가하세요.",
        icon="🔑",
    )
    st.stop()

client = _client(api_key)

if not uploaded_files:
    st.markdown("#### 이렇게 사용하세요")
    steps = [
        ("📤", "1. PDF 업로드", "사이드바에서 PDF 파일을 올리세요. 여러 개도 됩니다. (모바일은 왼쪽 위 » 버튼)"),
        ("🧭", "2. 분석 방식 선택", "요약, 질의응답, 문서 비교 중 원하는 탭을 고르세요."),
        ("📥", "3. 결과 확인", "출처 페이지와 함께 결과를 확인하고 Markdown 파일로 내려받으세요."),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), steps):
        with col.container(border=True, height="stretch"):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.caption("※ 스캔한 이미지 PDF(텍스트가 없는 PDF)와 암호가 걸린 PDF는 분석할 수 없습니다.")
    st.stop()

# ── 업로드 파일 처리 ─────────────────────────────────────────────
all_docs: list[PdfDocument] = []
failures: list[str] = []
notices: list[str] = []
sidebar_rows: list[str] = []
seen: set[str] = set()
with st.spinner("PDF에서 텍스트를 추출하는 중..."):
    for f in uploaded_files:
        data = f.getvalue()
        digest = file_hash(data)
        if digest in seen:
            notices.append(f"**{f.name}**: 이미 올린 파일과 내용이 같아 제외했습니다.")
            sidebar_rows.append(f"➖ {f.name} · 중복")
            continue
        seen.add(digest)
        doc, error = _load(digest, f.name, data)
        if error:
            failures.append(f"**{f.name}**: {error}")
            sidebar_rows.append(f"❌ {f.name} · 읽기 실패")
        elif doc.is_scanned:
            all_docs.append(doc)
            notices.append(
                f"**{doc.name}**: 추출된 텍스트가 거의 없어 분석에서 제외했습니다 "
                "(스캔한 이미지 PDF는 OCR을 지원하지 않습니다)."
            )
            sidebar_rows.append(f"⚠️ {doc.name} · 텍스트 없음")
        else:
            all_docs.append(doc)
            sidebar_rows.append(f"✅ {doc.name} · {doc.page_count}쪽")

with doc_list:
    st.caption(f"업로드된 문서 {len(sidebar_rows)}개")
    st.markdown("\n".join(f"- {row}" for row in sidebar_rows))

if failures:
    st.error("다음 파일은 읽을 수 없습니다.\n\n" + "\n".join(f"- {m}" for m in failures), icon="🚫")
if notices:
    st.warning("\n".join(f"- {m}" for m in notices), icon="⚠️")
if not all_docs:
    st.info("분석할 수 있는 문서가 없습니다. 다른 PDF 파일을 업로드해 주세요.")
    st.stop()

docs = [d for d in all_docs if not d.is_scanned]  # 분석 가능한 문서
by_hash = {d.file_hash: d for d in docs}
docs_key = "-".join(by_hash)  # 문서 구성이 바뀌면 선택 위젯을 초기화하기 위한 키


def _name(digest: str) -> str:
    return by_hash[digest].name


def _stem(name: str) -> str:
    return Path(name).stem


def _stream_with_status(
    make_stream: Callable[[Callable[[str], None]], Iterator[str]], label: str
) -> str | None:
    """진행 상태를 보여주며 스트리밍 출력을 실시간 표시하고, 끝나면 둘 다 지운 뒤 전체 텍스트를 반환한다.

    실패하면 오류를 표시하고 None을 반환한다.
    """
    status_slot, output_slot = st.empty(), st.empty()
    status = status_slot.status(label, state="running")
    try:
        with output_slot.container():
            return st.write_stream(make_stream(lambda message: status.update(label=message)))
    except llm.LLMError as e:
        st.error(str(e), icon="🚫")
        return None
    finally:
        status_slot.empty()
        output_slot.empty()


def _no_docs_message(what: str) -> None:
    st.info(f"{what} 수 있는 문서가 없습니다. 텍스트가 있는 PDF를 업로드하세요.")


tab_info, tab_summary, tab_qa, tab_compare = st.tabs(
    ["📋 문서 정보", "📝 요약", "💬 질의응답", "⚖️ 비교"]
)

# ── 문서 정보 ───────────────────────────────────────────────────
with tab_info:
    m1, m2, m3 = st.columns(3)
    m1.metric("문서 수", f"{len(all_docs)}개", border=True)
    m2.metric("총 페이지", f"{sum(d.page_count for d in all_docs):,}쪽", border=True)
    m3.metric("총 글자 수", f"{sum(d.char_count for d in all_docs):,}자", border=True)

    for doc in all_docs:
        badge = " · ⚠️ 텍스트 없음" if doc.is_scanned else ""
        with st.expander(
            f"📄 {doc.name} — {doc.page_count}쪽 · {doc.char_count:,}자{badge}",
            expanded=len(all_docs) == 1,
        ):
            page = st.number_input(
                "미리볼 페이지", min_value=1, max_value=doc.page_count, value=1,
                key=f"preview_{doc.file_hash}",
            )
            with st.container(height=320, border=True):
                st.text(doc.pages[page - 1] or "(이 페이지에서 추출된 텍스트가 없습니다)")

# ── 요약 ────────────────────────────────────────────────────────
with tab_summary:
    if not docs:
        _no_docs_message("요약할")
    else:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        digest = c1.selectbox(
            "요약할 문서", list(by_hash), format_func=_name, key=f"summary_doc_{docs_key}"
        )
        doc = by_hash[digest]
        cache_key = (digest, language)
        done = cache_key in state.summaries
        run = c2.button(
            "🔁 다시 요약" if done else "✨ 요약하기", type="primary", width="stretch",
            key="summary_run",
        )
        if run:
            text = _stream_with_status(
                lambda progress: llm.summarize(client, model, doc, language, progress),
                f"'{doc.name}' 요약을 작성하는 중...",
            )
            if text:
                state.summaries[cache_key] = text
                st.rerun()  # 버튼 문구를 '다시 요약'으로 바꾸고 결과를 정리된 형태로 표시

        if cache_key in state.summaries:
            ui.render_summary(state.summaries[cache_key], language)
            st.download_button(
                "📥 요약 결과 다운로드 (.md)",
                f"# {doc.name} 요약\n\n{state.summaries[cache_key]}\n",
                file_name=f"{_stem(doc.name)}_요약.md",
                mime="text/markdown",
                on_click="ignore",
            )
        elif not run:
            st.caption(f"**{doc.name}** ({doc.page_count}쪽)의 한 줄 요약, 상세 요약, 핵심 포인트, 키워드를 만듭니다.")

# ── 질의응답 ────────────────────────────────────────────────────
with tab_qa:
    if not docs:
        _no_docs_message("질문할")
    else:
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        selected = c1.multiselect(
            "질문 대상 문서", list(by_hash), default=list(by_hash), format_func=_name,
            key=f"qa_docs_{docs_key}",
        )
        if c2.button("🗑️ 대화 지우기", key="qa_clear", disabled=not state.messages, width="stretch"):
            state.messages = []
            st.rerun()

        # 새 메시지도 입력창 위에 쌓이도록 대화 영역을 입력창보다 먼저 만든다.
        conversation = st.container()
        with conversation:
            for m in state.messages:
                with st.chat_message(m["role"]):
                    if m["role"] == "assistant":
                        ui.render_answer(m["content"])
                    else:
                        st.markdown(m["content"])

        pending = None
        examples = conversation.empty()
        if not state.messages:
            with examples.container():
                st.caption("💡 예시 질문을 눌러 시작해 보세요")
                for col, q in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
                    if col.button(q, key=f"example_{q}", width="stretch"):
                        pending = q

        question = st.chat_input("문서 내용에 대해 질문하세요") or pending
        if question:
            examples.empty()
            if not selected:
                conversation.warning("질문 대상 문서를 하나 이상 선택하세요.")
            else:
                history = list(state.messages)
                with conversation.chat_message("user"):
                    st.markdown(question)
                with conversation.chat_message("assistant"):
                    reply = _stream_with_status(
                        lambda _progress: llm.answer(
                            client, model, [by_hash[h] for h in selected],
                            question, history, language,
                        ),
                        "문서에서 답을 찾는 중...",
                    )
                    if reply:
                        ui.render_answer(reply)
                        state.messages += [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": reply},
                        ]
                if reply:
                    st.rerun()  # '대화 지우기' 버튼 활성화, 다운로드 버튼 표시

        if state.messages:
            transcript = "\n\n".join(
                f"**{'질문' if m['role'] == 'user' else '답변'}**\n\n{m['content']}"
                for m in state.messages
            )
            st.download_button(
                "📥 대화 기록 다운로드 (.md)",
                f"# 질의응답 기록\n\n{transcript}\n",
                file_name="질의응답_기록.md",
                mime="text/markdown",
                on_click="ignore",
            )

# ── 비교 ────────────────────────────────────────────────────────
with tab_compare:
    if len(docs) < 2:
        st.info("문서를 비교하려면 텍스트가 있는 PDF를 **2개 이상** 업로드하세요.", icon="⚖️")
    else:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        chosen = c1.multiselect(
            "비교할 문서 (2개 이상)", list(by_hash), default=list(by_hash), format_func=_name,
            key=f"compare_docs_{docs_key}",
        )
        cache_key = (tuple(sorted(chosen)), language)
        done = cache_key in state.comparisons
        run = c2.button(
            "🔁 다시 비교" if done else "⚖️ 비교하기", type="primary", width="stretch",
            disabled=len(chosen) < 2, key="compare_run",
        )
        if len(chosen) < 2:
            st.caption("문서를 2개 이상 선택해야 비교할 수 있습니다.")
        if run:
            text = _stream_with_status(
                lambda progress: llm.compare(
                    client, model, [by_hash[h] for h in chosen], language, progress
                ),
                "문서를 비교하는 중...",
            )
            if text:
                state.comparisons[cache_key] = text
                st.rerun()

        if cache_key in state.comparisons:
            ui.render_comparison(state.comparisons[cache_key], language)
            st.download_button(
                "📥 비교 결과 다운로드 (.md)",
                f"# 문서 비교: {', '.join(_name(h) for h in chosen)}\n\n{state.comparisons[cache_key]}\n",
                file_name="문서_비교.md",
                mime="text/markdown",
                on_click="ignore",
            )
