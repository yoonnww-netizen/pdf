"""API 키와 모델 설정 로딩.

우선순위: Streamlit secrets (Cloud) → 환경변수 (.env, 로컬).
"""

import os

import streamlit as st
from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-6.1-terra"

load_dotenv()


def _get_setting(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 난다 (로컬 실행 시 정상 상황).
        value = None
    if not value:
        value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def get_api_key() -> str | None:
    return _get_setting("OPENAI_API_KEY")


def get_model() -> str:
    return _get_setting("OPENAI_MODEL") or DEFAULT_MODEL
