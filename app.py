"""Part B: Multi-turn chat UI with Streamlit native components."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

Message = Dict[str, str]


def get_hf_token() -> Optional[str]:
    """Read HF token from Streamlit secrets safely."""
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error(
            "Missing Hugging Face token. Add `.streamlit/secrets.toml` "
            "with `HF_TOKEN = \"your_token_here\"`."
        )
        return None

    if not isinstance(hf_token, str) or not hf_token.strip():
        st.error(
            "HF token is empty. Set a valid token in `.streamlit/secrets.toml` "
            "before using the chat."
        )
        return None

    return hf_token.strip()


def call_model(messages: List[Message], hf_token: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Hugging Face with full conversation history."""
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            HF_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
    except requests.RequestException as err:
        return None, f"Network error: {err}"

    if response.status_code == 401:
        return None, "Invalid or missing Hugging Face token."
    if response.status_code == 429:
        return None, "Rate limit reached. Please try again in a moment."
    if response.status_code >= 400:
        return None, f"API error {response.status_code}: {response.text}"

    try:
        body = response.json()
    except ValueError:
        return None, "Invalid JSON response from API."

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "Model response had no choices."

    first_choice = choices[0]
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None

    if not isinstance(content, str) or not content.strip():
        return None, "Model response content was empty."

    return content.strip(), None


def ensure_session_state() -> None:
    """Initialize conversation state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # type: ignore[attr-defined]


def render_conversation() -> None:
    """Render full message history above the fixed input bar."""
    chat_area = st.container(height=540)
    with chat_area:
        if not st.session_state.messages:  # type: ignore[attr-defined]
            st.info("Ask a question below to start the conversation.")

        for message in st.session_state.messages:  # type: ignore[attr-defined]
            role = message.get("role", "user")
            content = message.get("content", "")
            with st.chat_message(role):
                st.write(content)


def main() -> None:
    st.set_page_config(page_title="My AI Chat", layout="wide")
    st.title("My AI Chat")

    hf_token = get_hf_token()
    if hf_token is None:
        return

    ensure_session_state()
    render_conversation()

    user_message = st.chat_input("Type a message")
    if user_message is None:
        return

    user_message = user_message.strip()
    if not user_message:
        return

    st.session_state.messages.append({"role": "user", "content": user_message})  # type: ignore[attr-defined]

    with st.spinner("Thinking..."):
        assistant_reply, error = call_model(st.session_state.messages, hf_token)  # type: ignore[attr-defined]

    if error:
        st.error(error)
        return

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})  # type: ignore[attr-defined]

    st.rerun()


if __name__ == "__main__":
    main()
