"""Part C: Chat management with sidebar-based multi-chat support."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

import requests
import streamlit as st

HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

Message = Dict[str, str]
ChatState = Dict[str, object]


def get_hf_token() -> Optional[str]:
    """Read HF token from Streamlit secrets safely."""
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error(
            "Missing Hugging Face token. Add `.streamlit/secrets.toml` with "
            "`HF_TOKEN = \"your_token_here\"`."
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
    """Call Hugging Face with the full conversation history."""
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


def format_chat_label(chat: ChatState) -> str:
    return f"{chat['title']}\n{chat['timestamp']}"


def initialize_chats() -> None:
    """Initialize chat storage in session state."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}  # type: ignore[attr-defined]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None  # type: ignore[attr-defined]

    if not st.session_state.chats:  # type: ignore[attr-defined]
        create_new_chat()


def create_new_chat() -> None:
    chat_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chats[chat_id] = {  # type: ignore[attr-defined]
        "id": chat_id,
        "title": "New Chat",
        "timestamp": timestamp,
        "messages": [],
    }
    st.session_state.active_chat_id = chat_id  # type: ignore[attr-defined]


def current_chat() -> Tuple[str, ChatState]:
    active_id = st.session_state.active_chat_id  # type: ignore[attr-defined]
    if not active_id or active_id not in st.session_state.chats:  # type: ignore[attr-defined]
        create_new_chat()
        active_id = st.session_state.active_chat_id  # type: ignore[attr-defined]

    chat = st.session_state.chats[active_id]  # type: ignore[attr-defined]
    return active_id, chat


def delete_chat(chat_id: str) -> None:
    """Delete a chat and switch active chat safely."""
    chats: Dict[str, ChatState] = st.session_state.chats  # type: ignore[attr-defined]
    if chat_id not in chats:
        return

    was_active = chat_id == st.session_state.active_chat_id
    del chats[chat_id]

    if not chats:
        create_new_chat()
        return

    if was_active:
        # Keep the most recent remaining chat as active.
        latest_chat = max(chats.items(), key=lambda item: item[1]["timestamp"])
        st.session_state.active_chat_id = latest_chat[0]  # type: ignore[attr-defined]


def rename_if_needed(chat: ChatState, user_message: str) -> None:
    """Set chat title from first user message if this is a new chat."""
    if chat["title"] == "New Chat":
        title = user_message.strip()
        chat["title"] = title[:35] + "..." if len(title) > 35 else title


def render_sidebar() -> None:
    st.sidebar.title("Chats")

    if st.button("New Chat"):
        create_new_chat()
        st.rerun()

    chats: Dict[str, ChatState] = st.session_state.chats  # type: ignore[attr-defined]
    active_id = st.session_state.active_chat_id  # type: ignore[attr-defined]

    if not chats:
        st.sidebar.info("No chats yet.")
        return

    for chat_id, chat in chats.items():
        is_active = chat_id == active_id
        with st.sidebar:
            row = st.container(border=True)
        with row:
            cols = st.columns([0.84, 0.16])
            with cols[0]:
                if st.button(
                    format_chat_label(chat),
                    key=f"switch_{chat_id}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.active_chat_id = chat_id  # type: ignore[attr-defined]
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"delete_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()


def render_messages(messages: List[Message]) -> None:
    chat_area = st.container(height=520)
    with chat_area:
        if not messages:
            st.info("Start a conversation with the input at the bottom.")

        for message in messages:
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

    initialize_chats()
    render_sidebar()

    _, active_chat = current_chat()
    messages: List[Message] = active_chat["messages"]  # type: ignore[assignment]

    render_messages(messages)

    user_message = st.chat_input("Type a message")
    if not user_message:
        return

    user_message = user_message.strip()
    if not user_message:
        return

    rename_if_needed(active_chat, user_message)
    messages.append({"role": "user", "content": user_message})

    with st.spinner("Thinking..."):
        assistant_reply, error = call_model(messages, hf_token)

    if error:
        st.error(error)
        return

    messages.append({"role": "assistant", "content": assistant_reply})
    st.rerun()


if __name__ == "__main__":
    main()
