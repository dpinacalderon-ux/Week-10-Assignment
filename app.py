"""Part D: Chat persistence with sidebar chat management."""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple
import uuid

import requests
import streamlit as st

HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")

Message = Dict[str, str]
ChatState = Dict[str, object]


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def ensure_chats_directory() -> None:
    try:
        CHATS_DIR.mkdir(exist_ok=True)
    except OSError as err:
        st.error(f"Could not create chats directory: {err}")


def build_chat_payload(chat: ChatState) -> str:
    return json.dumps(
        {
            "id": chat["id"],
            "title": chat["title"],
            "timestamp": chat["timestamp"],
            "messages": chat["messages"],
        },
        indent=2,
        ensure_ascii=False,
    )


def load_chat_file(chat_path: Path) -> Optional[ChatState]:
    if not chat_path.exists():
        return None
    try:
        raw = chat_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    chat_id = data.get("id")
    title = data.get("title")
    timestamp = data.get("timestamp")
    messages = data.get("messages")

    if not isinstance(chat_id, str):
        chat_id = chat_path.stem
    if not isinstance(title, str):
        title = "Recovered Chat"
    if not isinstance(timestamp, str):
        timestamp = now_string()
    if not isinstance(messages, list):
        messages = []

    cleaned_messages: List[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        cleaned_messages.append({"role": role, "content": content})

    return {
        "id": chat_id,
        "title": title,
        "timestamp": timestamp,
        "messages": cleaned_messages,
    }


def load_saved_chats() -> Dict[str, ChatState]:
    ensure_chats_directory()
    chats: Dict[str, ChatState] = {}

    for chat_file in CHATS_DIR.glob("*.json"):
        chat = load_chat_file(chat_file)
        if chat is None:
            continue
        chats[chat["id"]] = chat

    return chats


def save_chat(chat: ChatState) -> None:
    ensure_chats_directory()
    chat_id = chat.get("id")
    if not isinstance(chat_id, str):
        return
    try:
        chat_path = CHATS_DIR / f"{chat_id}.json"
        chat_path.write_text(build_chat_payload(chat), encoding="utf-8")
    except OSError as err:
        st.error(f"Could not save chat '{chat_id}': {err}")


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


def call_model(
    messages: List[Message],
    hf_token: str,
    on_chunk: Optional[callable] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Call Hugging Face with streaming and return the full assistant reply."""
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            HF_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            stream=True,
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

    response_chunks: List[str] = []
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue

            payload_text = raw_line.removeprefix("data: ").strip()
            if payload_text == "[DONE]":
                break

            try:
                chunk_payload = json.loads(payload_text)
            except ValueError:
                continue

            choices = chunk_payload.get("choices")
            if not isinstance(choices, list) or not choices:
                continue

            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if not isinstance(delta, dict):
                continue

            chunk = delta.get("content")
            if not isinstance(chunk, str):
                continue

            response_chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)

            # Small delay makes streaming visible even for very fast models.
            time.sleep(0.03)
    finally:
        response.close()

    full_reply = "".join(response_chunks).strip()
    if not full_reply:
        return None, "Model response content was empty."

    return full_reply, None


def format_chat_label(chat: ChatState) -> str:
    return f"{chat['title']}\n{chat['timestamp']}"


def initialize_chats() -> None:
    """Initialize chat storage in session state."""
    if "chats" not in st.session_state:
        st.session_state.chats = load_saved_chats()  # type: ignore[attr-defined]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None  # type: ignore[attr-defined]

    if not st.session_state.chats:  # type: ignore[attr-defined]
        create_new_chat()


def create_new_chat() -> None:
    chat_id = str(uuid.uuid4())
    timestamp = now_string()
    st.session_state.chats[chat_id] = {  # type: ignore[attr-defined]
        "id": chat_id,
        "title": "New Chat",
        "timestamp": timestamp,
        "messages": [],
    }
    save_chat(st.session_state.chats[chat_id])  # type: ignore[attr-defined]
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
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        try:
            chat_file.unlink()
        except OSError as err:
            st.error(f"Could not delete chat file '{chat_id}.json': {err}")

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
        chat["timestamp"] = now_string()
        save_chat(chat)


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
    active_chat["timestamp"] = now_string()
    messages.append({"role": "user", "content": user_message})

    with st.chat_message("assistant"):
        reply_area = st.empty()
        reply_area.markdown("")
        assembled_reply = ""

        def on_chunk(chunk: str) -> None:
            nonlocal assembled_reply
            assembled_reply += chunk
            reply_area.markdown(assembled_reply)

        assistant_reply, error = call_model(messages, hf_token, on_chunk=on_chunk)

    if error:
        st.error(error)
        reply_area.empty()
        return

    messages.append({"role": "assistant", "content": assistant_reply})
    save_chat(active_chat)
    st.rerun()


if __name__ == "__main__":
    main()
