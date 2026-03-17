"""Task 3: User memory with personalization and streaming chat responses."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time
import uuid
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")
MEMORY_FILE = Path("memory.json")

Message = Dict[str, str]
ChatState = Dict[str, object]
Memory = Dict[str, object]


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


def load_memory() -> Memory:
    if not MEMORY_FILE.exists():
        return {}

    try:
        raw = MEMORY_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(data, dict):
        return data
    return {}


def save_memory(memory: Memory) -> None:
    try:
        MEMORY_FILE.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as err:
        st.error(f"Could not save memory: {err}")


def merge_memory(existing: Memory, extra: Memory) -> Memory:
    merged: Memory = dict(existing)

    for key, value in extra.items():
        if not isinstance(key, str):
            continue

        current = merged.get(key)

        if isinstance(current, list) and isinstance(value, list):
            merged_list = list(current)
            for item in value:
                if item not in merged_list:
                    merged_list.append(item)
            merged[key] = merged_list
            continue

        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_memory(current, value)
            continue

        if value in (None, "", [], {}):
            continue

        if value == current:
            continue

        merged[key] = value

    return merged


def parse_json_payload(text: str) -> Memory:
    text = text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}

    return {}


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


def build_memory_system_prompt(memory: Memory) -> str:
    if not memory:
        return ""

    lines = [
        "Use the known user preferences below to personalize your replies. "
        "Do not invent facts.",
        "Known user details:",
    ]

    for key, value in memory.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            joined = ", ".join(str(item) for item in value)
            lines.append(f"- {key}: {joined}")
        elif isinstance(value, dict):
            try:
                lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
            except TypeError:
                lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def call_model(
    messages: List[Message],
    hf_token: str,
    user_memory: Optional[Memory] = None,
    on_chunk: Optional[callable] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Call Hugging Face with the full conversation history."""
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    request_messages = list(messages)
    memory_prompt = build_memory_system_prompt(user_memory or {})
    if memory_prompt:
        request_messages = [{"role": "system", "content": memory_prompt}] + request_messages

    payload = {
        "model": HF_MODEL,
        "messages": request_messages,
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
                payload_json = json.loads(payload_text)
            except ValueError:
                continue

            choices = payload_json.get("choices")
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
            time.sleep(0.03)
    finally:
        response.close()

    full_reply = "".join(response_chunks).strip()
    if not full_reply:
        return None, "Model response content was empty."

    return full_reply, None


def extract_user_traits(user_message: str, hf_token: str) -> Tuple[Memory, Optional[str]]:
    """Extract personal facts/preferences from a user message."""
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    extraction_prompt = (
        "You are a JSON extractor. Given the user message, return only a JSON object "
        "with personal facts or preferences (for example: name, preferred_language, "
        "interests, communication_style, favorite_topics). If none, return {}. "
        "Do not add explanations."
    )
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 200,
    }

    try:
        response = requests.post(
            HF_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
    except requests.RequestException as err:
        return {}, f"Memory extraction failed: {err}"

    if response.status_code == 401:
        return {}, "Invalid or missing Hugging Face token for memory extraction."
    if response.status_code == 429:
        return {}, "Rate limit reached while extracting memory."
    if response.status_code >= 400:
        return {}, f"Memory extraction API error {response.status_code}: {response.text}"

    try:
        body = response.json()
    except ValueError:
        return {}, "Invalid JSON in memory extraction response."

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}, None

    first_choice = choices[0]
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return {}, None

    traits = parse_json_payload(content)
    return traits, None


def format_chat_label(chat: ChatState) -> str:
    return f"{chat['title']}\n{chat['timestamp']}"


def initialize_memory() -> None:
    if "user_memory" not in st.session_state:
        st.session_state.user_memory = load_memory()  # type: ignore[attr-defined]


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

    with st.sidebar.expander("User Memory", expanded=True):
        memory: Memory = st.session_state.user_memory  # type: ignore[attr-defined]
        if memory:
            st.json(memory)
        else:
            st.info("No memory saved yet.")

        if st.button("Clear memory"):
            st.session_state.user_memory = {}
            save_memory({})
            st.rerun()

    chats: Dict[str, ChatState] = st.session_state.chats  # type: ignore[attr-defined]
    active_id = st.session_state.active_chat_id  # type: ignore[attr-defined]

    if not chats:
        st.sidebar.info("No chats yet.")
        return

    with st.sidebar:
        chat_list = st.container(height=280)

    with chat_list:
        for chat_id, chat in chats.items():
            is_active = chat_id == active_id
            row = st.container(border=True)
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


def update_memory_from_user_message(user_message: str, hf_token: str) -> None:
    memory: Memory = st.session_state.user_memory  # type: ignore[attr-defined]
    extracted_traits, trait_error = extract_user_traits(user_message, hf_token)

    if trait_error:
        st.info(trait_error)
        return

    if not extracted_traits:
        return

    merged = merge_memory(memory, extracted_traits)
    if merged == memory:
        return

    st.session_state.user_memory = merged
    save_memory(merged)


def main() -> None:
    st.set_page_config(page_title="My AI Chat", layout="wide")
    st.title("My AI Chat")

    hf_token = get_hf_token()
    if hf_token is None:
        return

    initialize_memory()
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

        assistant_reply, error = call_model(
            messages,
            hf_token,
            user_memory=st.session_state.user_memory,  # type: ignore[attr-defined]
            on_chunk=on_chunk,
        )

    if error:
        st.error(error)
        return

    messages.append({"role": "assistant", "content": assistant_reply})
    save_chat(active_chat)
    update_memory_from_user_message(user_message, hf_token)
    st.rerun()


if __name__ == "__main__":
    main()
