"""Basic Streamlit To-Do app setup with safe todos.json and HF API wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


TODOS_FILE = Path("todos.json")
HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_todos() -> List[Dict[str, Any]]:
    """Load todos from file safely."""
    if not TODOS_FILE.exists():
        return []

    try:
        raw_content = TODOS_FILE.read_text(encoding="utf-8").strip()
        if not raw_content:
            return []
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        st.warning(
            "todos.json is not valid JSON. "
            "Using an empty list until this file is fixed."
        )
        return []
    except OSError as err:
        st.error(f"Could not read todos.json: {err}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("todos"), list):
        return data["todos"]

    st.warning("todos.json has an unexpected format. Using an empty list.")
    return []


def save_todos(todos: List[Dict[str, Any]]) -> None:
    """Save todos to file safely."""
    try:
        TODOS_FILE.write_text(
            json.dumps({"todos": todos}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as err:
        st.error(f"Could not write todos.json: {err}")


def get_hf_token() -> Optional[str]:
    """Read Hugging Face token from Streamlit secrets."""
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error(
            "Missing HF token. Please add `HF_TOKEN` in `.streamlit/secrets.toml` "
            "under your project root."
        )
        return None

    if not isinstance(hf_token, str) or not hf_token.strip():
        st.error(
            "`HF_TOKEN` is empty. Update `.streamlit/secrets.toml` to include "
            "a valid token before using AI features."
        )
        return None

    return hf_token.strip()


def ask_model(hf_token: str, prompt: str) -> Optional[str]:
    """Send a prompt to Hugging Face chat completions endpoint."""
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            HF_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as err:
        st.error(f"Could not connect to Hugging Face API: {err}")
        return None

    try:
        response_json = response.json()
        choices = response_json.get("choices", [])
        if not choices:
            return None
        return choices[0]["message"]["content"]
    except (ValueError, KeyError, TypeError) as err:
        st.error(f"Unexpected response format from API: {err}")
        return None


def main() -> None:
    """Render a minimal Streamlit page and test connection to the AI endpoint."""
    st.set_page_config(page_title="Streamlit To-Do App", page_icon="🗂️")
    st.title("Streamlit To-Do App")

    st.markdown("### Setup placeholder")
    st.write("Your app is running. Use this section to add UI components later.")

    todos = load_todos()
    save_todos(todos)

    with st.container():
        st.subheader("Saved to-dos")
        if todos:
            st.json(todos)
        else:
            st.info("No to-dos found yet. Your `todos.json` file is initialized as empty.")

        st.caption(f"Total items: {len(todos)}")

    st.divider()
    st.subheader("AI Connection Setup")

    hf_token = get_hf_token()
    prompt = st.text_input("Try a test message", value="Hello!")
    if st.button("Send test request"):
        if hf_token is None:
            st.stop()

        with st.spinner("Calling Hugging Face Inference Router..."):
            reply = ask_model(hf_token, prompt)

        if reply is None:
            st.warning("No response text was returned from the model.")
        else:
            st.success("Model response:")
            st.write(reply)


if __name__ == "__main__":
    main()
