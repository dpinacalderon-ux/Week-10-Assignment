"""Part A: Core Chat Application setup with Hugging Face API test call."""

from __future__ import annotations

from typing import Optional, Tuple

import requests
import streamlit as st

HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TEST_PROMPT = "Hello!"


def get_hf_token() -> Optional[str]:
    """Read HF token from Streamlit secrets safely."""
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error(
            "Missing Hugging Face token. Add `.streamlit/secrets.toml` with `HF_TOKEN = \"your_token_here\"`."
        )
        return None

    if not isinstance(hf_token, str) or not hf_token.strip():
        st.error(
            "HF token is empty. Add a valid token in `.streamlit/secrets.toml` under `HF_TOKEN`."
        )
        return None

    return hf_token.strip()


def ask_test_message(hf_token: str) -> Tuple[Optional[str], Optional[str]]:
    """Send hardcoded test message and return (response_text, error_message)."""
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            HF_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
    except requests.RequestException as err:
        return None, f"Network error while calling the API: {err}"

    if response.status_code == 401:
        return None, "Invalid or missing Hugging Face token. Please check `HF_TOKEN`."
    if response.status_code == 429:
        return None, "Rate limit reached. Try again in a little while."
    if response.status_code >= 400:
        return None, f"API request failed with status {response.status_code}: {response.text}"

    try:
        result = response.json()
    except ValueError:
        return None, "Invalid JSON response from API."

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "API response did not include any choices."

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "API response was missing message content."

    return content.strip(), None


def main() -> None:
    st.set_page_config(page_title="My AI Chat", layout="wide")
    st.title("My AI Chat")
    st.markdown("Send a test request to verify API setup.")

    hf_token = get_hf_token()
    if hf_token is None:
        return

    # Run once per user session so Streamlit reruns do not repeat the call automatically.
    if "test_reply" not in st.session_state:
        reply, error = ask_test_message(hf_token)
        st.session_state["test_reply"] = reply
        st.session_state["test_error"] = error

    if st.session_state.get("test_error"):
        st.error(st.session_state["test_error"])
        return

    st.subheader("Model reply")
    st.write(st.session_state.get("test_reply", "No response returned."))


if __name__ == "__main__":
    main()
