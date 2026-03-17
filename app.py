"""Basic Streamlit To-Do app setup with safe todos.json handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st


TODOS_FILE = Path("todos.json")


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
            json.dumps({"todos": todos}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as err:
        st.error(f"Could not write todos.json: {err}")


def main() -> None:
    """Render a minimal Streamlit page."""
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

        st.caption("Total items: " + str(len(todos)))


if __name__ == "__main__":
    main()
