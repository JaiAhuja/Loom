"""Configure and run Loom's Streamlit multipage application."""

from pathlib import Path

import streamlit as st


pages_dir = Path(__file__).parent / "pages"
navigation = st.navigation(
    [
        st.Page(pages_dir / "1_Chat.py", title="Chat", icon="🪡"),
        st.Page(
            pages_dir / "2_Knowledge_Graph.py",
            title="Knowledge Graph",
            icon="🕸️",
        ),
        st.Page(pages_dir / "3_Chat_History.py", title="Chat History", icon="📚"),
    ]
)
navigation.run()
