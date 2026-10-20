"""Fast smoke checks for Streamlit entry points without starting a server."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_all_streamlit_pages_parse_and_configure_page():
    for path in sorted((ROOT / "pages").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "st"
            and call.func.attr == "set_page_config"
            for call in calls
        ), f"{path} is missing st.set_page_config()"


def test_app_registers_every_page():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    for page_name in ("1_Chat.py", "2_Knowledge_Graph.py", "3_Chat_History.py", "4_Notebook.py"):
        assert page_name in text
