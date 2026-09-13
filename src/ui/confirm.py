"""Shared Streamlit confirm-before-destroy widget."""

from __future__ import annotations

from typing import Callable

import streamlit as st


def confirm_destructive(
    trigger_label: str,
    warning: str,
    *,
    key: str,
    on_confirm: Callable[[], None],
    trigger_kwargs: dict | None = None,
    confirm_container=None,
) -> bool:
    """Render a destructive-action button with a two-step confirmation.

    Call once per unique action; ``key`` must be stable across reruns.
    Returns ``True`` when ``on_confirm`` ran this rerun.

    Parameters
    ----------
    confirm_container:
        Optional Streamlit container (e.g. ``st.container()``) where the
        confirmation warning and Confirm/Cancel buttons are rendered.  Pass
        a container that was created *outside* a narrow column to prevent
        the button text from wrapping character-by-character.  Defaults to
        the current rendering context (``st``).
    """
    flag = f"_confirm_{key}"
    if st.button(trigger_label, key=f"{key}_trigger", **(trigger_kwargs or {})):
        st.session_state[flag] = True

    if not st.session_state.get(flag):
        return False

    # Render the confirmation UI in the provided container so it is never
    # squeezed inside a narrow column.
    ctx = confirm_container if confirm_container is not None else st
    ctx.warning(warning)
    col_ok, col_cancel, *_ = ctx.columns([1, 1, 4])
    fired = False
    if col_ok.button("Confirm", key=f"{key}_ok", type="primary"):
        on_confirm()
        fired = True
    if col_cancel.button("Cancel", key=f"{key}_cancel"):
        st.session_state.pop(flag, None)
        st.rerun()
    if fired:
        st.session_state.pop(flag, None)
    return fired
