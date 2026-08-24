"""The week strip's markup, because it is the only place the plan and what
actually happened are shown side by side."""

from __future__ import annotations

import app.ui as ui


def _render(days, **kw) -> str:
    """Collect what week_strip writes, without a Streamlit runtime."""
    written: list[str] = []
    original = ui.st.markdown
    ui.st.markdown = lambda body, **_: written.append(body)
    try:
        ui.week_strip(days, **kw)
    finally:
        ui.st.markdown = original
    return "\n".join(written)


def _day(name="Mon", items=None, **kw):
    return {"name": name, "date": "24-08", "today": False,
            "items": items or [], **kw}


def test_a_completed_session_is_ticked():
    html = _render([_day(items=[{"sport": "strength", "minutes": 24,
                                "done": True}])])
    assert "✓" in html
    assert "ic-item done" in html
    assert "settled" in html          # the whole day is accounted for
    assert "✓ logged" in html         # and the key explains the mark


def test_a_planned_session_carries_no_mark():
    html = _render([_day(items=[{"sport": "bike", "minutes": 40}])])
    assert "✓" not in html
    assert "ic-item done" not in html
    assert "logged" not in html       # nothing to explain


def test_a_day_that_went_by_unlogged_is_flagged():
    """The question every morning is whether the week is on track, and that is a
    question about what did not happen as much as what did."""
    html = _render([_day(items=[{"sport": "run", "minutes": 40,
                                "missed": True}])])
    assert "ic-item missed" in html
    assert "slipped" in html
    assert "planned, not logged" in html


def test_a_rest_day_is_not_a_missed_day():
    html = _render([_day(items=[])])
    assert "missed" not in html
    assert "ic-day rest" in html


def test_a_day_with_one_done_and_one_left_is_not_settled():
    html = _render([_day(items=[{"sport": "strength", "minutes": 20,
                                 "done": True},
                                {"sport": "bike", "minutes": 40}])])
    assert "settled" not in html
    assert "ic-item done" in html


def test_the_key_can_be_suppressed():
    """A future week has nothing logged, so the legend would be noise."""
    html = _render([_day(items=[{"sport": "bike", "minutes": 40, "done": True}])],
                   key=False)
    assert "✓" in html
    assert "logged" not in html


def test_everything_interpolated_is_escaped():
    """Activity names come from Garmin and notes from an AI."""
    html = _render([_day(items=[{"sport": "<script>", "minutes": 10,
                                 "zone": 'Z2" onload=x'}])])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
