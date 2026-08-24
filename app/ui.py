"""Presentation primitives for the dashboard.

Streamlit's own widgets are fine for input but weak for display: `st.metric`
renders a fixed shape with a green delta arrow whether or not there is a delta,
and its sparkline gets clipped when a card is given a height. So the read-only
parts of this dashboard are hand-built here instead — a small design system of
cards, banners and bars, with one typographic scale and one colour vocabulary.

Everything interpolated into HTML goes through `html.escape`. That matters: the
dashboard renders text that came from an AI, from Garmin activity names, and from
the athlete's own notes.
"""

from __future__ import annotations

import html
from contextlib import contextmanager
from datetime import date, timedelta
from itertools import count
from typing import Any, Iterable, Literal

import streamlit as st

Tone = Literal["good", "caution", "bad", "neutral"]

# One colour vocabulary, used for meaning only. Nothing is coloured decoratively:
# green means "this is going well", amber "watch this", red "act on this".
TONE_COLOR: dict[str, str] = {
    "good": "#3FB68B",
    "caution": "#E0A33E",
    "bad": "#DB5F5A",
    "neutral": "#8C9AA8",
}
SPORT_COLOR = {
    "swim": "#4C9BE8", "bike": "#F2A33C", "run": "#5FBF8F",
    "strength": "#A98BD9", "brick": "#E8776F", "rest": "#7C8B9A",
    "other": "#9AA7B4",
}
# Cool to hot, so easy zones read calm and hard zones read loud.
ZONE_COLOR = {1: "#7FB6DC", 2: "#3FB68B", 3: "#D8BC55", 4: "#E28A4E", 5: "#DB5F5A"}
# Restarts every rerun, which is what makes the keys stable run to run.
_FRAME_SEQ = count()

SPORT_EMOJI = {"swim": "🏊", "bike": "🚲", "run": "🏃", "strength": "🦵",
               "brick": "🚲🏃", "rest": "😴", "other": "•"}

CSS = """
<style>
  :root {
    /* The page background, needed explicitly: a sticky bar has to paint over
       what scrolls under it, and transparent inherits nothing useful. Streamlit
       exposes no CSS variable for its own theme background, so these are its
       documented light and dark defaults, matched below by media query. */
    --ic-page: #ffffff;
    --ic-good: #3FB68B; --ic-caution: #E0A33E; --ic-bad: #DB5F5A;
    --ic-line: rgba(140,158,176,.28);
    --ic-surface: rgba(140,158,176,.09);
    --ic-surface-2: rgba(140,158,176,.045);
  }
  /* Enough headroom for the wordmark. Streamlit's default is larger still; this
     trims it without cutting into the first element. */
  /* No max-width. layout="wide" already spans the window; capping it here was
     throwing away most of a wide screen and forcing the header controls to wrap
     into a column that had plenty of room beside it. */
  /* The title sits below Streamlit's header, not inside it. Sharing that band
     saved 40px and cost two collisions — the sidebar chevron on the left and
     Deploy on the right — which is a bad trade for a line of chrome that only
     appears on a hosted app. */
  .block-container { padding-top: 1rem; padding-bottom: 3rem;
                     max-width: none; padding-left: 2.2rem; padding-right: 2.2rem; }

  /* Every injected block owns its own vertical space. Streamlit's container gap
     is deliberately NOT overridden here: doing so removed the spacing between
     stacked elements and made custom divs collide. */
  .ic-stat, .ic-banner, .ic-today, .ic-week, .ic-bar, .ic-legend,
  .ic-section, .ic-section-note, .ic-sub { box-sizing: border-box; }

  h1 { font-size: 1.75rem !important; font-weight: 680; letter-spacing: -.02em;
       margin: 0 0 .2rem !important; }
  .ic-sub { font-size: .86rem; opacity: .6; margin: 0 0 .6rem; }
  .ic-section { font-size: .95rem; font-weight: 620; letter-spacing: -.005em;
                padding-top: .55rem; margin: 0 0 .15rem; }
  .ic-section-note { font-size: .81rem; opacity: .55; margin: 0 0 .7rem;
                     line-height: 1.45; }

  /* Stat cards: min-height, never height:100% — a percentage height inside an
     auto-height flex column collapses and the text spills over the border. */
  .ic-stat { border: 1px solid var(--ic-line); border-radius: 14px;
             background: var(--ic-surface); padding: 12px 15px;
             min-height: 92px; margin: 0 0 9px;
             display: flex; flex-direction: column; justify-content: flex-start;
             overflow: hidden; }
  .ic-stat-label { font-size: .71rem; letter-spacing: .07em; text-transform: uppercase;
                   opacity: .55; font-weight: 600; margin-bottom: 5px;
                   white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ic-stat-value { font-size: 1.6rem; font-weight: 660; line-height: 1.2;
                   letter-spacing: -.02em; margin-bottom: 4px; }
  .ic-stat-value.sm { font-size: 1.14rem; }
  .ic-stat-note { font-size: .77rem; opacity: .62; line-height: 1.35;
                  overflow-wrap: anywhere; }
  .ic-stat-note.good { color: var(--ic-good); opacity: .95; }
  .ic-stat-note.caution { color: var(--ic-caution); opacity: .95; }
  .ic-stat-note.bad { color: var(--ic-bad); opacity: .95; }

  /* Figures band: no cards. Used where the numbers are the content and a grid
     of bordered boxes reads as decoration rather than information. */
  /* Figures band: a strip of numbers, not a grid of cards. Twelve bordered
     boxes cost about 330px of height before the first chart and read as
     decoration; the same twelve numbers fit in one band a fifth of that. */
  .ic-figs { display: flex; flex-wrap: wrap; gap: 0;
             border-top: 1px solid var(--ic-line);
             border-bottom: 1px solid var(--ic-line);
             margin: .1rem 0 1.2rem; }
  /* Every cell padded identically, and the band pulled left by that padding so
     the first value still lines up with the page content. The previous version
     gave the first cell no left padding, which made it wider than the rest — so
     its label sat where no other label did and the row read as misaligned. */
  .ic-figs { margin-left: -.9rem; margin-right: -.9rem; }
  .ic-fig { flex: 1 1 7rem; padding: .5rem .9rem .55rem; min-width: 6rem; }
  .ic-fig + .ic-fig { border-left: 1px solid var(--ic-line); }
  .ic-fig-label { font-size: .63rem; letter-spacing: .085em;
                  text-transform: uppercase; opacity: .5;
                  margin-bottom: .12rem; white-space: nowrap;
                  overflow: hidden; text-overflow: ellipsis; }
  .ic-fig-value { font-size: 1.24rem; font-weight: 640; line-height: 1.2;
                  font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
  .ic-fig-value.good { color: var(--ic-good); }
  .ic-fig-value.caution { color: var(--ic-caution); }
  .ic-fig-value.bad { color: var(--ic-bad); }
  .ic-fig-note { font-size: .69rem; opacity: .5; margin-top: .05rem;
                 line-height: 1.3; }

  /* A hairline frame for a chart. Deliberately not the rounded, filled card
     used for stats: a chart needs a boundary so it reads as one object, but a
     heavy container competes with the plot for attention. */
  /* Bordered containers, targeted by the key frame() gives them. Streamlit puts
     the border on the stVerticalBlock itself in this version, not on a wrapper
     testid — the earlier selector matched nothing, so these panels were showing
     Streamlit's default styling rather than this one. */
  [class*="st-key-frame-"] {
      border-color: var(--ic-line) !important; border-radius: 6px !important;
      background: var(--ic-surface-2) !important;
      padding: .55rem .7rem .3rem !important; }
  /* No negative top margin here. The wordmark is the tallest text on the page,
     and pulling it up past the container's padding clipped its ascenders — the
     top of "Aerobic Engine" was sliced off. */
  /* Header bar pinned to the top: title, page tabs and filters — the three
     things you reach for from anywhere on a long page, and the only ones worth
     permanent screen space.

     Targeted by the container's `key`, which Streamlit renders as .st-key-*.
     That is the only stable handle it offers; the emotion class names change
     between releases. The background is opaque rather than blurred, because
     charts sliding under a translucent bar read as a rendering fault. */
  /* Pinned on the container's *wrapper*, not the container. A sticky element
     can only travel inside its own parent's box, and the keyed container's
     wrapper is only as tall as the header — so sticking the inner block held it
     for 131px and then let it scroll away with its parent. The wrapper's parent
     is the full-page block, which is what gives it the whole scroll to stick
     through. */
  .stMain [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
      /* Offset by Streamlit's own toolbar, which is absolutely positioned at
         z-index 999990 over the top 60px of the page. At top:0 the app title
         sat underneath it and was invisible once scrolled. */
      /* Below the 60px header, so nothing shares a line with Deploy or with
         the sidebar chevron. */
      position: sticky; top: 3.75rem; z-index: 90;
      background: var(--ic-page);
      padding: .35rem 0 .4rem;
      border-bottom: 1px solid var(--ic-line);
      /* Enough to clear the bar, and no more. Trimming this to .35rem put the
         first heading 7px *above* the bar's bottom edge — the bar is sticky, so
         anything that close is drawn underneath it. */
      margin-bottom: 1.35rem; }
  /* The popover panel has to clear the bar it hangs from. */
  .stMain .st-key-topbar [data-testid="stPopoverBody"] { z-index: 95; }

  @media (prefers-color-scheme: dark) {
    :root { --ic-page: #0e1117; }
  }

  /* Phones. Measured at 390x844 before this: the bar was 420px, half the
     viewport, with the title wrapping, the subtitle wrapping under it and six
     tabs wrapping onto three rows. Content did not start until 472px.

     Four changes, in order of how much they buy: the bar stops being sticky
     (half a phone screen permanently spent on navigation is not a trade worth
     making when scrolling back up is one flick), the subtitle is hidden as
     reference detail that belongs on About, the tabs become one horizontally
     scrollable row instead of three stacked ones, and the title and logo shrink. */
  @media (max-width: 700px) {
    /* Static, so it does not spend half a phone screen — but it still has to
       start below Streamlit's 60px header, which floats over the top. */
    .block-container { padding-top: 3.6rem; }
    .stMain [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
        position: static; padding: .1rem 0 .3rem; margin-bottom: .7rem; }
    .ic-brand-sub { display: none; }
    .ic-brand { gap: .5rem; margin-bottom: .25rem; }
    .ic-brand svg { width: 26px; height: 26px; }
    .ic-brand-name { font-size: 1.15rem; }
    /* One row that scrolls, rather than three that stack. */
    .st-key-topbar [data-testid="stButtonGroup"] {
        display: flex; flex-wrap: nowrap; overflow-x: auto;
        scrollbar-width: none; }
    .st-key-topbar [data-testid="stButtonGroup"]::-webkit-scrollbar {
        display: none; }
    .st-key-topbar [data-testid="stButtonGroup"] button { flex: 0 0 auto; }
    .block-container { padding-left: .9rem; padding-right: .9rem; }
    /* Figures read as a two-up grid on a phone; five across is unreadable. */
    .ic-fig { flex: 1 1 44%; min-width: 44%; }
  }

  .ic-brand { display: flex; align-items: center; gap: .55rem;
              margin: 0 0 .3rem; }
  .ic-brand svg { width: 26px; height: 26px; }
  .ic-brand svg { flex: 0 0 auto; opacity: .95; }
  .ic-brand-name { font-size: 1.22rem; font-weight: 680; line-height: 1.25;
                   letter-spacing: -.01em; }
  .ic-brand-sub { font-size: .8rem; opacity: .58; margin-top: .12rem; }
  .ic-sidebrand { display: none; }

  .ic-frame-title { font-size: .72rem; letter-spacing: .06em;
                    text-transform: uppercase; opacity: .55;
                    margin: 0 0 .15rem .05rem; }

  /* Dense two-column rows for "label ..... value" reference data. */
  .ic-rows { border-top: 1px solid var(--ic-line); margin: .1rem 0 1.1rem; }
  .ic-row { display: flex; justify-content: space-between; align-items: baseline;
            gap: 1rem; padding: .42rem .1rem;
            border-bottom: 1px solid var(--ic-line); }
  .ic-row-key { font-size: .87rem; opacity: .8; }
  .ic-row-val { font-size: .93rem; font-weight: 600;
                font-variant-numeric: tabular-nums; white-space: nowrap; }
  .ic-row-note { font-size: .74rem; opacity: .5; margin-left: .5rem;
                 font-weight: 400; }

  .ic-banner { border-radius: 14px; padding: 15px 18px; margin: 0 0 14px;
               border: 1px solid var(--ic-line); background: var(--ic-surface); }
  .ic-banner.good { border-left: 3px solid var(--ic-good); }
  .ic-banner.caution { border-left: 3px solid var(--ic-caution); }
  .ic-banner.bad { border-left: 3px solid var(--ic-bad); }
  .ic-banner-head { font-size: 1.04rem; font-weight: 640; line-height: 1.35;
                    margin-bottom: .3rem; }
  .ic-banner-body { font-size: .86rem; opacity: .78; line-height: 1.55; }

  .ic-today { border: 1px solid var(--ic-line); border-radius: 16px;
              background: var(--ic-surface); padding: 20px 22px; margin: 0 0 14px; }
  .ic-today-sport { font-size: 1.75rem; font-weight: 680; letter-spacing: -.02em;
                    line-height: 1.2; }
  .ic-today-meta { font-size: .95rem; opacity: .72; margin-top: .25rem; }
  .ic-today-why { font-size: .84rem; opacity: .58; margin-top: .6rem;
                  line-height: 1.55; }

  .ic-bar { display: flex; height: 11px; border-radius: 6px; overflow: hidden;
            margin: 6px 0 7px; background: var(--ic-surface-2); }
  .ic-bar span { display: block; height: 100%; }
  .ic-legend { font-size: .76rem; opacity: .62; display: flex; gap: 14px;
               flex-wrap: wrap; margin: 0 0 14px; }
  .ic-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px;
                 margin-right: 5px; }

  /* Week strip. min-width:0 on the cells stops long content forcing an overflow
     that pushes neighbours out of their column. */
  .ic-week { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr));
             gap: 8px; margin: 0 0 14px; }
  @media (max-width: 1100px) { .ic-week { grid-template-columns: repeat(4, minmax(0,1fr)); } }
  @media (max-width: 700px)  { .ic-week { grid-template-columns: repeat(2, minmax(0,1fr)); } }
  .ic-day { border: 1px solid var(--ic-line); border-radius: 11px; padding: 10px;
            background: var(--ic-surface-2); min-height: 82px; min-width: 0;
            overflow: hidden; }
  .ic-day.today { border-color: var(--ic-good); background: var(--ic-surface); }
  .ic-day.rest { opacity: .5; }
  .ic-day-name { font-size: .73rem; font-weight: 650; letter-spacing: .04em;
                 text-transform: uppercase; opacity: .75; }
  .ic-day-date { font-size: .67rem; opacity: .42; margin-left: 4px;
                 font-weight: 500; letter-spacing: 0; }
  .ic-item { font-size: .8rem; margin-top: 7px; line-height: 1.35;
             overflow-wrap: anywhere; }
  .ic-item b { font-weight: 620; }
  /* Done, and not done. Fading a completed session was the only signal before,
     and "faded" reads as less important rather than as finished — so a tick
     carries it and the colour says which of the two it is. */
  .ic-item.done { opacity: .72; }
  .ic-item.done b { color: var(--ic-good); }
  .ic-item.missed b { color: var(--ic-caution); }
  .ic-item-mark { font-weight: 700; margin-right: 3px; }
  .ic-item.done .ic-item-mark { color: var(--ic-good); }
  .ic-item.missed .ic-item-mark { color: var(--ic-caution); }
  .ic-item-zone { font-size: .69rem; opacity: .5; }
  /* A day whose work is all done gets a quiet green edge, so the week reads at
     a glance without having to look at each session. */
  .ic-day.settled { border-left: 3px solid var(--ic-good); }
  .ic-day.slipped { border-left: 3px solid var(--ic-caution); }
  .ic-strip-key { font-size: .74rem; opacity: .55; margin: -.5rem 0 1rem; }

  /* Chart panels */
  .ic-card-title { font-size: .93rem; font-weight: 620; letter-spacing: -.01em;
                   margin: 0 0 .15rem; }
  .ic-card-note { font-size: .78rem; opacity: .55; line-height: 1.45;
                  margin: 0 0 .5rem; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; height: 100%; }
  /* Equal-height panels across a row */
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { display: flex; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]
    > [data-testid="stVerticalBlock"] { width: 100%; }

  [data-testid="stTabs"] button p { font-size: .95rem; font-weight: 570; }
  section[data-testid="stSidebar"] { width: 320px !important; }
  [data-testid="stCaptionContainer"] p { font-size: .79rem; }
  hr { margin: 1.2rem 0; opacity: .35; }
</style>
"""


def esc(value: object) -> str:
    """Every interpolated value passes through here. Never skip it."""
    return html.escape(str(value), quote=True)


def load_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# The mark is the app's actual thesis drawn once: a rising performance line and
# a falling heart-rate line crossing. Inline SVG rather than an image file, for
# three reasons that all matter here — a deployed page makes no external request
# for it (the strict CSP on a hosted artifact would block a CDN anyway), it stays
# crisp at any size, and it inherits currentColor so it works in both themes
# without shipping two files.
LOGO_SVG = """
<svg viewBox="0 0 40 40" width="{size}" height="{size}" role="img"
     aria-label="Aerobic Engine" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="20" r="18.5" stroke="currentColor" stroke-opacity=".22"
          stroke-width="1.4"/>
  <path d="M7 27.5 C13 27.5 15.5 12.5 21 12.5 C26 12.5 30 16 33 12.5"
        stroke="#3FB68B" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M7 15.5 C12 15.5 13.5 22 17 22 L19 18 L21.5 26 L24 22 L33 22"
        stroke="#DB5F5A" stroke-width="1.7" stroke-linecap="round"
        stroke-opacity=".9"/>
</svg>
"""


def logo(size: int = 34) -> str:
    """The mark as inline SVG, ready to drop into a markdown block."""
    return LOGO_SVG.format(size=size)


def brand(title: str, subtitle: str = "") -> None:
    """Mark and wordmark, sized to sit level with Streamlit's toolbar buttons.

    The subtitle is optional and normally unused: reference detail at the top of
    every page wrapped onto a second line and pushed the content down, so it
    moved to the sidebar.
    """
    st.markdown(
        f'<div class="ic-brand">{logo(38)}'
        f'<div><div class="ic-brand-name">{esc(title)}</div>'
        + (f'<div class="ic-brand-sub">{esc(subtitle)}</div>' if subtitle else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str = "") -> None:
    # st.title rather than a markdown "#" heading, with anchor=False: Streamlit
    # attaches a linkable anchor to every markdown heading, which puts fragments
    # like #everything-so-far into the address bar and leaves them stranded there
    # when the heading is later renamed.
    #
    # Note st.title does its own escaping, so the raw string is correct here —
    # esc() is only for values interpolated into unsafe_allow_html blocks.
    st.title(title, anchor=False)
    if subtitle:
        st.markdown(f'<div class="ic-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="ic-section">{esc(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="ic-section-note">{esc(note)}</div>',
                    unsafe_allow_html=True)


def stat(
    label: str,
    value: object,
    note: str = "",
    tone: Tone = "neutral",
    small: bool = False,
) -> None:
    """One stat. `note` is only coloured when the tone is meaningful."""
    note_cls = f" {tone}" if tone != "neutral" and note else ""
    st.markdown(
        f'<div class="ic-stat">'
        f'<div class="ic-stat-label">{esc(label)}</div>'
        f'<div class="ic-stat-value{" sm" if small else ""}">{esc(value)}</div>'
        f'<div class="ic-stat-note{note_cls}">{esc(note) if note else "&nbsp;"}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def stats_row(items: list[dict]) -> None:
    """Equal-width, equal-height stat cards. Wraps on narrow screens."""
    if not items:
        return
    cols = st.columns(len(items), wrap=True, gap="small")
    for col, item in zip(cols, items):
        with col:
            stat(**item)


@contextmanager
def frame(title: str = ""):
    """Hairline boundary around a chart, so it reads as one object.

    Uses Streamlit's bordered container rather than a pair of raw divs: every
    Streamlit element is wrapped in its own block, so opening a div in one
    markdown call and closing it in another never actually contains what is
    drawn between them. The border is restyled to a rule instead of the default
    filled card, which competes with the plot for attention.
    """
    # Keyed, because the key is the only stable CSS handle Streamlit offers for
    # a container. Containers are not widgets, so a per-run counter is safe.
    box = st.container(border=True, key=f"frame-{next(_FRAME_SEQ)}")
    with box:
        if title:
            st.markdown(f'<div class="ic-frame-title">{esc(title)}</div>',
                        unsafe_allow_html=True)
        yield


def figures(items: list[dict]) -> None:
    """A band of numbers with hairline rules instead of cards.

    For places where the figures *are* the content: a row of bordered boxes there
    reads as decoration and takes three times the vertical space to say the same
    thing.
    """
    items = [i for i in items if i]
    if not items:
        return
    cells = "".join(
        f'<div class="ic-fig">'
        f'<div class="ic-fig-label">{esc(i.get("label", ""))}</div>'
        f'<div class="ic-fig-value'
        + (f' {i["tone"]}' if i.get("tone") in ("good", "caution", "bad") else "")
        + f'">{esc(i.get("value", "—"))}</div>'
        + (f'<div class="ic-fig-note">{esc(i["note"])}</div>' if i.get("note") else "")
        + "</div>"
        for i in items
    )
    st.markdown(f'<div class="ic-figs">{cells}</div>', unsafe_allow_html=True)


def rows(items: list[tuple]) -> None:
    """Dense `label — value` list for reference data, in place of a table widget.

    A dataframe brings its own chrome, sorting affordances and row indices; for a
    dozen fixed facts that is all noise.
    """
    items = [i for i in items if i]
    if not items:
        return
    out = []
    for item in items:
        key, val = item[0], item[1]
        note = item[2] if len(item) > 2 else ""
        out.append(
            f'<div class="ic-row"><div class="ic-row-key">{esc(key)}</div>'
            f'<div class="ic-row-val">{esc(val)}'
            + (f'<span class="ic-row-note">{esc(note)}</span>' if note else "")
            + "</div></div>"
        )
    st.markdown(f'<div class="ic-rows">{"".join(out)}</div>',
                unsafe_allow_html=True)


def banner(headline: str, body: str = "", tone: Tone = "neutral") -> None:
    st.markdown(
        f'<div class="ic-banner {tone}">'
        f'<div class="ic-banner-head">{esc(headline)}</div>'
        + (f'<div class="ic-banner-body">{esc(body)}</div>' if body else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def today_card(sport: str, meta: str, why: str = "") -> None:
    st.markdown(
        f'<div class="ic-today">'
        f'<div class="ic-today-sport">{SPORT_EMOJI.get(sport, "•")} '
        f"{esc(sport.title())}</div>"
        f'<div class="ic-today-meta">{esc(meta)}</div>'
        + (f'<div class="ic-today-why">{esc(why)}</div>' if why else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def proportion_bar(parts: Iterable[tuple[str, float, str]]) -> None:
    """`parts` is (label, value, colour). Values are normalised to the total."""
    parts = [(lbl, max(0.0, float(v)), c) for lbl, v, c in parts]
    total = sum(v for _, v, _ in parts) or 1.0
    segs = "".join(
        f'<span style="width:{v / total * 100:.2f}%;background:{c}"></span>'
        for _, v, c in parts if v > 0
    )
    legend = "".join(
        f'<div><i style="background:{c}"></i>{esc(lbl)} '
        f"{v / total * 100:.0f}%</div>"
        for lbl, v, c in parts if v > 0
    )
    st.markdown(f'<div class="ic-bar">{segs}</div>'
                f'<div class="ic-legend">{legend}</div>', unsafe_allow_html=True)


def week_strip(days: list[dict], key: bool = True) -> None:
    """days: [{name, date, today, items:[{sport,minutes,zone,done,missed}]}]

    `done` marks a session that has been logged, `missed` one whose day has gone
    by without it. Both matter on the page you look at every morning: "is this
    week on track" is a question about what did and did not happen, and a strip
    that only shows the plan cannot answer it.
    """
    cells = []
    any_done = any_missed = False
    for d in days:
        items = d.get("items") or []
        cls = "ic-day today" if d.get("today") else "ic-day"
        if not items:
            cls += " rest"
        elif all(it.get("done") for it in items):
            cls += " settled"
        elif any(it.get("missed") for it in items):
            cls += " slipped"
        parts = []
        for it in items:
            state = ("done" if it.get("done")
                     else "missed" if it.get("missed") else "")
            mark = ("✓" if state == "done"
                    else "·" if state == "missed" else "")
            any_done = any_done or state == "done"
            any_missed = any_missed or state == "missed"
            parts.append(
                f'<div class="ic-item{" " + state if state else ""}">'
                + (f'<span class="ic-item-mark">{mark}</span>' if mark else "")
                + f'{SPORT_EMOJI.get(it["sport"], "•")} <b>{esc(it["sport"])}</b>'
                + (f' {esc(it["minutes"])}′' if it.get("minutes") else "")
                + (f'<div class="ic-item-zone">{esc(it["zone"])}'
                   + (f' · {esc(it["hr"])}' if it.get("hr") else "")
                   + "</div>" if it.get("zone") or it.get("hr") else "")
                + "</div>")
        body = "".join(parts) or '<div class="ic-item" style="opacity:.4">rest</div>'
        cells.append(
            f'<div class="{cls}">'
            f'<div class="ic-day-name">{esc(d["name"])}'
            f'<span class="ic-day-date">{esc(d.get("date", ""))}</span></div>'
            f"{body}</div>"
        )
    st.markdown(f'<div class="ic-week">{"".join(cells)}</div>',
                unsafe_allow_html=True)
    # Only explain the marks that are actually on screen.
    if key and (any_done or any_missed):
        bits = []
        if any_done:
            bits.append("✓ logged")
        if any_missed:
            bits.append("· planned, not logged")
        st.markdown(f'<div class="ic-strip-key">{esc(" · ".join(bits))}</div>',
                    unsafe_allow_html=True)


@contextmanager
def card(title: str = "", note: str = ""):
    """A bordered panel. Charts sitting directly on the page background read as
    unfinished; giving each one a surface also makes a two-column grid legible."""
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="ic-card-title">{esc(title)}</div>',
                        unsafe_allow_html=True)
        if note:
            st.markdown(f'<div class="ic-card-note">{esc(note)}</div>',
                        unsafe_allow_html=True)
        yield


def chart(fig, height: int = 260, date_axis: bool = False) -> None:
    """One chart style for the whole app: transparent, minimal, no chartjunk.

    `date_axis` puts the weekday on the ticks. Training is planned by day of the
    week — a long run is a Sunday thing — so a bare "24 Aug" makes the reader do
    a calendar lookup to answer the question they actually have.
    """
    fig.update_layout(
        # Tight margins: Plotly's defaults reserve room for a title and axis
        # labels this app draws in HTML above the chart instead.
        # The legend sits above the plot, not below it. Below, at y=-0.2, it
        # landed in the same band as the tick labels and the two overlapped;
        # there is no bottom margin large enough to separate them without
        # wasting the space on every chart that has no legend.
        height=height, margin=dict(t=26, b=4, l=2, r=2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    if date_axis:
        # Explicit ticks, one per day, evenly sampled. Plotly's automatic ticks
        # land on sub-day intervals when the span is short, and formatting those
        # to day precision printed "Wed 19 Aug" twice in a row — which reads as a
        # rendering bug rather than as two sessions.
        # Only dates that actually carry a value. The database holds a wellness
        # row per day whether or not the watch recorded anything, so a trace's x
        # can run back weeks further than its y does — 47 rows from 9 July with
        # every metric empty until 18 August. Ranging on x alone drew six weeks
        # of blank axis before the first point, which reads as missing data
        # rather than as data that does not exist yet.
        #
        # `is not None` rather than truthiness throughout: x and y are often
        # numpy arrays or pandas Series, which raise on bool() instead of
        # answering it. A zero is also a real value and must not be dropped.
        def _blank(value: Any) -> bool:
            if value is None:
                return True
            return value != value          # NaN, without importing numpy

        days = set()
        for trace in fig.data:
            xs = getattr(trace, "x", None)
            if xs is None:
                continue
            ys = getattr(trace, "y", None)
            for i, x in enumerate(xs):
                if x is None:
                    continue
                if ys is not None and i < len(ys) and _blank(ys[i]):
                    continue
                days.add(str(x)[:10])
        days = sorted(days)
        if 1 < len(days) <= 400:
            step = max(1, (len(days) + 7) // 8)
            picked = days[::step]
            if days[-1] not in picked:
                picked.append(days[-1])
            fig.update_xaxes(tickmode="array", tickvals=picked)
        # Day-month on the ticks, with the weekday kept in front: training is
        # planned by day of the week, and the full year on every tick is noise.
        fig.update_xaxes(tickformat="%a %d-%m", hoverformat="%a %d-%m-%Y")
        # Clamp the axis to the data. Several charts ask for a fixed window —
        # twelve weeks of volume, ninety days of sessions — so an account with
        # three weeks of history drew two months of blank axis before the first
        # point, which reads as missing data rather than as data that does not
        # exist yet.
        if len(days) > 1:
            span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
            pad = timedelta(days=max(1, round(span * 0.04)))
            fig.update_xaxes(range=[
                (date.fromisoformat(days[0]) - pad).isoformat(),
                (date.fromisoformat(days[-1]) + pad).isoformat(),
            ])
    fig.update_yaxes(gridcolor="rgba(140,158,176,.15)", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
