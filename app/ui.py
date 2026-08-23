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
from typing import Iterable, Literal

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
SPORT_EMOJI = {"swim": "🏊", "bike": "🚲", "run": "🏃", "strength": "🦵",
               "brick": "🚲🏃", "rest": "😴", "other": "•"}

CSS = """
<style>
  :root {
    --ic-good: #3FB68B; --ic-caution: #E0A33E; --ic-bad: #DB5F5A;
    --ic-line: rgba(140,158,176,.28);
    --ic-surface: rgba(140,158,176,.09);
    --ic-surface-2: rgba(140,158,176,.045);
  }
  .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1080px; }

  /* Every injected block owns its own vertical space. Streamlit's container gap
     is deliberately NOT overridden here: doing so removed the spacing between
     stacked elements and made custom divs collide. */
  .ic-stat, .ic-banner, .ic-today, .ic-week, .ic-bar, .ic-legend,
  .ic-section, .ic-section-note, .ic-sub { box-sizing: border-box; }

  h1 { font-size: 1.75rem !important; font-weight: 680; letter-spacing: -.02em;
       margin: 0 0 .2rem !important; }
  .ic-sub { font-size: .86rem; opacity: .6; margin: 0 0 .6rem; }
  .ic-section { font-size: 1.02rem; font-weight: 620; letter-spacing: -.01em;
                padding-top: .9rem; margin: 0 0 .2rem; }
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
  .ic-figs { display: flex; flex-wrap: wrap; gap: 0;
             border-top: 1px solid var(--ic-line);
             border-bottom: 1px solid var(--ic-line);
             margin: .1rem 0 1.1rem; }
  .ic-fig { flex: 1 1 8.5rem; padding: .7rem 1.1rem .75rem 0; min-width: 7rem; }
  .ic-fig + .ic-fig { padding-left: 1.1rem;
                      border-left: 1px solid var(--ic-line); }
  .ic-fig-label { font-size: .66rem; letter-spacing: .09em; text-transform: uppercase;
                  opacity: .55; margin-bottom: .2rem; }
  .ic-fig-value { font-size: 1.5rem; font-weight: 640; line-height: 1.15;
                  font-variant-numeric: tabular-nums; }
  .ic-fig-note { font-size: .74rem; opacity: .55; margin-top: .1rem;
                 font-variant-numeric: tabular-nums; }

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
  .ic-item.done { opacity: .48; }
  .ic-item-zone { font-size: .69rem; opacity: .5; }

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
        f'<div class="ic-fig-value">{esc(i.get("value", "—"))}</div>'
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


def week_strip(days: list[dict]) -> None:
    """days: [{name, date, today, items:[{sport,minutes,zone,done}]}]"""
    cells = []
    for d in days:
        items = d.get("items") or []
        cls = "ic-day today" if d.get("today") else "ic-day"
        if not items:
            cls += " rest"
        body = "".join(
            f'<div class="ic-item{" done" if it.get("done") else ""}">'
            f'{SPORT_EMOJI.get(it["sport"], "•")} <b>{esc(it["sport"])}</b>'
            + (f' {esc(it["minutes"])}′' if it.get("minutes") else "")
            + (f'<div class="ic-item-zone">{esc(it["zone"])}'
               + (f' · {esc(it["hr"])}' if it.get("hr") else "")
               + "</div>" if it.get("zone") or it.get("hr") else "")
            + "</div>"
            for it in items
        ) or '<div class="ic-item" style="opacity:.4">rest</div>'
        cells.append(
            f'<div class="{cls}">'
            f'<div class="ic-day-name">{esc(d["name"])}'
            f'<span class="ic-day-date">{esc(d.get("date", ""))}</span></div>'
            f"{body}</div>"
        )
    st.markdown(f'<div class="ic-week">{"".join(cells)}</div>',
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
        height=height, margin=dict(t=8, b=4, l=4, r=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    if date_axis:
        fig.update_xaxes(tickformat="%a %-d %b", hoverformat="%a %-d %b %Y")
    fig.update_yaxes(gridcolor="rgba(140,158,176,.15)", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
