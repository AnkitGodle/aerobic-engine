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
       exposes no CSS variable for its own theme background, so this is the one
       set in .streamlit/config.toml. The app is dark for everyone, so this is a
       constant rather than a media query — as a query it painted a white bar
       over a dark page for any viewer whose browser preferred light. */
    --ic-page: #0e1117;
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
    .ic-brand svg { width: 27px; height: 27px; }
    .ic-brand-name { font-size: 1.3rem; }
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
  .ic-brand svg { width: 32px; height: 32px; }
  .ic-brand svg { flex: 0 0 auto; opacity: .95; }
  /* The wordmark carries the whole top bar, and at 1.22rem it read as a label
     rather than a name. Sized up with the mark beside it. */
  .ic-brand-name { font-size: 1.52rem; font-weight: 700; line-height: 1.2;
                   letter-spacing: -.02em; }
  .ic-brand-sub { font-size: .8rem; opacity: .58; margin-top: .12rem; }
  .ic-sidebrand { display: none; }

  /* Poster type. Used on About, which is a page someone reads once rather than
     a dashboard they scan — so it gets a headline that behaves like one. */
  .ic-hero { margin: .2rem 0 1.1rem; }
  .ic-hero-kicker { font-size: .72rem; letter-spacing: .12em;
                    text-transform: uppercase; opacity: .55; font-weight: 650; }
  .ic-hero-head { font-size: 2.1rem; font-weight: 700; line-height: 1.16;
                  letter-spacing: -.03em; margin: .3rem 0 .5rem;
                  max-width: 26ch; }
  .ic-hero-body { font-size: 1.02rem; line-height: 1.6; opacity: .8;
                  max-width: 62ch; }
  @media (max-width: 700px) { .ic-hero-head { font-size: 1.6rem; } }

  .ic-pull { border-left: 3px solid var(--ic-good); padding: .1rem 0 .1rem 1rem;
             margin: 1.1rem 0; font-size: 1.16rem; font-weight: 600;
             line-height: 1.45; letter-spacing: -.01em; max-width: 46ch; }
  .ic-pull-note { font-size: .84rem; font-weight: 400; opacity: .62;
                  margin-top: .3rem; line-height: 1.5; }

  .ic-prose { font-size: .95rem; line-height: 1.65; max-width: 66ch;
              margin: 0 0 .9rem; }
  .ic-prose b { font-weight: 640; }

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

  /* A link that leaves the app. Deliberately not st.link_button, which is
     styled as a form control and so reads like something that changes state
     here. The brand colour carries it. */
  /* !important on the underline only: Streamlit styles links inside its own
     markdown containers with a more specific selector, so the chip arrived
     underlined and read as body text with a border round it. */
  a.ic-linkchip, a.ic-linkchip:hover, a.ic-linkchip:visited {
      display: inline-flex; align-items: center; gap: .4rem;
      border: 1px solid currentColor; border-radius: 999px;
      padding: .3rem .75rem; font-size: .84rem; font-weight: 600;
      text-decoration: none !important; margin: .1rem 0 .35rem; }
  a.ic-linkchip:hover { filter: brightness(1.08); }
  /* Mark only, where the brand is the label — three words of text each turned
     the sidebar into a row of buttons that all read the same. */
  a.ic-linkchip.mark { padding: .34rem; border-radius: 50%; }
  .ic-chiprow { display: flex; flex-wrap: wrap; gap: .4rem;
                margin: .1rem 0 .35rem; }

  /* A flow of steps. Streamlit renders no diagram of its own and mermaid needs a
     component, so this is boxes and arrows in CSS — which also wraps on a phone,
     where a fixed-width image would not. */
  .ic-flow { display: flex; flex-wrap: wrap; align-items: stretch;
             gap: .35rem; margin: .2rem 0 1.1rem; }
  .ic-flow-step { flex: 1 1 8.5rem; min-width: 7.5rem;
                  border: 1px solid var(--ic-line); border-radius: 12px;
                  background: var(--ic-surface); padding: .6rem .7rem; }
  .ic-flow-step.accent { border-color: var(--ic-good);
                         background: rgba(63,182,139,.10); }
  .ic-flow-step.guard { border-color: var(--ic-caution);
                        background: rgba(224,163,62,.10); }
  .ic-flow-icon { font-size: 1.15rem; line-height: 1.2; }
  .ic-flow-name { font-size: .84rem; font-weight: 640; margin-top: .15rem; }
  .ic-flow-note { font-size: .72rem; opacity: .62; line-height: 1.35;
                  margin-top: .1rem; }
  /* Splits, in the shape everyone already knows from Strava: the kilometre, the
     pace, a bar as long as that pace was quick, and what it cost in heartbeats.
     A table of divs rather than a chart — it is a list, and a list renders
     instantly. */
  .ic-splits { margin: .2rem 0 .9rem; font-variant-numeric: tabular-nums; }
  .ic-split { display: grid; grid-template-columns: 2.6rem 3.4rem 1fr 4.2rem 3.4rem;
              align-items: center; gap: .55rem; padding: .3rem .1rem;
              border-bottom: 1px solid var(--ic-line); font-size: .84rem; }
  .ic-split.head { font-size: .68rem; letter-spacing: .07em; opacity: .5;
                   text-transform: uppercase; border-bottom-width: 1px; }
  .ic-split-bar { height: 9px; border-radius: 5px; background: var(--ic-surface);
                  overflow: hidden; }
  .ic-split-bar span { display: block; height: 100%; border-radius: 5px; }
  .ic-split-hr { text-align: right; font-weight: 600; }
  .ic-split-elev { text-align: right; opacity: .55; font-size: .78rem; }
  .ic-split-num { opacity: .55; }
  @media (max-width: 700px) {
    .ic-split { grid-template-columns: 2.2rem 3.2rem 1fr 3.8rem; }
    .ic-split-elev { display: none; }
  }

  .ic-flow-arrow { align-self: center; opacity: .35; font-size: .95rem;
                   padding: 0 .1rem; }
  @media (max-width: 700px) { .ic-flow-arrow { display: none; } }
  .ic-chiprow a.ic-linkchip { margin: 0; }

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


# Brand marks for the outbound links, as inline SVG so the page makes no external
# request for them (and a hosted page's CSP would block one anyway). Paths from
# simple-icons, which publishes them for exactly this use.
BRAND_ICONS: dict[str, str] = {
    "strava": "M15.387 17.944l-2.089-4.116h-3.065L15.387 24l5.15-10.172h-3.066m-7.008-5.599l2.836 5.598h4.172L10.463 0l-7 13.828h4.169",
    "instagram": "M7.0301.084c-1.2768.0602-2.1487.264-2.911.5634-.7888.3075-1.4575.72-2.1228 1.3877-.6652.6677-1.075 1.3368-1.3802 2.127-.2954.7638-.4956 1.6365-.552 2.914-.0564 1.2775-.0689 1.6882-.0626 4.947.0062 3.2586.0206 3.6671.0825 4.9473.061 1.2765.264 2.1482.5635 2.9107.308.7889.72 1.4573 1.388 2.1228.6679.6655 1.3365 1.0743 2.1285 1.38.7632.295 1.6361.4961 2.9134.552 1.2773.056 1.6884.069 4.9462.0627 3.2578-.0062 3.668-.0207 4.9478-.0814 1.28-.0607 2.147-.2652 2.9098-.5633.7889-.3086 1.4578-.72 2.1228-1.3881.665-.6682 1.0745-1.3378 1.3795-2.1284.2957-.7632.4966-1.636.552-2.9124.056-1.2809.0692-1.6898.063-4.948-.0063-3.2583-.021-3.6668-.0817-4.9465-.0607-1.2797-.264-2.1487-.5633-2.9117-.3084-.7889-.72-1.4568-1.3876-2.1228C21.2982 1.33 20.628.9208 19.8378.6165 19.074.321 18.2017.1197 16.9244.0645 15.6471.0093 15.236-.005 11.977.0014 8.718.0076 8.31.0215 7.0301.0839m.1402 21.6932c-1.17-.0509-1.8053-.2453-2.2287-.408-.5606-.216-.96-.4771-1.3819-.895-.422-.4178-.6811-.8186-.9-1.378-.1644-.4234-.3624-1.058-.4171-2.228-.0595-1.2645-.072-1.6442-.079-4.848-.007-3.2037.0053-3.583.0607-4.848.05-1.169.2456-1.805.408-2.2282.216-.5613.4762-.96.895-1.3816.4188-.4217.8184-.6814 1.3783-.9003.423-.1651 1.0575-.3614 2.227-.4171 1.2655-.06 1.6447-.072 4.848-.079 3.2033-.007 3.5835.005 4.8495.0608 1.169.0508 1.8053.2445 2.228.408.5608.216.96.4754 1.3816.895.4217.4194.6816.8176.9005 1.3787.1653.4217.3617 1.056.4169 2.2263.0602 1.2655.0739 1.645.0796 4.848.0058 3.203-.0055 3.5834-.061 4.848-.051 1.17-.245 1.8055-.408 2.2294-.216.5604-.4763.96-.8954 1.3814-.419.4215-.8181.6811-1.3783.9-.4224.1649-1.0577.3617-2.2262.4174-1.2656.0595-1.6448.072-4.8493.079-3.2045.007-3.5825-.006-4.848-.0608M16.953 5.5864A1.44 1.44 0 1 0 18.39 4.144a1.44 1.44 0 0 0-1.437 1.4424M5.8385 12.012c.0067 3.4032 2.7706 6.1557 6.173 6.1493 3.4026-.0065 6.157-2.7701 6.1506-6.1733-.0065-3.4032-2.771-6.1565-6.174-6.1498-3.403.0067-6.156 2.771-6.1496 6.1738M8 12.0077a4 4 0 1 1 4.008 3.9921A3.9996 3.9996 0 0 1 8 12.0077",
    "whatsapp": "M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.347-.347.52-.52.174-.174.232-.298.347-.497.115-.198.057-.371-.058-.52-.115-.148-.669-1.611-.916-2.206-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413Z",
    "github": "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12",
}


def brand_icon(name: str, size: int = 14) -> str:
    """One brand mark, inheriting the surrounding colour."""
    path = BRAND_ICONS.get(name.lower())
    if not path:
        return ""
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" '
            f'fill="currentColor" aria-hidden="true" '
            f'style="flex:0 0 auto"><path d="{path}"/></svg>')


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
        f'<div class="ic-brand">{logo(44)}'
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


def link_chips(items: Iterable[tuple[str, str, str]],
               labels: bool = True) -> None:
    """Outbound links as chips, each in its destination's own colour.

    One markdown block for the row: Streamlit puts every call in its own block,
    so a chip per call stacked them vertically and turned two links into two
    rows of sidebar.

    `target="_blank"` with `rel="noopener noreferrer"`: the app sits behind a PIN
    and a new tab must not be handed a reference back to this one.
    """
    chips = "".join(
        f'<a class="ic-linkchip{"" if labels else " mark"}" '
        f'style="color:{esc(color)}" title="{esc(label)}" aria-label="{esc(label)}" '
        f'href="{esc(url)}" target="_blank" rel="noopener noreferrer">'
        + brand_icon(label, 15 if labels else 18)
        + (f"{esc(label)} ↗" if labels else "")
        + "</a>"
        for label, url, color in items if url
    )
    if chips:
        st.markdown(f'<div class="ic-chiprow">{chips}</div>',
                    unsafe_allow_html=True)


def hero(kicker: str, headline: str, body: str = "") -> None:
    """The top of a page that is read rather than scanned."""
    st.markdown(
        '<div class="ic-hero">'
        + (f'<div class="ic-hero-kicker">{esc(kicker)}</div>' if kicker else "")
        + f'<div class="ic-hero-head">{esc(headline)}</div>'
        + (f'<div class="ic-hero-body">{esc(body)}</div>' if body else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def pull(text: str, note: str = "") -> None:
    """A line worth stopping on, with an optional line of detail under it."""
    st.markdown(
        f'<div class="ic-pull">{esc(text)}'
        + (f'<div class="ic-pull-note">{esc(note)}</div>' if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def prose(text: str) -> None:
    """A paragraph, at a width that can be read. `text` may contain <b> tags."""
    st.markdown(f'<div class="ic-prose">{text}</div>', unsafe_allow_html=True)


def flow(steps: Iterable[tuple[str, str, str]], accent: str = "",
         guard: str = "") -> None:
    """A left-to-right flow of steps: (icon, name, note).

    `accent` and `guard` name a step to colour — the one that does the thinking
    and the one that checks it — because a flow where every box looks the same
    says the layers are equal, and here they are deliberately not.
    """
    parts = []
    for i, (icon, name, note) in enumerate(steps):
        cls = "ic-flow-step"
        if name == accent:
            cls += " accent"
        elif name == guard:
            cls += " guard"
        if i:
            parts.append('<div class="ic-flow-arrow">→</div>')
        parts.append(
            f'<div class="{cls}"><div class="ic-flow-icon">{esc(icon)}</div>'
            f'<div class="ic-flow-name">{esc(name)}</div>'
            f'<div class="ic-flow-note">{esc(note)}</div></div>')
    st.markdown(f'<div class="ic-flow">{"".join(parts)}</div>',
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


def splits(rows: list[dict], unit: str = "km") -> None:
    """rows: [{label, pace, bar (0-1), bar_color, hr, hr_color, elev}]

    The bar is proportional to speed, not to time, so the longest bar is the
    quickest split — which is the way Strava draws it and the way people already
    read it.
    """
    rows = [r for r in rows if r]
    if not rows:
        return
    head = ('<div class="ic-split head"><div>' + esc(unit) + "</div><div>pace</div>"
            "<div></div><div class='ic-split-hr'>bpm</div>"
            "<div class='ic-split-elev'>elev</div></div>")
    body = []
    for row in rows:
        width = max(4.0, min(100.0, float(row.get("bar") or 0) * 100))
        colour = row.get("bar_color") or TONE_COLOR["neutral"]
        hr = row.get("hr")
        hr_colour = row.get("hr_color") or ""
        body.append(
            '<div class="ic-split">'
            f'<div class="ic-split-num">{esc(row.get("label", ""))}</div>'
            f'<div>{esc(row.get("pace", "—"))}</div>'
            f'<div class="ic-split-bar"><span style="width:{width:.1f}%;'
            f'background:{esc(colour)}"></span></div>'
            f'<div class="ic-split-hr"'
            + (f' style="color:{esc(hr_colour)}"' if hr_colour else "")
            + f">{esc(f'{hr:.0f}' if hr else '—')}</div>"
            f'<div class="ic-split-elev">'
            + (esc(row["elev"]) if row.get("elev") else "")
            + "</div></div>")
    st.markdown(f'<div class="ic-splits">{head}{"".join(body)}</div>',
                unsafe_allow_html=True)


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
