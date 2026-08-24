# Aerobic Engine — Build Brief

A personal endurance training analytics + adaptive planning web app. Reads Garmin
data, shows whether fitness is rising while HR drops, and generates an adaptive
weekly plan (swim / bike / run / leg strength) that responds to how the user feels.

---

## 1. Scope

- **Personal, single user, free.** Garmin only. No Strava. There is no
  multi-user or commercial phase — this is one athlete's dashboard and nothing
  in it should be built for anyone else.
- Watch is a **Garmin Forerunner 265**. It captures run, bike, pool/open-water swim
  and strength, plus Training Readiness / Training Status / HRV / VO2max / race
  predictor. It does **not** have Triathlon Coach, so the app is the cross-sport
  coach; Garmin is just the sensor + data layer.

## 2. Tech stack (decided — don't re-litigate)

- **Language:** Python.
- **Data source:** `garminconnect` (python-garminconnect) — unofficial,
  credentials-based. Personal use only.
- **Storage:** Postgres (Neon) is the live database; SQLite is the local
  fallback when `DATABASE_URL` is unset. One SQL dialect in the source,
  translated on the way out — see `Store.sql()`.
- **UI:** Streamlit.
- **AI layer:** Gemini (`gemini-3.6-flash`, free tier) via the OpenAI-compatible
  endpoint. Structured JSON in/out, behind an interface, so the provider is a
  config swap — Groq, Cerebras, OpenRouter, Anthropic and Azure all work.
- **Deploy:** local first; then the dashboard on Streamlit Community Cloud
  (private app). See Section 11.

## 3. Architecture principle (important)

The **data-fetch, analysis and planner are pure-Python modules with no Streamlit
imports.** The UI is a thin layer over them. Not for portability — because
training logic that is tangled into a UI cannot be tested, and the guardrails are
the product. Same rule for the AI call: it sits behind
`plan_week(payload) -> plan`, so switching provider is a config change.

Enforced in practice: nothing under `core/` imports `streamlit`, and only
`core/ai.py` knows an LLM exists.

## 4. Repo structure

```
aerobic-engine/
  app/streamlit_app.py       # UI only — thin
  core/
    garmin_client.py         # login (cached session), incremental fetch
    store.py                 # Postgres/SQLite read/write, schema, migrations
    analysis.py              # EF, decoupling, RHR/HRV/VO2max trends, weekly load
    planner.py               # facts + rules envelope + enforcement + calls ai
    ai.py                    # plan_week(payload) -> plan ; JSON contract
    insights.py              # deterministic page/chart readings, then prose
    sync.py                  # the sync itself, shared by CLI and dashboard
    auth.py                  # read and write PIN gates, hashed, rate-limited
    garmin_guard.py          # request pacing, budgets, breaker, sync lock
    garmin_workout.py        # push a planned session to the watch
    strength.py              # fixed exercise library + deterministic progression
    schemas.py               # pydantic models for payloads
  scripts/fetch.py           # the incremental sync (local / scheduled)
  scripts/migrate_to_postgres.py  # copy a local SQLite file into Postgres
  scripts/set_pin.py         # generate a read or write PIN salt + hash
  scripts/export_tokens.py   # export the Garmin session for a hosted deploy
  tests/                     # guardrail + analysis regression suite
  data/aerobic_engine.db         # gitignored
  .env                       # gitignored — Garmin, database and AI keys
```

## 5. Data layer (`garmin_client.py`, `store.py`)

- Login with email/password from `.env`. **Log in once, cache the session, reuse
  it.** Garmin flags repeated datacenter logins and may require MFA. The MFA path
  is handled.
- **Fetch/display split:** the fetch step needs Garmin login and runs **locally or
  as a scheduled job**, never from the hosted dashboard. The dashboard is
  read-only against SQLite.
- **Incremental:** only pull activities newer than the latest stored `start_time`
  (with a two-day overlap, because Garmin backfills late-syncing activities).
- Per activity: id, sport, start_time, duration, distance, avg/max HR, avg
  speed/pace, avg power (bike), HR stream, elevation, RPE if present.
- Daily wellness: resting HR, HRV status, VO2max (run + bike), Training Readiness,
  Training Status, training load.
- Method names vary by library version — **check the installed version's API**
  rather than assuming.
- Tables: `activities`, `activity_metrics`, `hr_streams`, `daily_wellness`,
  `race_predictions`, `strength_log`, `checkins`, `plans`, `sync_state`.

## 6. Analysis (`analysis.py`) — the "am I improving / is HR dropping" layer

All deterministic. No AI here.

- **Efficiency Factor (EF)** per sport, per activity = aerobic output ÷ avg HR.
  Run/swim: `speed / avg_HR`. Bike: `avg_power / avg_HR` (Pw:HR) when power is
  available, else speed.
- **Filter to steady aerobic sessions only** before trending — excludes intervals,
  races, short efforts, high anaerobic TE, wide HR spread, ragged HR streams.
  Rising EF over weeks = fitness up at the same HR. This is the headline chart.
- **Aerobic decoupling** on long steady sessions: EF(first half) vs EF(second
  half), reported as % drift. Under 5% = good aerobic durability.
- **Resting HR** trend (down = good) and **HRV** trend (up = adapting), each 7-day
  vs 28-day baseline.
- **VO2max / race predictor** over time, straight from Garmin.
- **Weekly volume + load** per sport, plus strength session count.

## 7. Planner (`planner.py`) — three layers

1. **Facts (deterministic):** this week's completed sessions by sport, volume/load,
   EF trend, recovery signals. Ground truth from `analysis.py`.
2. **Envelope (rules — the safety backstop):** endurance **base phase** default week
   — swim 2–3, bike 2–3 (incl. 1 long), run 2–3 (incl. 1 long, conservative),
   legs 2, ≥1 full rest day.
   - **Bike-heavy for aerobic base** (best low-impact aerobic volume tool *and* a
     race discipline); running frequent but conservative to protect tendons; swims
     mainly technique.
   - Progression cap: **volume +10%/week max.** Every 4th week = deload (~35% cut).
   - A **brick** (bike→run) every 2 weeks.
   - **Deload trigger** (forced regardless of mood): HRV ≥5% below baseline, OR
     resting HR >5 bpm over baseline, OR Training Readiness <35, OR acute:chronic
     load ratio >1.3, OR a bad Garmin HRV/training status.
3. **AI layer (`ai.py`):** adapts the *remaining* week **within the envelope** and
   explains why. It adjusts volume / intensity / placement and writes short
   rationales. It must not invent training science, override the deload flag, or
   exceed the progression cap.

**Why the guardrail matters:** a raw LLM is sycophantic — say "I feel great" and it
hands you a reckless week; say "I'm tired" and it cancels everything. Rules hold
the non-negotiables; the AI negotiates around them.

`planner.enforce()` is where that is made true. It re-checks every constraint in
code after the model answers: per-session ceilings, deload zone downgrades, session
counts, strength library membership and placement, the volume budget, required long
sessions, and minimum rest days. It also rewrites any rationale the numbers no
longer support, and returns `source="ai_repaired"` when it had to intervene.

## 8. Check-in input loop

- UI inputs: sliders for sleep, soreness, motivation, time-available-today, plus a
  free-text box ("knee feels off", "only 45 min", "race in 3 weeks").
- The planner re-plans the remaining week from check-in + metrics.
- The user can push back in plain language → re-plan again.
- The last two weeks of check-ins and completions are fed as context so the AI
  responds to recurring patterns, not just today.

## 9. Strength module (`strength.py`) — inside the planner, not standalone

Goal: strengthen **tendons and muscles** to protect run volume. Muscle wants
heavy/low-rep; tendons want slow, heavy or isometric loading and adapt slowly.

- **Fixed, curated exercise library — hard-coded. The AI must NOT invent
  exercises; it may only pick from these IDs and adjust sets/volume/placement.**
  This is a safety line, enforced by `validate_exercise_ids()`.
  Calf raises (straight-leg, bent-knee soleus, single-leg) · split squat / reverse
  lunge · RDL → single-leg RDL · step-ups · isometrics (wall sit, Spanish squat,
  single-leg calf hold) · tibialis raises.
- Session: 2×/week, ~20–30 min, minimal kit. Strength 3–4×5–8 reps, slow tempo;
  isometrics ~4–5×30–45s.
- **Progression is deterministic:** log reps/load; add one rep, then one load step
  at the top of the rep range — and only if the last session was completed cleanly
  and pain-free. **Tendon injuries come from load jumps, not load — no
  plyometrics/jumping in base.**
- **Smart placement:** the planner slots legs away from long runs and quality
  bikes (never the same day, never the day before a long run) and lightens or
  drops them when readiness/HRV is low.
- **Garmin loop:** log the session in the 265's strength mode so it counts toward
  training load, then it is pulled back in so load/readiness stay accurate.

## 10. AI contract (`ai.py`)

- `plan_week(payload) -> plan`, behind an interface. `gemini` is the default
  (~1500 requests/day free, and a planner call is a chunky ~2.5K tokens, so a
  request-per-day cap suits it better than a tokens-per-minute one). Also
  supported: `groq`, `cerebras`, `openrouter`, `anthropic`, `azure`, or `none`
  to disable. `AI_BACKEND` takes a comma-separated chain (`gemini,groq`); the
  summary phase fans out across every entry in it.
- **Input JSON:** `{ completed_this_week, recovery_signals, envelope,
  strength_state, checkin, history }`.
- **Output JSON (strict — no prose outside JSON):**
  ```json
  {
    "week_plan": [
      {"day":"Mon","sport":"bike","duration_min":90,"target_zone":"Z2",
       "purpose":"aerobic base","exercise_ids":[],"why":"one-line reason"}
    ],
    "flags": ["e.g. recommending deload, HRV 15% below baseline"],
    "adjustments_made": ["short bullets"]
  }
  ```
- Strength days set `sport:"strength"` and populate `exercise_ids` **from the
  library only**.
- **Constraints enforced in code, not just the prompt:** respect the deload flag;
  never exceed the progression cap; strength `exercise_ids` must exist in the
  library; parse/validate the JSON and fall back to the rules-only plan if the
  output is invalid.

## 11. Deployment

- **Database:** a Neon Postgres project (ap-southeast-1) is the live store. The
  local sync writes there and the dashboard reads it, so the two cannot drift.
  Verified against the live server, not just the translation layer.
- **Access from anywhere:** deploy the **dashboard** to Streamlit Community Cloud
  as a **private app** (health data must not sit on a public URL), or set
  a read PIN (`scripts/set_pin.py --read`) for the built-in gate. Read and
  write PINs are separate and separately rate-limited.
- **Keep fetch local/scheduled** — don't run Garmin login from a cloud host.
- **Why not SQLite when hosted:** a free host's disk is ephemeral. If the file
  vanished, the next sync would re-pull months of history — hundreds of requests
  from a datacenter IP, which is exactly what gets an account flagged.
- `scripts/migrate_to_postgres.py` copies a local SQLite file up, costing zero
  Garmin calls. Re-runnable: every table upserts on its real key.

## 12. Constraints & guardrails (summary)

- Rules override the AI, always. The AI adjusts volume / intensity / placement only.
- Strength exercises are a fixed library; the AI cannot add exercises. Strength
  progression is deterministic.
- Deload is rule-triggered by recovery signals regardless of user mood.
- **No Strava in v1.** If added later, Strava data must never reach the AI layer
  (their API bans AI use), and Strava stays an optional connector, never the
  backbone.
- Not medical advice. Persistent tendon pain → see a physio; the app surfaces a
  gentle note and does not try to treat it.

---

## Implementation notes (deviations from the brief, and why)

- **`garminconnect` 0.3.2 no longer depends on `garth`.** The brief said to use
  garth's session cache; that version handles token persistence itself via
  `Garmin.login(tokenstore=<path>)`, which loads a cached session if present and
  otherwise logs in and writes the tokens there. MFA goes through
  `prompt_mfa=<callable>`. `core/garmin_client.py` targets that API.
- **Garmin JSON shapes move between firmware and account combinations**, so field
  extraction goes through `dig()`, a recursive key search, instead of hard-coded
  paths. Missing values become `None` rather than exceptions.
- **HR streams are stored downsampled** (≤600 points/activity) rather than either
  full-resolution or summary-only, so decoupling can be recomputed after a change
  to the maths without re-fetching from Garmin.
- **Sessions are dropped, not shrunk, when the volume cap bites.** Scaling alone
  produced weeks of useless 20-minute stubs, so anything that would fall below a
  per-sport floor is dropped instead — swims first, the long ride last.
- **A brick satisfies the long-*bike* requirement, never the long run**, since its
  run portion is only 20–25 minutes.
- **A required long session may overshoot the weekly budget slightly**, and says so
  in `adjustments_made`. Losing the week's aerobic anchor is worse than a 2%
  overshoot.
### Findings from the first real sync (Garmin FR265, Aug 2026)

Verified against a live account. Each of these was a real bug or a real design
error, found only by looking at actual payloads:

- **`dig()` must honour key priority, not dict order.** `hrvSummary` lists
  `weeklyAvg` before `lastNightAvg`, so a breadth-first search returned the weekly
  average where last night's value was asked for. Keys are now tried in the order
  given.
- **`get_training_readiness` returns every intraday recalculation**, not one row
  per day. On this account: 39 at wake-up, then 1 an hour after a 78-minute
  multisport session. Taking the newest snapshot means planning off a
  post-exercise trough that recovers overnight, so `morning_entry()` takes the
  `AFTER_WAKEUP_RESET` snapshot instead — which is also the number Garmin's own
  widget shows.
- **ACWR is meaningless before ~3 weeks of history.** With one week of data the
  chronic average is just the acute week over four, giving a ratio near 4.0 and
  forcing a permanent deload on every new account. `acwr_from_activities` now
  returns None until there is enough history, and the deload trigger simply
  doesn't fire on it.
- **Garmin writes `"NONE"` as a string** for HRV status during the multi-week
  onboarding period (`feedbackPhrase: "ONBOARDING_1"`). Those sentinels normalise
  to None rather than displaying as a status.
- **`get_race_predictions(start, end, "daily")` returns all-null rows**; the
  no-argument call returns the current prediction. Both are used — history where
  it exists, plus today's standing numbers.
- **HR streams have to be backfilled from the database, not from what was new in
  this run.** A sync with `--no-streams`, or a transient Cloudflare 504 on one
  activity (which happened), otherwise leaves that activity without a stream
  permanently. `--stream-limit` caps the calls per run.
- **`mostRecentTrainingStatus` and the load fields were null**, so Training Status
  and Garmin's own acute/chronic load stay empty until the account has more
  history. Not a parsing bug — the trends degrade to None and the planner falls
  back to its own load maths.

- **A hosted container can outlive a deploy.** Three times, a commit that added a
  function to a `core` module and imported it at the top of the app brought the
  hosted dashboard down with `ImportError` on a name that was plainly in the
  checked-out source: Streamlit Cloud had pulled the new revision and re-executed
  the app script while `sys.modules` still held the previous revision's modules.
  `app/freshness.py` stamps every app module's source mtime right after import and
  drops the lot on the next run if a file has changed since, so the imports that
  follow re-execute from disk. A cold start pays ten `stat` calls and purges
  nothing.

- **Tests over the guardrails, not just the maths.** `tests/test_guardrails.py`
  feeds `enforce()` a deliberately reckless plan (Z5 intervals during a deload,
  invented plyometrics, three strength days, no rest day, 1485 minutes) and asserts
  every constraint survives. If the guardrails are the product, they need tests.
