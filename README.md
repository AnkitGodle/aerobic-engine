# Aerobic Engine

**Training analytics and adaptive planning for endurance athletes, built on your
own Garmin data.** It answers one question properly — *am I getting faster at the
same heart rate?* — and then plans the rest of the week around the answer.

Rules first, AI second. A language model adjusts volume, intensity and placement;
it cannot talk its way past a deload, invent an exercise, or exceed the weekly
progression cap, because those are enforced in code after the model answers.

> Personal project, single user. Not affiliated with or endorsed by Garmin.
> Not medical advice.

---

## What it takes as input

**Garmin Connect**, via the unofficial [`garminconnect`](https://github.com/cyberjunky/python-garminconnect)
library, pulled with your own credentials into a local database:

- **Activities** — run, bike, pool and open-water swim, strength, and multisport
  (split into its individual legs, because the parent record carries no heart rate)
- **Heart-rate streams** per session, downsampled, so aerobic drift can be
  recomputed without re-fetching
- **Time in each heart-rate zone** per session — the strongest available signal
  for whether a session was actually easy
- **Daily wellness** — resting heart rate, overnight HRV, VO2max, Training
  Readiness, sleep, body battery
- **Thresholds** — lactate-threshold heart rate, running and cycling FTP
- **Strength sets** recorded in the watch's strength mode, mapped into the app's
  exercise library and logged automatically

Plus your own input: a daily check-in (sleep, soreness, motivation, time
available, free text) and weekly targets per sport.

## What it tells you

- **Efficiency factor** per sport — speed or watts per heartbeat — trended over
  *steady aerobic sessions only*, because mixing in intervals makes the trend
  track how hard you trained rather than how fit you are
- **Aerobic drift** on long sessions: efficiency in the second half versus the first
- **Where your effort actually goes** — the easy/moderate/hard split, against a
  base-phase target of 70%+ easy
- **Whether the engine is improving** — resting HR and HRV baselines, 28 days
  against the 28 before
- **A plan for the rest of the week** that responds to recovery data, and a
  plain-English summary of every page so you needn't read the charts

## Design

Three layers, in this order:

1. **Facts** — deterministic analysis of what you actually did (`core/analysis.py`)
2. **Rules envelope** — the non-negotiables: 10%/week volume cap, deload every
   fourth week or whenever recovery says so, session counts, required long
   sessions, minimum rest days, and where leg strength may be placed
   (`core/planner.py`)
3. **The AI** — adjusts inside that envelope and explains each choice
   (`core/ai.py`)

Then `planner.enforce()` re-checks every constraint **in code**. A prompt is not a
guardrail: a raw model is sycophantic — say "I feel great" and it hands you a
reckless week. `tests/test_guardrails.py` feeds the enforcement layer a
deliberately reckless plan (Z5 intervals during a deload, invented plyometrics,
three leg days, no rest day, 1485 minutes against a 229-minute budget) and asserts
that none of it survives.

Nothing under `core/` imports Streamlit, and only `core/ai.py` knows a language
model exists — so the analysis and planner carry over unchanged if the UI is ever
replaced.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your Garmin credentials
python scripts/set_pin.py     # PIN for write actions
python scripts/fetch.py --days 45
streamlit run app/streamlit_app.py
```

The fetch needs a Garmin login and runs locally. The dashboard is read-only
against the database.

| Flag | |
| --- | --- |
| `--days N` | re-scan the last N days |
| `--full` | all available history |
| `--metrics-only` | recompute efficiency/zones/drift, no network |
| `--refresh-wellness` | re-pull wellness already stored |
| `--guard-status` | show request budgets and breaker state |

## The AI layer is optional, and can be free

`AI_BACKEND=none` gives the rules-only plan, which is a complete, usable week on
its own. For the adaptive layer, the written page summaries and the per-chart
readings, the cheapest good option is **Groq's free tier** — no card, and
`openai/gpt-oss-120b` is a capable 131K-context model with guaranteed JSON output:

```bash
# key from https://console.groq.com/keys
AI_BACKEND=groq
GROQ_API_KEY=gsk_...
```

The free plan's binding limit is **8,000 tokens per minute** (plus 30 requests a
minute, 1,000 a day). A planner call is around 2,500 tokens, so that is
comfortable — summaries and chart captions are cached for 30–60 minutes so a page
reload does not spend the budget, and a 429 falls back to the rules plan rather
than retrying into the next minute's allowance.

Other backends behind the same interface: `anthropic` (API key), `claude_cli` (a
Claude Pro/Max subscription via the Claude Code CLI — local only, since a hosted
app has no CLI to call), and `azure` (AI Foundry).

## Being a good citizen with an unofficial API

Garmin publishes no rate limits, is stricter with datacenter IPs than home ones,
and does lock accounts. Every request therefore goes through `core/garmin_guard.py`:

- **Pacing and budgets** — a minimum gap between calls, hourly and daily ceilings
- **A circuit breaker** — one 429 stops everything for an hour; `Retry-After` is
  honoured only when it asks for *longer*
- **Single-flight** — one sync at a time plus a cooldown, so a Refresh button
  cannot be double-clicked into a burst
- **No SSO from a host** — a deployed instance resumes an exported session and
  runs with password login disabled

All of that state lives in the database, so a restart or a redeploy does not reset
it. 12 tests cover it.

## Security

Writes are PIN-gated (salted PBKDF2-SHA256, constant-time comparison, lockout with
exponential backoff persisted server-side); the PIN itself is never stored.
Credentials, tokens and the database are gitignored and never leave the machine
that fetches. See [DEPLOY.md](DEPLOY.md).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

60 tests: guardrails, analysis maths, the rate guard, PIN security, and the
SQLite→Postgres dialect translation.

## Documentation

- [CLAUDE.md](CLAUDE.md) — the full build brief and design rationale, including
  everything the first real Garmin sync got wrong
- [DEPLOY.md](DEPLOY.md) — hosting it, with the account-safety measures explained
