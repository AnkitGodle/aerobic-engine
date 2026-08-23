# Deploying Aerobic Engine to the web

Goal: reach the dashboard from anywhere, hit **Refresh**, and have it pull from
Garmin — without getting the Garmin account flagged, and without letting a
stranger change the data.

Three things make that safe, and none of them are optional.

| Risk | What is done about it |
| --- | --- |
| Garmin blocks logins from datacenter IPs | The host **never logs in**. You export a session locally and the host resumes it. Password login is disabled in the hosted app. |
| Garmin flags request bursts | Every call goes through a rate guard: pacing, hourly/daily budgets, a circuit breaker that shuts for an hour on a single 429, and one-sync-at-a-time locking. |
| A public URL means anyone can click Refresh | Every write action — sync, log, plan edit — needs a PIN, stored as a salted hash, with lockout after three wrong guesses. |

---

## 1. Set your PIN

```bash
python scripts/set_pin.py
```

Writes `REFRESH_PIN_SALT` and `REFRESH_PIN_HASH` to `.env` and prints them for
your host's secrets. **The PIN itself is stored nowhere** — not in the repo, not
in the database, not in the browser. Losing it means running this again.

## 2. Get storage that survives a restart

Free hosts have ephemeral disks. If the SQLite file disappears, the app re-pulls
months of Garmin history — around 900 requests — which is precisely the traffic
that gets an account flagged. So use managed Postgres.

Create a free database at [Neon](https://neon.tech) or
[Supabase](https://supabase.com), then copy the connection string. Verify it
before deploying:

```bash
DATABASE_URL='postgresql://…?sslmode=require' python -c "
from core.store import Store
s = Store(); print('connected, tables:', s.counts()); s.close()"
```

That creates the schema and prints empty counts. The SQL is written once in
SQLite's dialect and translated for Postgres, so both backends run the same code
path — but **this repo's Postgres path has only been tested via that translation,
not against a live server**, so run the command above and read the output rather
than assuming.

Copy your existing local data up (optional but saves a big first sync):

```bash
python scripts/fetch.py --db "$DATABASE_URL" --metrics-only   # creates schema
# then re-sync into Postgres directly:
DATABASE_URL='postgresql://…' python scripts/fetch.py --days 45
```

## 3. Export your Garmin session

Do this **on your own machine**, on your home connection:

```bash
python scripts/export_tokens.py
```

Copy the long blob it prints. This is what lets the host resume a session instead
of logging in. Treat it like a password — it grants access to your Garmin account
until you change your Garmin password.

The blob expires eventually. When the hosted app says the session is unusable,
re-run this and replace the secret. That is the whole maintenance burden.

## 4. Push to GitHub

```bash
git init && git add -A && git commit -m "Aerobic Engine"
gh repo create aerobic-engine --public --source=. --push
```

Check before pushing that `.env`, `data/*.db`, `.garmin_tokens/` and
`.streamlit/secrets.toml` are all ignored — `.gitignore` covers them, but
`git status` is the proof.

## 5. Deploy on Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → **New app** → your repo.
2. Main file: `app/streamlit_app.py`.
3. **Advanced settings → Secrets**: paste the contents of
   `.streamlit/secrets.toml.example`, filled in with your real values.
4. Deploy.

Keep the app **private** in Streamlit's sharing settings if you would rather the
page not be public at all. If you do share it publicly, also set
`DASHBOARD_PASSWORD` — the PIN protects *writes*, not *reads*, and this is health
data.

### What the hosted app deliberately cannot do

- **Log in to Garmin.** `allow_password_login=False` on the hosted sync path. A
  stale token blob produces a clear error telling you to re-export, rather than
  an SSO attempt from a datacenter.
- **Sync more than once every 30 minutes**, or more than 300 times an hour /
  1500 times a day (the values in the secrets template).
- **Sync twice at once**, however many times Refresh is clicked.
- **Retry after a 429.** One rate-limit response shuts the breaker for an hour,
  and the state lives in the database, so a redeploy does not reset it.

Watch all of that on the **Data** tab, which shows the live request budget.

---

## Keeping the Garmin account safe: the short version

- **Do not schedule frequent syncs.** Once a day is plenty; the watch uploads on
  its own schedule anyway. There is no cron job in this repo on purpose.
- **Do not use `--full` on a schedule.** It walks your whole history.
- **If you ever see a 429, stop.** The breaker already will, and waiting it out is
  the correct response. `python scripts/fetch.py --guard-status` shows the state.
- **Prefer syncing from home** for anything large. The hosted Refresh button is
  for topping up a few new activities, not for backfilling a year.
- The library used here is unofficial and credentials-based. It is fine for
  personal use, and it is your account on the line — which is why the defaults in
  this repo are conservative rather than fast.

## Commercial later

FastAPI + a real frontend + Postgres + real auth, likely on Azure. The analysis,
planner, guard and auth modules carry over unchanged; only `app/streamlit_app.py`
is thrown away.
