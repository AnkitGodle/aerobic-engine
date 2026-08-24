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

Create a free database at [Neon](https://neon.tech) (this project uses a Neon
project in `ap-southeast-1`) or [Supabase](https://supabase.com), then copy the
connection string into `.env` as `DATABASE_URL`. It takes precedence over
`AEROBIC_ENGINE_DB`, so the local sync and the hosted dashboard end up reading
and writing the same rows instead of drifting apart.

Verify it, which also creates the schema:

```bash
DATABASE_URL='postgresql://…?sslmode=require' python -c "
from core.store import Store
s = Store(); print('connected:', s.counts()); s.close()"
```

The SQL is written once in SQLite's dialect and translated on the way out, so
both backends run the same code path. That translation has now been exercised
against a live Neon server — all 13 tables built on the first attempt — rather
than only against the dialect tests.

Then copy the data you already have up, which costs **zero Garmin calls**:

```bash
python scripts/migrate_to_postgres.py            # uses DATABASE_URL
python scripts/migrate_to_postgres.py --dry-run  # report first, write nothing
```

It prints per-table counts and then the destination totals, so a short copy is
visible rather than assumed. Re-running is safe: every table upserts on its real
key, and the id sequences are pushed past the copied rows so the next insert does
not collide.

Do **not** reach for `fetch.py --days 45` to populate a fresh hosted database.
That re-pulls history from Garmin — hundreds of requests, from a datacenter IP,
which is the exact pattern the rate guard exists to prevent.

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
a read PIN with `python scripts/set_pin.py --read` — the write PIN protects
*changes*, not *reading*, and this is health
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
