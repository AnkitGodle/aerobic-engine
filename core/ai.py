"""The AI layer: `plan_week(payload) -> plan`, and nothing else.

This module is transport only. It does not decide anything about training —
validation and constraint enforcement live in `planner.enforce()`, deliberately,
because a prompt is not a guardrail. If the model returns garbage, malformed
JSON, or a week that breaks the envelope, the planner discards it and falls back
to the rules plan.

Backends are selected by the AI_BACKEND env var so the provider can change —
or be switched off entirely — without touching the planner.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Protocol

log = logging.getLogger("aerobic_engine.ai")

DEFAULT_MODEL = os.getenv("AI_MODEL", "claude-sonnet-5")
USER_AGENT = os.getenv("AI_USER_AGENT", "aerobic-engine/1.0")
# Generous on purpose. Gemini 3.x and the gpt-oss models spend tokens on internal
# reasoning before they emit anything, and that spend counts against max_tokens:
# a 1-token answer measured 113 total tokens in testing. Too small a ceiling
# truncates the plan JSON mid-object, which fails to parse and silently costs the
# athlete the AI layer.
MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "6000"))
RETRY_ATTEMPTS = int(os.getenv("AI_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_S = float(os.getenv("AI_RETRY_BACKOFF", "1.5"))

SYSTEM_PROMPT = """\
You are the adaptive-planning layer of a personal Iron Man training app. The \
athlete trains swim, bike, run and leg strength, and wears a Garmin Forerunner \
265. You are ONE layer of three: deterministic analysis produced the facts, \
deterministic rules produced the envelope, and you adjust inside that envelope.

Your job: take the remaining days of this week and set volume, intensity and \
placement, then explain each choice in one short line.

Hard limits — the calling code enforces every one of these, so breaking them \
just gets your output discarded:
- Plan ONLY the days listed in envelope.days_remaining. Never touch a past day.
- Total planned minutes for the remaining days must not exceed \
envelope.remaining_minutes_budget.
- Respect envelope.by_sport min/max session counts and max_minutes.
- If envelope.deload is true, prescribe Z1-Z2 only, no intervals, no brick, and \
stay well under the budget. You may not argue with the deload flag; it comes \
from recovery data, not from mood.
- Strength days: sport "strength", and exercise_ids chosen ONLY from \
strength_state.allowed_exercise_ids. Never invent an exercise. Never prescribe \
plyometrics or jumping.
- Keep at least envelope.min_rest_days full rest days in the week (sport "rest", \
duration_min 0).
- Do not schedule leg strength on the same day before a long run or a quality \
bike, and do not schedule it the day before a long run.

Read `history.intensity_distribution` before anything else. It is the share of \
the last 28 days spent easy, moderate and hard, against a base-phase target of \
70%+ easy and under 15% hard. If its verdict is "too_hard", the athlete's problem \
is not volume — it is that their easy sessions are not easy. In that case:
- Prescribe Z2 on the swim, bike and run sessions, and say in "why" what that
  actually feels like — once, on one session, not on every row. Never put zone
  or pace language on a strength row: its target_zone is "n/a".
- Do NOT add volume to fix it. Fixing the mix comes first.
- Raise it in flags once, not on every session.

`history.training_heart_rate` gives, per sport, the change in heart rate at the \
same pace. Negative is progress. If it is positive while intensity is high, the \
athlete is digging a hole: hold volume flat rather than adding.

`history.recent_sessions` is the last dozen sessions with their durations, heart \
rates and whether each counted as steady aerobic work. Use it. Referring to what \
actually happened — "your last two runs both sat above threshold" — is worth more \
than a generic instruction, and `why_not_steady` tells you exactly why a session \
was excluded from the fitness trend.

`envelope.heart_rate_zones_bpm` gives this athlete's real zone boundaries. You \
may mention them in "why" (for example "keep it under 130"), but do NOT put a \
heart-rate range in any other field: the exact target is attached in code after \
you answer, so it is always correct.

`envelope.sports_switched_off` lists sports the athlete has turned off. Never \
schedule them, and do not comment on their absence.

Judgement you DO own, within those limits:
- Shifting sessions between the remaining days to fit the athlete's stated time \
and how they feel.
- Trimming duration or dropping intensity when the check-in reports poor sleep, \
soreness or low motivation; nudging up when they report feeling strong AND the \
recovery signals agree.
- Choosing which optional session to cut when time is short. Protect, in order: \
the long ride, the long run, then one strength session, then swims.
- Reading the recent check-in history for repeated complaints and responding to \
the pattern, not just today.

Tone: terse and concrete. "why" is one clause, e.g. "sleep 2/5 and RHR +6, so \
easy spin instead of intervals". Do not lecture, do not cite studies, do not \
invent training science, do not give medical advice.

Return ONE JSON object and no other text:
{"week_plan":[{"day":"Mon","sport":"bike","duration_min":90,\
"target_zone":"Z2","purpose":"aerobic base","exercise_ids":[],\
"why":"one-line reason"}],"flags":["..."],"adjustments_made":["..."]}
sport is one of: swim, bike, run, strength, brick, rest.
target_zone is one of: Z1, Z2, Z3, Z4, Z5, mixed, technique, n/a.
"""


class LLMBackend(Protocol):
    """Anything that can turn a payload into raw model text.

    `json_mode` asks the provider to guarantee a JSON object where it can. The
    planner needs it; the prose summaries must not have it.
    """

    name: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class AIUnavailable(RuntimeError):
    """No backend configured, or the backend refused. Caller falls back to rules.

    `retryable` marks the transient cases — a busy model, a dropped connection —
    that are worth one more attempt. Rate limits and auth failures are not.
    """

    retryable = False
    advance_model = False
    retry_after: float | None = None


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise AIUnavailable("ANTHROPIC_API_KEY is not set")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise AIUnavailable("anthropic package not installed") from exc

        client = Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )


class AzureAIFoundryBackend:
    """Claude on Azure AI Foundry — same messages shape, different host."""

    name = "azure"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
    ) -> None:
        self.endpoint = (endpoint or os.getenv("AZURE_AI_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.getenv("AZURE_AI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_AI_DEPLOYMENT", DEFAULT_MODEL)
        if not self.endpoint or not self.api_key:
            raise AIUnavailable("AZURE_AI_ENDPOINT / AZURE_AI_API_KEY are not set")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self.deployment,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/anthropic/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise AIUnavailable(f"Azure AI Foundry call failed: {exc}") from exc
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )


class ClaudeCLIBackend:
    """Drive the Claude Code CLI in headless mode, using its own login.

    A Claude Pro/Max subscription is not API credit — the Messages API bills
    separately and there is no key to point the SDK at. The CLI, though, is
    already authenticated against the subscription and has a print mode built
    for scripting, so shelling out to it is a legitimate way to use one.

    Two consequences worth knowing before choosing this backend:

      * It only works where the CLI is installed and logged in, which puts the
        planning step in the same box as the Garmin fetch: local only, never the
        hosted dashboard. The dashboard stays read-only over SQLite.
      * Subscription usage limits apply, and this is a personal-use
        arrangement.

    Flags are kept to the two most stable ones (`-p`, `--output-format json`);
    anything else goes in CLAUDE_CLI_EXTRA_ARGS so a CLI update cannot break
    this by renaming a flag we hard-coded.
    """

    name = "claude_cli"

    def __init__(self, binary: str | None = None, model: str | None = None) -> None:
        self.binary = binary or os.getenv("CLAUDE_CLI_BIN", "claude")
        self.model = model or os.getenv("CLAUDE_CLI_MODEL", "")
        self.extra_args = shlex.split(os.getenv("CLAUDE_CLI_EXTRA_ARGS", ""))
        self.timeout = int(os.getenv("CLAUDE_CLI_TIMEOUT", "180"))
        if shutil.which(self.binary) is None:
            raise AIUnavailable(
                f"{self.binary!r} is not on PATH. Install it with "
                "`npm install -g @anthropic-ai/claude-code`, run `claude` once to "
                "log in, then retry."
            )

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        # The CLI's system-prompt flags have moved between versions, so the
        # instructions ride along in the prompt itself.
        prompt = f"{system}\n\n---\n\n{user}"
        cmd = [self.binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        cmd += self.extra_args
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except FileNotFoundError as exc:
            raise AIUnavailable(f"{self.binary!r} disappeared from PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise AIUnavailable(f"{self.binary} timed out after {self.timeout}s") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:300]
            raise AIUnavailable(f"{self.binary} exited {proc.returncode}: {detail}")

        out = (proc.stdout or "").strip()
        if not out:
            raise AIUnavailable(f"{self.binary} returned nothing")
        # --output-format json wraps the answer; older builds print it bare.
        try:
            envelope = json.loads(out)
        except json.JSONDecodeError:
            return out
        if isinstance(envelope, dict):
            for key in ("result", "text", "content", "response"):
                value = envelope.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return out


# Providers that speak the OpenAI chat-completions dialect. Base URLs and model
# IDs below were checked against each provider's own docs; free-tier limits move,
# so treat the notes as a starting point rather than a guarantee.
#
# The practical difference for this app is the *shape* of the free allowance. A
# planner call is a chunky ~2.5K tokens, so a provider that caps tokens-per-minute
# pinches, while one that caps requests-per-day does not.
OPENAI_COMPAT: dict[str, dict[str, Any]] = {
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "fallbacks": ("openai/gpt-oss-20b",),
        "keys": ("GROQ_API_KEY",),
        "console": "https://console.groq.com/keys",
        "free": "30 req/min, 1000/day, 8K tokens/min — the token cap pinches",
        "json": "json_object",
    },
    "gemini": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        # Not the newest on purpose. Probed live: gemini-3.7-flash and
        # gemini-flash-latest both returned 503 (oversubscribed), and
        # gemini-2.5-flash is retired (404). 3.6-flash answered in 3s with JSON
        # mode working, so it leads, with faster siblings behind it.
        "model": "gemini-3.6-flash",
        "fallbacks": ("gemini-3.5-flash", "gemini-3.5-flash-lite"),
        "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "console": "https://aistudio.google.com/apikey",
        "free": "~1500 requests/day, 1M context, no card — most generous for "
                "chunky calls like ours",
        "json": "json_object",
    },
    "cerebras": {
        "base": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
        "keys": ("CEREBRAS_API_KEY",),
        "console": "https://cloud.cerebras.ai",
        "free": "~1M tokens/day — the largest token budget of the three",
        "json": "json_schema",
    },
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "keys": ("OPENROUTER_API_KEY",),
        "console": "https://openrouter.ai/keys",
        "free": "many models tagged :free behind one key; limits vary per model",
        "json": "json_object",
    },
}


class OpenAICompatBackend:
    """One backend for every provider that speaks OpenAI chat-completions.

    Groq, Google's Gemini, Cerebras and OpenRouter all expose the same dialect,
    so switching provider is a base URL, a key and a model name rather than new
    code. `AI_BASE_URL` lets an unlisted provider work without touching this
    file at all.

    `json_mode` asks for a guaranteed JSON object where the provider supports it,
    which stops a stray sentence of preamble from invalidating a week's plan.
    """

    def __init__(
        self,
        provider: str = "groq",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        spec = OPENAI_COMPAT.get(provider, {})
        self.name = provider
        self.spec = spec
        self.base = (base_url or os.getenv("AI_BASE_URL") or spec.get("base", "")).rstrip("/")
        self.model = model or os.getenv("AI_MODEL_OVERRIDE") or os.getenv(
            f"{provider.upper()}_MODEL", spec.get("model", "")
        )
        self.timeout = int(os.getenv("AI_TIMEOUT", "90"))
        # An explicit model choice is honoured alone; the default carries a chain,
        # because the newest model is reliably the busiest one.
        explicit = model or os.getenv("AI_MODEL_OVERRIDE") or os.getenv(
            f"{provider.upper()}_MODEL")
        self.models = ([explicit] if explicit
                       else [self.model, *spec.get("fallbacks", ())])
        self.api_key = api_key or os.getenv("AI_API_KEY") or next(
            (os.getenv(k) for k in spec.get("keys", ()) if os.getenv(k)), None
        )
        if not self.base:
            raise AIUnavailable(
                f"No base URL for provider {provider!r}. Set AI_BASE_URL, or use "
                f"one of: {', '.join(sorted(OPENAI_COMPAT))}."
            )
        if not self.api_key:
            wanted = " or ".join(spec.get("keys", ("AI_API_KEY",)))
            console = spec.get("console", "")
            raise AIUnavailable(
                f"{wanted} is not set."
                + (f" Free key at {console}" if console else "")
            )

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Send the request, retrying only what is worth retrying.

        A 503 from Gemini means the model is momentarily oversubscribed, not that
        anything is wrong — losing the AI layer over a blip is a poor trade. Rate
        limits and auth failures are never retried: a 429 retry just spends the
        next minute's budget, and a bad key will not improve.
        """
        last: Exception | None = None
        for model in self.models:
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    return self._post(system, user, json_mode, model=model)
                except AIUnavailable as exc:
                    last = exc
                    if getattr(exc, "advance_model", False):
                        break        # this model is metered out; try the next
                    if not getattr(exc, "retryable", False):
                        raise
                    if attempt < RETRY_ATTEMPTS - 1:
                        time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            if model != self.models[-1]:
                log.info("%s: %s unavailable, trying the next model", self.name, model)
        raise last if last else AIUnavailable(f"{self.name} failed")

    def _post(self, system: str, user: str, json_mode: bool = False,
              model: str | None = None) -> str:
        import urllib.error
        import urllib.request

        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.4,
        }
        if json_mode and self.spec.get("json"):
            body["response_format"] = {"type": "json_object"}
            body["stream"] = False   # Cerebras rejects JSON mode with streaming

        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Required, not cosmetic: urllib's default "Python-urllib/x.y"
                # is blocked by Groq's Cloudflare with error 1010, which surfaces
                # as a 403 and looks exactly like a bad API key.
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = (json.loads(exc.read()).get("error") or {}).get("message", "")
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 429:
                err = AIUnavailable(
                    f"{self.name} rate limit reached on {body['model']}. "
                    f"{detail}".strip())
                # Providers meter per model, so the next model in the chain has
                # its own bucket — worth trying. Retrying the SAME model is not:
                # that just spends the next window too.
                err.advance_model = True
                err.retry_after = _retry_after_seconds(detail)
                raise err from exc
            if exc.code == 404:
                err = AIUnavailable(
                    f"{self.name}: model {body['model']!r} not available. "
                    f"{detail}".strip())
                err.retryable = True   # so the chain moves on to the next model
                raise err from exc
            if exc.code in (500, 502, 503, 504):
                err = AIUnavailable(
                    f"{self.name} is busy ({exc.code}). {detail}".strip())
                err.retryable = True
                raise err from exc
            if exc.code in (401, 403):
                raise AIUnavailable(
                    f"{self.name} rejected the key ({exc.code}). {detail}".strip()
                ) from exc
            raise AIUnavailable(
                f"{self.name} returned {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            err = AIUnavailable(f"{self.name} unreachable: {exc.reason}")
            err.retryable = True
            raise err from exc

        choices = data.get("choices") or []
        if not choices:
            raise AIUnavailable(f"{self.name} returned no choices")
        finish = choices[0].get("finish_reason")
        if finish == "length" and not (choices[0].get("message") or {}).get("content"):
            raise AIUnavailable(
                f"{self.name} spent its whole {MAX_TOKENS}-token budget on internal "
                f"reasoning and returned nothing. Raise AI_MAX_TOKENS."
            )
        return (choices[0].get("message") or {}).get("content") or ""


class NullBackend:
    """Explicitly no AI. `plan_week` raises so the planner uses rules only."""

    name = "none"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        raise AIUnavailable("AI_BACKEND=none")


def _retry_after_seconds(detail: str) -> float | None:
    """Google puts "Please retry in 16.528294029s" in the 429 body. Use it —
    guessing a backoff when the provider has told you the number is silly."""
    m = re.search(r"retry in ([\d.]+)\s*s", detail or "", re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def get_backend(name: str | None = None) -> LLMBackend:
    name = (name or os.getenv("AI_BACKEND", "anthropic")).lower()
    if name == "anthropic":
        return AnthropicBackend()
    if name == "azure":
        return AzureAIFoundryBackend()
    if name in ("claude_cli", "claude-cli", "cli", "subscription"):
        return ClaudeCLIBackend()
    if name in OPENAI_COMPAT:
        return OpenAICompatBackend(name)
    if name == "openai_compat":
        # Any provider not listed above: set AI_BASE_URL, AI_API_KEY, AI_MODEL_OVERRIDE.
        return OpenAICompatBackend("openai_compat")
    if name in ("none", "off", ""):
        return NullBackend()
    raise AIUnavailable(f"Unknown AI_BACKEND {name!r}")


def available(name: str | None = None) -> bool:
    try:
        get_backend(name)
        return True
    except AIUnavailable:
        return False


def plan_week(
    payload: dict[str, Any],
    backend: LLMBackend | None = None,
    user_pushback: str | None = None,
    previous_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one interface the planner calls. Returns a parsed (unvalidated) dict.

    Raises AIUnavailable when there is no usable backend or no parseable JSON —
    the planner treats that as "use the rules plan".
    """
    backend = backend or get_backend()
    parts = [json.dumps(payload, indent=2, default=str)]
    if previous_plan:
        parts.append(
            "The plan you produced last time:\n"
            + json.dumps(previous_plan, indent=2, default=str)
        )
    if user_pushback:
        parts.append(
            "The athlete pushed back on that plan, in their own words:\n"
            f"{user_pushback}\n"
            "Revise the remaining days accordingly, still inside the envelope. "
            "If what they are asking for breaks a hard limit, keep the limit and "
            "say so in flags."
        )
    raw = backend.complete(SYSTEM_PROMPT, "\n\n".join(parts), json_mode=True)
    parsed = extract_json(raw)
    if parsed is None:
        raise AIUnavailable("model returned no parseable JSON")
    return parsed


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a response, tolerating fences and stray prose."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                start = None
    log.warning("Could not extract JSON from model output: %.200s", text)
    return None
