"""The AI layer: `plan_week(payload) -> plan`, and nothing else.

This module is transport only. It does not decide anything about training —
validation and constraint enforcement live in `planner.enforce()`, deliberately,
because a prompt is not a guardrail. If the model returns garbage, malformed
JSON, or a week that breaks the envelope, the planner discards it and falls back
to the rules plan.

Backends are selected by the AI_BACKEND env var (anthropic | azure | none) so the
commercial version can move hosts without touching the planner.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Any, Protocol

log = logging.getLogger("iron_coach.ai")

DEFAULT_MODEL = os.getenv("AI_MODEL", "claude-sonnet-5")
MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "2000"))

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
    """Anything that can turn a payload into raw model text."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


class AIUnavailable(RuntimeError):
    """No backend configured, or the backend refused. Caller falls back to rules."""


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise AIUnavailable("ANTHROPIC_API_KEY is not set")

    def complete(self, system: str, user: str) -> str:
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

    def complete(self, system: str, user: str) -> str:
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
      * Subscription usage limits apply, and this is a personal-use arrangement.
        The commercial phase in the spec needs the API.

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

    def complete(self, system: str, user: str) -> str:
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


class NullBackend:
    """Explicitly no AI. `plan_week` raises so the planner uses rules only."""

    name = "none"

    def complete(self, system: str, user: str) -> str:
        raise AIUnavailable("AI_BACKEND=none")


def get_backend(name: str | None = None) -> LLMBackend:
    name = (name or os.getenv("AI_BACKEND", "anthropic")).lower()
    if name == "anthropic":
        return AnthropicBackend()
    if name == "azure":
        return AzureAIFoundryBackend()
    if name in ("claude_cli", "claude-cli", "cli", "subscription"):
        return ClaudeCLIBackend()
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
    raw = backend.complete(SYSTEM_PROMPT, "\n\n".join(parts))
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
