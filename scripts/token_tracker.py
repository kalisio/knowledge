"""
Token usage tracker for LLM API calls (Anthropic Claude & Ollama).

Provides two layers of tracking (Anthropic only — Ollama is free):

1. **Session tracking** — `TrackedClient` records every call in the current
   Python session and can print a session summary.

2. **Persistent ledger** — A shared JSON file (`outputs/token_ledger.json`)
   that accumulates ALL calls across every run, every notebook, every user.
   Survives restarts.  Anyone who clones the project and runs notebooks will
   have their calls appended to the same ledger.

Usage:

    from token_tracker import TrackedClient

    # Default: uses Ollama (set by LLM_PROVIDER env var)
    client = TrackedClient()
    answer = client.ask(prompt)

    # Explicit Anthropic usage (with token tracking)
    client = TrackedClient(provider="anthropic")
    answer = client.ask(prompt)
    client.summary()
    client.ledger_summary()
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Provider defaults ──
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.109:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ── Pricing (USD per million tokens, as of 2025-05) ──
MODEL_PRICING = {
    "claude-opus-4-6":           {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":         {"input": 3.0,   "output": 15.0},
    "claude-sonnet-4-20250514":  {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.0},
}

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Persistent ledger lives in outputs/ (which is .gitignored)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER_PATH = _PROJECT_ROOT / "outputs" / "token_ledger.json"


@dataclass
class CallRecord:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float
    source: str          # which notebook / script produced this call
    prompt_preview: str


class TokenLedger:
    """Append-only persistent ledger that accumulates token usage to a JSON file."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    # ── Read / write ──
    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                return self._empty()
        return self._empty()

    @staticmethod
    def _empty() -> dict:
        return {
            "cumulative": {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            },
            "sessions": [],
        }

    def _flush(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Append a finished session ──
    def record_session(self, source: str, calls: list[CallRecord]) -> None:
        if not calls:
            return

        session_input = sum(c.input_tokens for c in calls)
        session_output = sum(c.output_tokens for c in calls)
        session_cost = sum(c.cost_usd for c in calls)

        self._data["sessions"].append({
            "source": source,
            "started_at": calls[0].timestamp,
            "finished_at": calls[-1].timestamp,
            "calls": len(calls),
            "input_tokens": session_input,
            "output_tokens": session_output,
            "total_tokens": session_input + session_output,
            "cost_usd": round(session_cost, 6),
            "detail": [
                {
                    "timestamp": c.timestamp,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": round(c.cost_usd, 6),
                    "duration_s": c.duration_s,
                    "prompt_preview": c.prompt_preview,
                }
                for c in calls
            ],
        })

        cum = self._data["cumulative"]
        cum["total_calls"] += len(calls)
        cum["total_input_tokens"] += session_input
        cum["total_output_tokens"] += session_output
        cum["total_tokens"] += session_input + session_output
        cum["total_cost_usd"] = round(cum["total_cost_usd"] + session_cost, 6)

        self._flush()

    # ── Query ──
    @property
    def cumulative(self) -> dict:
        return self._data["cumulative"]

    @property
    def sessions(self) -> list[dict]:
        return self._data["sessions"]

    def summary(self) -> str:
        c = self.cumulative
        n_sessions = len(self.sessions)
        lines = [
            "",
            "All-Time Token Usage (persistent ledger)",
            "-" * 42,
            f"  Sessions:        {n_sessions:>10}",
            f"  Total calls:     {c['total_calls']:>10,}",
            f"  Input tokens:    {c['total_input_tokens']:>10,}",
            f"  Output tokens:   {c['total_output_tokens']:>10,}",
            f"  Total tokens:    {c['total_tokens']:>10,}",
            f"  Total cost:      ${c['total_cost_usd']:>9.4f}",
            f"  Ledger file:     {self.path}",
        ]
        text = "\n".join(lines)
        print(text)
        return text


# ── Global ledger singleton ──
_ledger: TokenLedger | None = None


def get_ledger(path: str | Path = DEFAULT_LEDGER_PATH) -> TokenLedger:
    global _ledger
    if _ledger is None or _ledger.path != Path(path):
        _ledger = TokenLedger(path)
    return _ledger


class TrackedClient:
    """LLM client that supports both Anthropic Claude and Ollama backends.

    - provider="anthropic": tracks per-session token usage + persistent ledger.
    - provider="ollama": no token tracking (free local model).
    """

    def __init__(
        self,
        model: str | None = None,
        source: str = "",
        provider: str | None = None,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        **client_kwargs,
    ):
        self.provider = provider or DEFAULT_PROVIDER
        self.source = source
        self.ledger_path = Path(ledger_path)
        self.calls: list[CallRecord] = []

        if self.provider == "anthropic":
            import anthropic
            self.model = model or DEFAULT_ANTHROPIC_MODEL
            self._client = anthropic.Anthropic(**client_kwargs)
        else:
            self.model = model or DEFAULT_OLLAMA_MODEL
            self._ollama_url = OLLAMA_BASE_URL

    # ── Core call method ──
    def ask(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a single-turn message and return the text response."""
        model = model or self.model
        if self.provider == "ollama":
            return self._ask_ollama(prompt, system=system, model=model, max_tokens=max_tokens)
        return self._ask_anthropic(prompt, system=system, model=model, max_tokens=max_tokens)

    # ── Anthropic backend ──
    def _ask_anthropic(
        self, prompt: str, *, system: str | None, model: str, max_tokens: int,
    ) -> str:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        t0 = time.time()
        message = self._client.messages.create(**kwargs)
        duration = time.time() - t0

        usage = message.usage
        cost = self._compute_cost(model, usage.input_tokens, usage.output_tokens)

        self.calls.append(CallRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            duration_s=round(duration, 3),
            source=self.source,
            prompt_preview=prompt[:120].replace("\n", " "),
        ))

        return message.content[0].text.strip()

    # ── Ollama backend ──
    def _ask_ollama(
        self, prompt: str, *, system: str | None, model: str, max_tokens: int,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        resp = requests.post(
            f"{self._ollama_url}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        return data["message"]["content"].strip()

    # ── Session aggregate properties ──
    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    # ── Session summary ──
    def summary(self) -> str:
        """Print session-level stats (this run only)."""
        if self.provider == "ollama":
            print(f"\nSession: {self.source or '(unnamed)'} — provider=ollama (no token tracking)")
            return ""
        lines = [
            "",
            "Session Token Usage (this run)",
            "-" * 34,
            f"  Source:           {self.source or '(unnamed)':>10}",
            f"  Calls:           {len(self.calls):>10,}",
            f"  Input tokens:    {self.total_input_tokens:>10,}",
            f"  Output tokens:   {self.total_output_tokens:>10,}",
            f"  Total tokens:    {self.total_tokens:>10,}",
            f"  Total cost:      ${self.total_cost_usd:>9.4f}",
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ── Persist: session → ledger ──
    def save(self, session_path: str | Path | None = None) -> None:
        """Append this session to the persistent ledger (Anthropic only)."""
        if self.provider == "ollama":
            return

        # 1. Append to global ledger
        ledger = get_ledger(self.ledger_path)
        ledger.record_session(self.source, self.calls)

        # 2. Optionally write a standalone session file
        if session_path is not None:
            data = {
                "source": self.source,
                "summary": {
                    "total_calls": len(self.calls),
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_tokens,
                    "total_cost_usd": round(self.total_cost_usd, 6),
                },
                "calls": [
                    {
                        "timestamp": c.timestamp,
                        "model": c.model,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                        "cost_usd": round(c.cost_usd, 6),
                        "duration_s": c.duration_s,
                        "prompt_preview": c.prompt_preview,
                    }
                    for c in self.calls
                ],
            }
            Path(session_path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(f"Session detail saved to {session_path}")

        print(f"Ledger updated: {self.ledger_path}")

    def ledger_summary(self) -> str:
        """Print all-time cumulative stats from the persistent ledger."""
        if self.provider == "ollama":
            return ""
        return get_ledger(self.ledger_path).summary()

    # ── Internal ──
    @staticmethod
    def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
