"""
Token usage tracker for Anthropic Claude API calls.

Provides two layers of tracking:

1. **Session tracking** — `TrackedClient` records every call in the current
   Python session and can print a session summary.

2. **Persistent ledger** — A shared JSON file (`outputs/token_ledger.json`)
   that accumulates ALL calls across every run, every notebook, every user.
   Survives restarts.  Anyone who clones the project and runs notebooks will
   have their calls appended to the same ledger.

Usage:

    from token_tracker import TrackedClient

    client = TrackedClient()           # auto-loads the persistent ledger
    answer = client.ask(prompt)        # recorded in session + ledger

    client.summary()                   # session stats
    client.ledger_summary()            # all-time stats (across all runs)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Pricing (USD per million tokens, as of 2025-05) ──
MODEL_PRICING = {
    "claude-opus-4-6":           {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":         {"input": 3.0,   "output": 15.0},
    "claude-sonnet-4-20250514":  {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.0},
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"

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
            "╔═══════════════════════════════════════════════════╗",
            "║       All-Time Token Usage (persistent ledger)    ║",
            "╠═══════════════════════════════════════════════════╣",
           f"║  Sessions:      {n_sessions:>10}                  ║",
           f"║  Total calls:   {c['total_calls']:>10,}           ║",
           f"║  Input tokens:  {c['total_input_tokens']:>10,}    ║",
           f"║  Output tokens: {c['total_output_tokens']:>10,}   ║",
           f"║  Total tokens:  {c['total_tokens']:>10,}          ║",
           f"║  Total cost:     ${c['total_cost_usd']:>9.4f}     ║",
            "╠═══════════════════════════════════════════════════╣",
           f"║  Ledger file: {str(self.path):<37}                ║",
            "╚═══════════════════════════════════════════════════╝",
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
    """Thin wrapper around anthropic.Anthropic that records token usage.

    - Tracks per-session usage (this Python process).
    - Automatically appends to a persistent ledger on `save()`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        source: str = "",
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        **client_kwargs,
    ):
        self._client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.source = source
        self.ledger_path = Path(ledger_path)
        self.calls: list[CallRecord] = []

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
        lines = [
            "",
            "╔═══════════════════════════════════════════════════╗",
            "║       Session Token Usage (this run)              ║",
            "╠═══════════════════════════════════════════════════╣",
            f"║  Source:        {self.source or '(unnamed)':>10}                      ║",
            f"║  Calls:         {len(self.calls):>10,}                      ║",
            f"║  Input tokens:  {self.total_input_tokens:>10,}                      ║",
            f"║  Output tokens: {self.total_output_tokens:>10,}                      ║",
            f"║  Total tokens:  {self.total_tokens:>10,}                      ║",
            f"║  Total cost:     ${self.total_cost_usd:>9.4f}                      ║",
            "╚═══════════════════════════════════════════════════╝",
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ── Persist: session → ledger ──
    def save(self, session_path: str | Path | None = None) -> None:
        """Append this session to the persistent ledger.

        Optionally also write a standalone session file to *session_path*.
        """
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
        return get_ledger(self.ledger_path).summary()

    # ── Internal ──
    @staticmethod
    def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
