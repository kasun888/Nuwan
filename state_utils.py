from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytz

# DATA_DIR is the single source of truth — defined in config_loader.py.
# Importing here avoids the duplicate Path(os.environ.get(...)) definition
# that previously existed in both modules.
from config_loader import DATA_DIR

logger = logging.getLogger(__name__)

SG_TZ = pytz.timezone('Asia/Singapore')

CALENDAR_CACHE_FILE = DATA_DIR / 'calendar_cache.json'
SCORE_CACHE_FILE    = DATA_DIR / 'signal_cache.json'
OPS_STATE_FILE      = DATA_DIR / 'ops_state.json'
TRADE_HISTORY_FILE  = DATA_DIR / 'trade_history.json'
RUNTIME_STATE_FILE  = DATA_DIR / 'runtime_state.json'
# Removed: TRADE_HISTORY_ARCHIVE_FILE — archival removed; 90-day rolling window is sufficient
# Removed: LAST_TRADE_CANDLE_FILE     — never used anywhere in the codebase


def load_json(path: Path, default: Any):
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(default, dict) and not isinstance(data, dict):
                    return default.copy()
                if isinstance(default, list) and not isinstance(data, list):
                    return default.copy()
                return data
    except Exception as exc:
        logger.warning('Failed to load %s: %s', path, exc)
    return default.copy() if isinstance(default, (dict, list)) else default


def save_json(path: Path, data: Any):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile('w', delete=False, dir=str(path.parent), encoding='utf-8') as tmp:
            json.dump(data, tmp, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, path)
    except Exception as exc:
        logger.warning('Failed to save %s: %s', path, exc)


def update_runtime_state(**kwargs) -> None:
    state = load_json(RUNTIME_STATE_FILE, {})
    if not isinstance(state, dict):
        state = {}
    state.update(kwargs)
    state['updated_at_sgt'] = datetime.now(SG_TZ).strftime('%Y-%m-%d %H:%M:%S')
    save_json(RUNTIME_STATE_FILE, state)


def parse_sgt_timestamp(value: str | None) -> datetime | None:
    """Parse a SGT timestamp string into a timezone-aware datetime.

    Accepts both '%Y-%m-%d %H:%M:%S' and ISO '%Y-%m-%dT%H:%M:%S' formats.
    Returns None if value is falsy or unparseable.

    Canonical implementation — imported by bot.py and calendar_fetcher.py
    so the logic lives in exactly one place.
    """
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return SG_TZ.localize(datetime.strptime(value, fmt))
        except Exception:
            pass
    return None


# ── v2.0 — Win Candle Lock helpers ────────────────────────────────────────────
# After a TP (winning) close, the bot records the M15 candle boundary at which
# the win occurred.  _guard_phase() checks this before allowing a new entry:
# if the current candle is still the same candle the win closed on, entry is
# blocked.  No timer involved — the lock clears automatically when the next
# M15 candle opens.  This prevents re-entering on exhausted price moves in
# the seconds/minutes immediately after a TP hit.
#
# Key design decisions:
#   - Candle-boundary based, NOT time-based (no cooldown minutes in settings)
#   - Consistent with require_candle_close=True — both wait for candle boundaries
#   - Lock is stored in runtime_state.json so it survives process restarts
#   - Lock auto-expires: once the current candle != win candle, it clears itself
#   - Only TP wins set the lock; SL losses do NOT (losses use loss-streak logic)

def get_m15_candle_floor(dt: datetime) -> str:
    """Return the M15 candle floor timestamp string for a given SGT datetime.

    Examples:
        10:47 SGT  →  '2026-03-23 10:45'
        10:53 SGT  →  '2026-03-23 10:45'
        11:01 SGT  →  '2026-03-23 11:00'
        11:15 SGT  →  '2026-03-23 11:15'
    """
    floored_min = (dt.minute // 15) * 15
    candle_floor = dt.replace(minute=floored_min, second=0, microsecond=0)
    return candle_floor.strftime("%Y-%m-%d %H:%M")


def set_last_win_candle(dt: datetime) -> None:
    """Record the M15 candle floor at which a TP win was detected.

    Called from backfill_pnl() whenever pnl > 0 is first detected on a
    previously-open trade (i.e. the trade just closed as a winner).
    The candle floor — not the exact close time — is stored so that
    comparison in _guard_phase() is purely candle-index based.
    """
    candle_ts = get_m15_candle_floor(dt)
    update_runtime_state(last_win_candle_ts=candle_ts)
    logger.info("Win candle lock SET — candle=%s (no new entry until next candle)", candle_ts)


def get_last_win_candle() -> str | None:
    """Return the stored win-candle floor string, or None if not set / cleared."""
    state = load_json(RUNTIME_STATE_FILE, {})
    val = state.get("last_win_candle_ts")
    # Treat explicit None or empty string as "not set"
    return val if val else None


def clear_last_win_candle() -> None:
    """Explicitly clear the win candle lock.

    Called by _guard_phase() when it detects the current candle has advanced
    past the win candle — the lock is no longer needed and is removed from
    runtime_state.json so subsequent log output stays clean.
    """
    update_runtime_state(last_win_candle_ts=None)
    logger.info("Win candle lock CLEARED — new M15 candle confirmed, entries re-enabled")


# ── v5.2 — Post-win score improvement lock ────────────────────────────────────
# After a TP win, new entries are blocked unless the score has IMPROVED beyond
# the score that won.  The lock uses a "dip-and-recover" mechanism:
#
#   - Win happens at score W  → lock set, post_win_score = W
#   - Score stays >= W        → BLOCKED (no improvement proven)
#   - Score dips below W      → score_dipped_below_win flag set
#   - Score comes back >= W   → ALLOWED (fresh setup confirmed by dip+recover)
#   - Score rises above W     → ALLOWED immediately (strict improvement)
#
# State stored in runtime_state.json:
#   post_win_score          : int  — the score of the winning trade
#   post_win_score_dipped   : bool — True once score has dipped below win score
#
# The lock is cleared when a new trade is successfully placed.

def set_post_win_score(score: int) -> None:
    """Record the score of the most recent TP win and arm the score-improve lock."""
    update_runtime_state(
        post_win_score=score,
        post_win_score_dipped=False,
    )
    logger.info("Post-win score lock SET — win_score=%d, waiting for score dip + recover", score)


def get_post_win_score_state() -> tuple[int | None, bool]:
    """Return (win_score, dipped) from runtime_state, or (None, False) if not set."""
    state = load_json(RUNTIME_STATE_FILE, {})
    win_score = state.get("post_win_score")
    dipped    = bool(state.get("post_win_score_dipped", False))
    return (int(win_score) if win_score is not None else None, dipped)


def mark_post_win_score_dipped() -> None:
    """Record that the score has dipped below the winning score."""
    update_runtime_state(post_win_score_dipped=True)
    logger.info("Post-win score lock — score dipped below win score, recovery will allow entry")


def clear_post_win_score() -> None:
    """Clear the post-win score lock after a new trade is placed."""
    update_runtime_state(post_win_score=None, post_win_score_dipped=False)
    logger.info("Post-win score lock CLEARED — new trade placed")


# ── v5.2 — Last winning TP price guard ───────────────────────────────────────
# After a TP win, store the exact TP price.  If the next signal's computed TP
# lands at the same price level (within tolerance), block entry — the market
# has already cleared that level and a fresh TP at the same spot is stale.
#
# Tolerance: 1 pip on XAUUSD = $0.10.  Using 5 pips ($0.50) as safe default.

def set_last_win_tp(tp_price: float) -> None:
    """Store the TP price of the most recently won trade."""
    update_runtime_state(last_win_tp_price=round(tp_price, 2))
    logger.info("Last win TP price stored: %.2f", tp_price)


def get_last_win_tp() -> float | None:
    """Return the stored last-win TP price, or None if not set."""
    state = load_json(RUNTIME_STATE_FILE, {})
    val = state.get("last_win_tp_price")
    return float(val) if val is not None else None


def clear_last_win_tp() -> None:
    """Clear the last-win TP price (called after a new trade is placed)."""
    update_runtime_state(last_win_tp_price=None)
    logger.info("Last win TP price CLEARED")
