"""Human-readable printing of raw simulator combat states."""

from __future__ import annotations

import sys
from pathlib import Path

_SIMULATOR_PYTHON_DIR = Path(__file__).resolve().parents[2] / "external" / "sts2-cli" / "python"


def _load_simulator_show_combat():
    """Import the simulator's rich combat renderer from play.py, if possible."""
    try:
        if str(_SIMULATOR_PYTHON_DIR) not in sys.path:
            sys.path.insert(0, str(_SIMULATOR_PYTHON_DIR))
        import play
    except Exception:
        return None

    play.LANG = "en"
    # play.py draws HP bars with characters Windows console code pages reject.
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            return None
    return play.show_combat


_show_combat = _load_simulator_show_combat()


def print_state(state: dict) -> None:
    """Print a combat state using the simulator's rich renderer when available."""
    if _show_combat is not None and state.get("decision") == "combat_play":
        print(f"\n== {state.get('decision')} ==")
        _show_combat(state)
        return
    print_state_plain(state)


def print_state_plain(state: dict) -> None:
    """Print the most important combat fields in a compact human-readable form."""
    player = state.get("player") or {}
    print(f"\n== {state.get('decision')} | round {state.get('round')} ==")
    print(
        "Player: "
        f"hp {player.get('hp')}/{player.get('max_hp')} "
        f"block {player.get('block', 0)} "
        f"energy {state.get('energy')}/{state.get('max_energy')} "
        f"deck {player.get('deck_size')}"
    )

    print("Hand:")
    for card in state.get("hand") or []:
        print(
            f"  [{card.get('index')}] {card.get('name')} "
            f"({card.get('id')}) "
            f"cost={card.get('cost')} "
            f"type={card.get('type')} "
            f"target={card.get('target_type')} "
            f"can_play={card.get('can_play')}"
            f"{_format_stats(card.get('stats'))}"
        )

    print("Enemies:")
    for enemy in state.get("enemies") or []:
        print(
            f"  [{enemy.get('index')}] {enemy.get('name')} "
            f"({enemy.get('id')} {enemy.get('combat_id')}) "
            f"hp {enemy.get('hp')}/{enemy.get('max_hp')} "
            f"block {enemy.get('block', 0)} "
            f"intent={_format_intents(enemy.get('intents'))} "
            f"powers={_format_powers(enemy.get('powers'))}"
        )


def _format_stats(stats: object) -> str:
    if not isinstance(stats, dict) or not stats:
        return ""
    parts = [f"{key}={value}" for key, value in stats.items()]
    return " stats[" + ", ".join(parts) + "]"


def _format_intents(intents: object) -> str:
    if not isinstance(intents, list) or not intents:
        return "none"

    parts = []
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        text = str(intent.get("type", "Unknown"))
        if "damage" in intent:
            text += f" {intent.get('damage')}"
            if "hits" in intent:
                text += f"x{intent.get('hits')}"
        parts.append(text)
    return ", ".join(parts) if parts else "none"


def _format_powers(powers: object) -> str:
    if not isinstance(powers, list) or not powers:
        return "none"

    parts = []
    for power in powers:
        if not isinstance(power, dict):
            continue
        amount = power.get("amount")
        suffix = f"({amount})" if amount is not None else ""
        parts.append(f"{power.get('name', power.get('id'))}{suffix}")
    return ", ".join(parts) if parts else "none"
