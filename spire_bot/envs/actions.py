"""Action encoding helpers for the minimal Silent combat environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


State = dict[str, Any]

MAX_HAND_SIZE = 10
END_TURN_ACTION = 0
ACTION_SPACE_SIZE = MAX_HAND_SIZE + 1


@dataclass(frozen=True)
class SimulatorAction:
    """A decoded action ready to send to SimulatorClient.act()."""

    name: str
    args: dict[str, Any]


def valid_action_mask(state: State) -> list[bool]:
    """Return which discrete actions are legal in the current combat state."""
    mask = [False] * ACTION_SPACE_SIZE
    mask[END_TURN_ACTION] = True

    enemies = state.get("enemies") or []
    has_enemy = len(enemies) > 0
    hand = state.get("hand") or []
    for card in hand[:MAX_HAND_SIZE]:
        card_index = int(card.get("index", -1))
        action_index = card_index + 1
        if action_index <= END_TURN_ACTION or action_index >= ACTION_SPACE_SIZE:
            continue

        can_play = bool(card.get("can_play", False))
        target_type = card.get("target_type")
        needs_enemy = target_type == "AnyEnemy"
        mask[action_index] = can_play and (not needs_enemy or has_enemy)

    return mask


def decode_action(action: int, state: State) -> SimulatorAction:
    """Convert a discrete action index into a simulator action."""
    if action == END_TURN_ACTION:
        return SimulatorAction("end_turn", {})

    if action < 0 or action >= ACTION_SPACE_SIZE:
        raise ValueError(f"Action {action} is outside the action space.")

    hand_index = action - 1
    card = _card_by_index(state, hand_index)
    if card is None:
        raise ValueError(f"Action {action} refers to missing hand index {hand_index}.")

    if not valid_action_mask(state)[action]:
        raise ValueError(f"Action {action} is not valid in the current state.")

    args: dict[str, Any] = {"card_index": hand_index}
    if card.get("target_type") == "AnyEnemy":
        args["target_index"] = _first_enemy_index(state)

    return SimulatorAction("play_card", args)


def _card_by_index(state: State, hand_index: int) -> dict[str, Any] | None:
    for card in state.get("hand") or []:
        if card.get("index") == hand_index:
            return card
    return None


def _first_enemy_index(state: State) -> int:
    enemies = state.get("enemies") or []
    if not enemies:
        raise ValueError("No enemy is available to target.")
    return int(enemies[0].get("index", 0))
