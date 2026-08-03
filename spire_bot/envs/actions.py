"""Action encoding helpers for the minimal Silent combat environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from spire_bot.envs.constants import MAX_ENEMIES, MAX_HAND_SIZE
State = dict[str, Any]

# Game constants
END_TURN_ACTION = 0
ACTION_SPACE_SIZE = MAX_HAND_SIZE + 1

# Constants that define factored action parameters
ACTION_TYPE_SIZE = 2
CARD_INDEX_SIZE = MAX_HAND_SIZE + 1
NO_CARD_INDEX = MAX_HAND_SIZE
TARGET_INDEX_SIZE = MAX_ENEMIES + 1
NO_TARGET_INDEX = MAX_ENEMIES


class ActionType(IntEnum):
    END_TURN = 0
    PLAY_CARD = 1
    # DISCARD_CARD = 2

@dataclass(frozen=True)
class FactoredAction:
    """A series of decisions that result in a single simulator action."""
    
    action_type: ActionType
    card_index: int
    target_index: int

@dataclass(frozen=True)
class FactoredActionMasks:
    type_mask: list[bool]
    card_mask: list[bool]
    target_mask: list[bool]
    
@dataclass(frozen=True)
class SimulatorAction:
    """A decoded action ready to send to SimulatorClient.act()."""

    name: str
    args: dict[str, Any]

# Pure functions for factored action steps
def factored_action_masks(state: State) -> FactoredActionMasks:
    type_mask = [False] * ACTION_TYPE_SIZE
    card_mask = [False] * CARD_INDEX_SIZE
    target_mask = [False] * TARGET_INDEX_SIZE
    
    type_mask[ActionType.END_TURN.value] = True

    return None

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
