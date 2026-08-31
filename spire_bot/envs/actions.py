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
    """A series of decisions that result in a single simulator action.
    
    Fields:
        action_type: First-stage action choice. Uses ActionType values.
        card_index: Hand slot for PLAY_CARD, or NO_CARD_INDEX for END_TURN.
        target_index: Enemy slot for AnyEnemy cards, or NO_TARGET_INDEX otherwise.    
    """
    
    action_type: int
    card_index: int
    target_index: int

@dataclass(frozen=True)
class FactoredActionMasks:
    """Boolean masks for the three-stage factored action space.
    Shapes:
        action_mask: ACTION_TYPE_SIZE.
        card_mask: CARD_INDEX_SIZE. Valid card slots for PLAY_CARD only.
        target_mask: CARD_INDEX_SIZE x TARGET_INDEX_SIZE.
            Each row gives the valid target slots for that card slot.

    Slot conventions:
        NO_CARD_INDEX is the dummy card slot used by END_TURN.
        NO_TARGET_INDEX is the dummy target slot used by END_TURN and
        cards that do not require a specific enemy target.
    """
    
    action_mask: list[bool]
    card_mask: list[bool]
    target_mask: list[list[bool]]
    
@dataclass(frozen=True)
class SimulatorAction:
    """A decoded action ready to send to SimulatorClient.act()."""

    name: str
    args: dict[str, Any]

# Pure functions for factored action steps
def create_factored_action_masks(state: State) -> FactoredActionMasks:
    """Build legal choices for the current combat state."""
    
    action_mask = [False] * ACTION_TYPE_SIZE
    card_mask = [False] * CARD_INDEX_SIZE
    target_mask = [[False] * TARGET_INDEX_SIZE for _ in range(CARD_INDEX_SIZE)]
    
    # Default every card/source slot to the no-target choice; enemy-target cards override this row.
    for mask in target_mask:
        mask[NO_TARGET_INDEX] = True
        
    hand = state.get("hand") or []
    enemies = state.get("enemies") or []
    num_enemies = len(enemies)
    has_enemies = num_enemies > 0
    
    if len(hand) > MAX_HAND_SIZE:
        raise ValueError(f"MAX_HAND_SIZE limit exceeded: got hand size of {len(hand)} when MAX_HAND_SIZE is {MAX_HAND_SIZE}.")
    
    if num_enemies > MAX_ENEMIES:
        raise ValueError(f"MAX_ENEMIES limit exceeded: got enemy count of {len(enemies)} when MAX_ENEMIES is {MAX_ENEMIES}.")
        
    for card in hand:
        card_index = int(card.get("index", -1))
        if card_index < 0 or card_index >= MAX_HAND_SIZE:
            raise ValueError(f"Invalid card index in hand: {card_index}")
        
        can_play = bool(card.get("can_play", False))
        needs_enemy = str(card.get("target_type", "")) == "AnyEnemy"
        
        card_mask[card_index] = can_play and (not needs_enemy or has_enemies)
        
        if needs_enemy and has_enemies:
            target_mask[card_index][NO_TARGET_INDEX] = False            
            for enemy_idx, _ in enumerate(enemies):
                target_mask[card_index][enemy_idx] = True

    action_mask[ActionType.PLAY_CARD.value] = any(card_mask)        
    action_mask[ActionType.END_TURN.value] = True

    return FactoredActionMasks(action_mask=action_mask, card_mask=card_mask, target_mask=target_mask)

def _only_no_card_mask() -> list[bool]:
    mask = [False] * CARD_INDEX_SIZE
    mask[NO_CARD_INDEX] = True
    return mask

def _only_no_target_mask() -> list[bool]:
    mask = [False] * TARGET_INDEX_SIZE
    mask[NO_TARGET_INDEX] = True
    return mask

def card_mask_for_action_type(action: int, masks: FactoredActionMasks) -> list[bool]:
    action_type = ActionType(int(action))
    
    if action_type == ActionType.END_TURN:
        return _only_no_card_mask()
    
    if action_type == ActionType.PLAY_CARD:
        return masks.card_mask.copy()

    raise ValueError(f"Unsupported action type: {action_type}")        

def target_mask_for_card_and_action(action: int, card_index: int, masks: FactoredActionMasks) -> list[bool]:
    action_type = ActionType(int(action))
    
    if action_type == ActionType.END_TURN:
        return _only_no_target_mask()
    
    if action_type == ActionType.PLAY_CARD:
        if card_index < 0 or card_index >= NO_CARD_INDEX:
            raise ValueError(f"Invalid card index: {card_index}")
        return masks.target_mask[card_index].copy()
    
    raise ValueError(f"Unsupported action type: {action_type}")

def is_valid_factored_action(action: FactoredAction, masks: FactoredActionMasks) -> bool:
    try:
        action_type = int(action.action_type)
        selected_card_idx = int(action.card_index)
        selected_target_idx = int(action.target_index)
    except (AttributeError, TypeError, ValueError):
        return False
    
    if action_type < 0 or action_type >= ACTION_TYPE_SIZE:
        return False
    if selected_card_idx < 0 or selected_card_idx >= CARD_INDEX_SIZE:
        return False
    if selected_target_idx < 0 or selected_target_idx >= TARGET_INDEX_SIZE:
        return False
    
    if not masks.action_mask[action_type]:
        return False
    
    card_mask = card_mask_for_action_type(action_type, masks)
    if not card_mask[selected_card_idx]:
        return False
    
    target_mask = target_mask_for_card_and_action(action_type, selected_card_idx, masks)
    if not target_mask[selected_target_idx]:
        return False
    
    return True

def decode_factored_action(action: FactoredAction, state: State) -> SimulatorAction:    
    masks = create_factored_action_masks(state)
    
    if not is_valid_factored_action(action, masks):
        raise ValueError(f"Invalid factored action: {action}")
    
    action_type = int(action.action_type)
    selected_card_idx = int(action.card_index)
    selected_target_idx = int(action.target_index)
    
    if action_type == ActionType.END_TURN:
        return SimulatorAction("end_turn", {})
    
    if action_type == ActionType.PLAY_CARD:
        if selected_card_idx == NO_CARD_INDEX:
            raise ValueError(f"Cannot perform action PLAY_CARD with NO_CARD_INDEX: {NO_CARD_INDEX}")
        
        hand = {int(card["index"]): card for card in state.get("hand") or []}
        card = hand.get(selected_card_idx)
        
        if card is None:
            raise ValueError(f"Missing card at hand index {selected_card_idx}.")
        
        args: dict[str, Any] = {"card_index": selected_card_idx}
        
        if str(card.get("target_type", "")) == "AnyEnemy":
            if selected_target_idx == NO_TARGET_INDEX:
                card_name = card.get("name", "INVALID_CARD")
                raise ValueError(f"A target is required for the card: {card_name}")
            args["target_index"] = selected_target_idx
        
        return SimulatorAction("play_card", args=args)    
        
    raise ValueError(f"The action type index:{action_type} is not a valid action!")
    
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
