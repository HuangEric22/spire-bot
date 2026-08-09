from __future__ import annotations

import copy
from typing import Any

from spire_bot.envs.constants import MAX_ENEMIES, MAX_HAND_SIZE


SAMPLE_HAND_CARDS = [
    {
        "id": "CARD.NEUTRALIZE",
        "name": "Neutralize",
        "cost": 0,
        "type": "Attack",
        "rarity": "Basic",
        "target_type": "AnyEnemy",
        "can_play": True,
        "stats": {"damage": 3, "weakpower": 1},
    },
    {
        "id": "CARD.STRIKE_SILENT",
        "name": "Strike",
        "cost": 1,
        "type": "Attack",
        "rarity": "Basic",
        "target_type": "AnyEnemy",
        "can_play": True,
        "stats": {"damage": 6},
    },
    {
        "id": "CARD.STRIKE_SILENT",
        "name": "Strike",
        "cost": 1,
        "type": "Attack",
        "rarity": "Basic",
        "target_type": "AnyEnemy",
        "can_play": True,
        "stats": {"damage": 6},
    },
    {
        "id": "CARD.DEFEND_SILENT",
        "name": "Defend",
        "cost": 1,
        "type": "Skill",
        "rarity": "Basic",
        "target_type": "Self",
        "can_play": True,
        "stats": {"block": 5},
    },
    {
        "id": "CARD.DEFEND_SILENT",
        "name": "Defend",
        "cost": 1,
        "type": "Skill",
        "rarity": "Basic",
        "target_type": "Self",
        "can_play": True,
        "stats": {"block": 5},
    },
    {
        "id": "CARD.STRIKE_SILENT",
        "name": "Strike",
        "cost": 1,
        "type": "Attack",
        "rarity": "Basic",
        "target_type": "AnyEnemy",
        "can_play": True,
        "stats": {"damage": 6},
    },
    {
        "id": "CARD.DEFEND_SILENT",
        "name": "Defend",
        "cost": 1,
        "type": "Skill",
        "rarity": "Basic",
        "target_type": "Self",
        "can_play": True,
        "stats": {"block": 5},
    },
    {
        "id": "CARD.STRIKE_SILENT",
        "name": "Strike",
        "cost": 1,
        "type": "Attack",
        "rarity": "Basic",
        "target_type": "AnyEnemy",
        "can_play": True,
        "stats": {"damage": 6},
    },
    {
        "id": "CARD.DEFEND_SILENT",
        "name": "Defend",
        "cost": 1,
        "type": "Skill",
        "rarity": "Basic",
        "target_type": "Self",
        "can_play": True,
        "stats": {"block": 5},
    },
    {
        "id": "CARD.STRIKE_SILENT",
        "name": "Strike",
        "cost": 1,
        "type": "Attack",
        "rarity": "Basic",
        "target_type": "AnyEnemy",
        "can_play": True,
        "stats": {"damage": 6},
    },
]


def shrinker_beetle_state(hand_size: int = 7, enemy_count: int = 1) -> dict[str, Any]:
    max_hand = min(len(SAMPLE_HAND_CARDS), MAX_HAND_SIZE)
    if not 0 <= hand_size <= max_hand:
        raise ValueError(f"hand_size must be between 0 and {max_hand}.")

    if not 0 <= enemy_count <= MAX_ENEMIES:
        raise ValueError(f"enemy_count must be between 0 and {MAX_ENEMIES}.")

    hand = []
    for index, card in enumerate(SAMPLE_HAND_CARDS[:hand_size]):
        card_copy = copy.deepcopy(card)
        card_copy["index"] = index
        hand.append(card_copy)

    enemies = []
    for index in range(enemy_count):
        enemies.append(
            {
                "index": index,
                "combat_id": index + 1,
                "id": "MONSTER.SHRINKER_BEETLE",
                "entry": "SHRINKER_BEETLE",
                "name": "Shrinker Beetle",
                "hp": 40 - index,
                "max_hp": 40,
                "block": 0,
                "intends_attack": False,
                "intents": [{"type": "DebuffStrong"}],
                "powers": None,
            }
        )

    return {
        "type": "decision",
        "decision": "combat_play",
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "player": {
            "hp": 70,
            "max_hp": 70,
            "block": 0,
            "deck_size": 12,
        },
        "enemies": enemies,
        "hand": hand,
        "player_powers": [],
    }

def play_card(state: dict[str, Any], card_idx, target_idx) -> None:
    hand = state.get("hand")
    enemies = state.get("enemies")
    
    target = enemies[target_idx] if 0 < target_idx < len(enemies) else None
    card = hand[card_idx]

    if card["type"] == "Attack" and target:
        target["hp"] -= card["stats"].get("damage", 0)
        del hand[card["index"]]