from __future__ import annotations

import copy
from typing import Any


SILENT_STARTER_HAND = [
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
    hand = []
    for index, card in enumerate(SILENT_STARTER_HAND[:hand_size]):
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
    }
