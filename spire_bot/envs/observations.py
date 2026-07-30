"""Observation encoding for the minimal Silent combat environment."""

from __future__ import annotations

from typing import Any

from spire_bot.envs.actions import MAX_HAND_SIZE

import numpy as np

State = dict[str, Any]

MAX_ENEMIES = 5
PLAYER_FEATURES = 6
ENEMY_FEATURES = 10
CARD_FEATURES = 12
OBSERVATION_SIZE = PLAYER_FEATURES + (MAX_ENEMIES * ENEMY_FEATURES) + (MAX_HAND_SIZE * CARD_FEATURES)

CARD_TYPE_TO_INDEX = {
    "Attack": 0,
    "Skill": 1,
    "Power": 2,
    "Status": 3,
    "Curse": 4,
}

TARGET_TYPE_TO_INDEX = {
    "None": 0,
    "Self": 1,
    "AnyEnemy": 2,
    "AllEnemies": 3,
}

def flatten_observation_dict(obs: dict[str, Any]) -> list[float]:
    flattened_obs = np.concatenate([
        obs["player_features"].reshape(-1),
        obs["enemy_features"].reshape(-1),
        obs["card_features"].reshape(-1),
    ])
    
    if len(flattened_obs) != OBSERVATION_SIZE:
        raise ValueError(f"Observation has length {len(values)}, expected {OBSERVATION_SIZE}.")    
    
    return flattened_obs.tolist()

def encode_observation_dict(state: State) -> dict[str, Any]:
    player_features = np.asarray(_player_features(state), dtype=np.float32)

    enemy_features = np.zeros((MAX_ENEMIES, ENEMY_FEATURES), dtype=np.float32)
    enemies = state.get("enemies") or []
    for enemy_slot, enemy in enumerate(enemies[:MAX_ENEMIES]):
        enemy_features[enemy_slot] = _enemy_features(enemy)

    card_features = np.zeros((MAX_HAND_SIZE, CARD_FEATURES), dtype=np.float32)
    cards_by_index = _cards_by_index(state)
    for hand_index in range(MAX_HAND_SIZE):
        card_features[hand_index] = _card_features(cards_by_index.get(hand_index))

    return {
        "player_features": player_features,
        "enemy_features": enemy_features,
        "card_features": card_features,
    }

def encode_observation(state: State) -> list[float]:
    return flatten_observation_dict(encode_observation_dict(state))

def old_encode_observation(state: State) -> list[float]:
    """Convert a simulator combat state into a fixed-size numeric vector."""
    values: list[float] = []
    values.extend(_player_features(state))

    enemies = state.get("enemies") or []
    for enemy in enemies[:MAX_ENEMIES]:
        values.extend(_enemy_features(enemy))
    values.extend([0.0] * ((MAX_ENEMIES - min(len(enemies), MAX_ENEMIES)) * ENEMY_FEATURES))

    cards_by_index = _cards_by_index(state)
    for hand_index in range(MAX_HAND_SIZE):
        values.extend(_card_features(cards_by_index.get(hand_index)))

    if len(values) != OBSERVATION_SIZE:
        raise ValueError(f"Observation has length {len(values)}, expected {OBSERVATION_SIZE}.")
    return values


def _player_features(state: State) -> list[float]:
    player = state.get("player") or {}
    hp = _number(player.get("hp"))
    max_hp = max(_number(player.get("max_hp")), 1.0)
    energy = _number(state.get("energy"))
    max_energy = max(_number(state.get("max_energy")), 1.0)

    return [
        hp / max_hp,
        _number(player.get("block")) / 50.0,
        energy / max_energy,
        max_energy / 10.0,
        _number(state.get("round")) / 20.0,
        _number(player.get("deck_size")) / 50.0,
    ]


def _enemy_features(enemy: dict[str, Any]) -> list[float]:
    hp = _number(enemy.get("hp"))
    max_hp = max(_number(enemy.get("max_hp")), 1.0)
    powers = _powers_by_id(enemy)
    attack_damage, attack_hits, has_debuff_intent, has_block_intent = _intent_features(enemy)

    return [
        1.0,
        hp / max_hp,
        _number(enemy.get("block")) / 50.0,
        1.0 if enemy.get("intends_attack") else 0.0,
        attack_damage / 50.0,
        attack_hits / 10.0,
        has_debuff_intent,
        has_block_intent,
        _power_amount(powers, "POWER.WEAK_POWER") / 10.0,
        _power_amount(powers, "POWER.VULNERABLE_POWER") / 10.0,
    ]


def _card_features(card: dict[str, Any] | None) -> list[float]:
    if card is None:
        return [0.0] * CARD_FEATURES

    stats = card.get("stats") or {}
    card_type = card.get("type")
    target_type = card.get("target_type")

    return [
        1.0,
        _number(card.get("cost")) / 5.0,
        1.0 if card.get("can_play") else 0.0,
        1.0 if card.get("upgraded") else 0.0,
        _one_hot_value(card_type, CARD_TYPE_TO_INDEX, 0),
        _one_hot_value(card_type, CARD_TYPE_TO_INDEX, 1),
        _one_hot_value(card_type, CARD_TYPE_TO_INDEX, 2),
        _one_hot_value(target_type, TARGET_TYPE_TO_INDEX, 1),
        _one_hot_value(target_type, TARGET_TYPE_TO_INDEX, 2),
        _one_hot_value(target_type, TARGET_TYPE_TO_INDEX, 3),
        _number(stats.get("damage")) / 50.0,
        _number(stats.get("block")) / 50.0,
    ]


def _cards_by_index(state: State) -> dict[int, dict[str, Any]]:
    cards: dict[int, dict[str, Any]] = {}
    for card in state.get("hand") or []:
        index = card.get("index")
        if index is not None:
            cards[int(index)] = card
    return cards


def _powers_by_id(entity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    powers: dict[str, dict[str, Any]] = {}
    for power in entity.get("powers") or []:
        power_id = power.get("id")
        if power_id:
            powers[str(power_id)] = power
    return powers


def _power_amount(powers: dict[str, dict[str, Any]], power_id: str) -> float:
    return _number((powers.get(power_id) or {}).get("amount"))


def _intent_features(enemy: dict[str, Any]) -> tuple[float, float, float, float]:
    total_damage = 0.0
    total_hits = 0.0
    has_debuff = 0.0
    has_block = 0.0

    for intent in enemy.get("intents") or []:
        intent_type = str(intent.get("type", ""))
        if "Attack" in intent_type and "damage" in intent:
            total_damage += _number(intent.get("damage"))
            total_hits += max(_number(intent.get("hits"), 1.0), 1.0)
        if "Debuff" in intent_type:
            has_debuff = 1.0
        if "Block" in intent_type:
            has_block = 1.0

    return total_damage, total_hits, has_debuff, has_block


def _one_hot_value(value: object, mapping: dict[str, int], index: int) -> float:
    return 1.0 if mapping.get(str(value)) == index else 0.0


def _number(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def print_observation_debug(state: dict) -> None:
    obs = encode_observation(state)
    i = 0

    player_labels = [
        "player_hp_ratio",
        "player_block",
        "energy_ratio",
        "max_energy",
        "round",
        "deck_size",
    ]

    print("\nObservation debug:")
    print("Player:")
    for label in player_labels:
        print(f"  {i:03d} {label}: {obs[i]}")
        i += 1

    print("Enemies:")
    for enemy_slot in range(MAX_ENEMIES):
        labels = [
            "exists",
            "hp_ratio",
            "block",
            "intends_attack",
            "attack_damage",
            "attack_hits",
            "has_debuff_intent",
            "has_block_intent",
            "weak",
            "vulnerable",
        ]
        print(f"  Enemy slot {enemy_slot}:")
        for label in labels:
            print(f"    {i:03d} {label}: {obs[i]}")
            i += 1

    print("Hand:")
    for card_slot in range(MAX_HAND_SIZE):
        labels = [
            "exists",
            "cost",
            "can_play",
            "upgraded",
            "is_attack",
            "is_skill",
            "is_power",
            "targets_self",
            "targets_enemy",
            "targets_all_enemies",
            "damage",
            "block",
        ]
        print(f"  Card slot {card_slot}:")
        for label in labels:
            print(f"    {i:03d} {label}: {obs[i]}")
            i += 1