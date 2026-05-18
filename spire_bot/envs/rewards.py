"""Reward helpers for the minimal Silent combat environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


State = dict[str, Any]

DAMAGE_REWARD_SCALE = 0.10
HP_LOSS_PENALTY_SCALE = -0.20
STEP_PENALTY = -0.01
COMBAT_WIN_REWARD = 10.0
COMBAT_LOSS_PENALTY = -10.0
POST_COMBAT_WIN_DECISIONS = {"card_reward", "map_select"}


@dataclass(frozen=True)
class RewardBreakdown:
    """Named reward pieces for debugging and tuning."""

    damage_dealt: float
    hp_lost: float
    step: float
    terminal: float

    @property
    def total(self) -> float:
        return self.damage_dealt + self.hp_lost + self.step + self.terminal


def compute_reward(previous_state: State, next_state: State) -> float:
    """Return the scalar reward for one simulator transition."""
    return reward_breakdown(previous_state, next_state).total


def reward_breakdown(previous_state: State, next_state: State) -> RewardBreakdown:
    """Return named reward terms for easier debugging."""
    damage_dealt = _enemy_hp_lost(previous_state, next_state) * DAMAGE_REWARD_SCALE
    hp_lost = _player_hp_lost(previous_state, next_state) * HP_LOSS_PENALTY_SCALE
    terminal = _terminal_reward(previous_state, next_state)

    return RewardBreakdown(
        damage_dealt=damage_dealt,
        hp_lost=hp_lost,
        step=STEP_PENALTY,
        terminal=terminal,
    )


def _enemy_hp_lost(previous_state: State, next_state: State) -> float:
    previous_hp = _enemy_hp_by_identity(previous_state)
    next_hp = _enemy_hp_by_identity(next_state)

    hp_lost = 0.0
    for enemy_id, old_hp in previous_hp.items():
        new_hp = next_hp.get(enemy_id, 0.0)
        hp_lost += max(old_hp - new_hp, 0.0)
    return hp_lost


def _player_hp_lost(previous_state: State, next_state: State) -> float:
    return max(_player_hp(previous_state) - _player_hp(next_state), 0.0)


def _terminal_reward(previous_state: State, next_state: State) -> float:
    if _combat_lost(next_state):
        return COMBAT_LOSS_PENALTY
    if _combat_won(previous_state, next_state):
        return COMBAT_WIN_REWARD
    return 0.0


def _combat_won(previous_state: State, next_state: State) -> bool:
    if previous_state.get("decision") != "combat_play":
        return False

    if next_state.get("decision") == "game_over":
        return bool(next_state.get("victory"))

    return next_state.get("decision") in POST_COMBAT_WIN_DECISIONS and not _combat_lost(next_state)


def _combat_lost(state: State) -> bool:
    if state.get("decision") == "game_over":
        return not bool(state.get("victory"))
    return _player_hp(state) <= 0.0


def _enemy_hp_by_identity(state: State) -> dict[int, float]:
    enemies: dict[int, float] = {}
    for position, enemy in enumerate(state.get("enemies") or []):
        enemy_id = enemy.get("combat_id", position)
        enemies[enemy_id] = _number(enemy.get("hp"))
    return enemies

def _player_hp(state: State) -> float:
    player = state.get("player") or {}
    return _number(player.get("hp"))

def _number(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
