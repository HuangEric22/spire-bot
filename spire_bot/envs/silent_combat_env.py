"""Gymnasium wrapper for a minimal Silent combat task."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from spire_bot.envs.actions import ACTION_SPACE_SIZE, decode_action, valid_action_mask
from spire_bot.envs.observations import OBSERVATION_SIZE, encode_observation
from spire_bot.envs.rewards import compute_reward
from spire_bot.envs.simulator_client import SimulatorClient


class SilentCombatEnv(gym.Env):
    """One-combat Silent environment backed by the STS2 simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        encounter: str = "SHRINKER_BEETLE_WEAK",
        base_seed: str = "silent-poc",
        max_steps: int = 200,
        no_build: bool = True,
        invalid_action_penalty: float = -1.0,
        restart_on_reset: bool = True,
    ) -> None:
        self.encounter = encounter
        self.base_seed = base_seed
        self.max_steps = max_steps
        self.no_build = no_build
        self.invalid_action_penalty = invalid_action_penalty
        self.restart_on_reset = restart_on_reset

        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self._sim = self._new_simulator()
        self._state: dict[str, Any] | None = None
        self._episode_index = 0
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_index += 1
        if self.restart_on_reset:
            self._sim.close()
            self._sim = self._new_simulator()

        encounter = (options or {}).get("encounter", self.encounter)
        run_seed = self._run_seed(seed)

        self._sim.start_run(character="Silent", seed=run_seed)
        self._state = self._settle_card_select(self._sim.enter_room("combat", encounter=encounter))
        if self._state.get("decision") != "combat_play":
            raise RuntimeError(
                "Expected reset to enter combat_play, "
                f"got decision={self._state.get('decision')!r} "
                f"hand={len(self._state.get('hand') or [])} "
                f"enemies={len(self._state.get('enemies') or [])}."
            )

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Returns observation, reward, terminated, truncated, info"""
        
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        previous_state = self._state
        mask = valid_action_mask(self._state)

        if action < 0 or action >= len(mask) or not mask[action]:
            reward = self.invalid_action_penalty
            truncated = self._step_count >= self.max_steps
            return self._observation(), reward, False, truncated, self._info(invalid_action=True)

        simulator_action = decode_action(action, previous_state)
        self._state = self._settle_card_select(
            self._sim.act(simulator_action.name, **simulator_action.args)
        )

        reward = compute_reward(previous_state, self._state)
        terminated = self._state.get("decision") != "combat_play"
        truncated = self._step_count >= self.max_steps

        return self._observation(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        return np.asarray(valid_action_mask(self._state), dtype=np.bool_)

    def close(self) -> None:
        self._sim.close()

    def _observation(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        return np.asarray(encode_observation(self._state), dtype=np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "action_mask": self.action_masks(),
            "state": self._state,
            "step_count": self._step_count,
        }
        info.update(extra)
        return info

    def _settle_card_select(self, state: dict[str, Any]) -> dict[str, Any]:
        """Auto-answer mid-combat card_select prompts.

        Cards like Survivor ("Discard 1 card.") pause the combat on a
        card_select decision. The action space has no card-selection action
        yet, so choose for the agent: skip when allowed, otherwise pick the
        first required cards. Without this, a mid-combat card_select would end
        the episode early, and because that state has no enemy list, the
        vanished enemies would be scored as phantom damage reward.
        """
        for _ in range(10):
            if state.get("decision") != "card_select":
                return state
            min_select = int(state.get("min_select") or 0)
            if min_select <= 0:
                state = self._sim.act("skip_select")
            else:
                indices = ",".join(str(i) for i in range(min_select))
                state = self._sim.act("select_cards", indices=indices)
        raise RuntimeError("card_select did not settle after 10 auto-selections.")

    def _run_seed(self, seed: int | None) -> str:
        if seed is None:
            return f"{self.base_seed}-{self._episode_index}"
        return f"{self.base_seed}-{seed}-{self._episode_index}"

    def _new_simulator(self) -> SimulatorClient:
        return SimulatorClient(no_build=self.no_build)
