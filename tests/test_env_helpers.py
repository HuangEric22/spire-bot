import unittest

from spire_bot.envs.actions import ACTION_SPACE_SIZE, decode_action, valid_action_mask
from spire_bot.envs.observations import OBSERVATION_SIZE, encode_observation
from spire_bot.envs.rewards import compute_reward, reward_breakdown


def sample_state() -> dict:
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
        "enemies": [
            {
                "index": 0,
                "combat_id": 1,
                "id": "MONSTER.SHRINKER_BEETLE",
                "entry": "SHRINKER_BEETLE",
                "name": "Shrinker Beetle",
                "hp": 40,
                "max_hp": 40,
                "block": 0,
                "intends_attack": False,
                "intents": [{"type": "DebuffStrong"}],
                "powers": None,
            },
        ],
        "hand": [
            {
                "index": 0,
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
                "index": 1,
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
                "index": 2,
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
                "index": 3,
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
                "index": 4,
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
                "index": 5,
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
                "index": 6,
                "id": "CARD.DEFEND_SILENT",
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "rarity": "Basic",
                "target_type": "Self",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
    }


class ActionTests(unittest.TestCase):
    def test_action_mask_and_decode(self) -> None:
        state = sample_state()

        mask = valid_action_mask(state)

        self.assertEqual(len(mask), ACTION_SPACE_SIZE)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertTrue(mask[4])
        self.assertTrue(mask[5])
        self.assertTrue(mask[6])
        self.assertTrue(mask[7])
        self.assertTrue(all(value is False for value in mask[8:]))

        self.assertEqual(decode_action(0, state).name, "end_turn")

        play_attack = decode_action(1, state)
        self.assertEqual(play_attack.name, "play_card")
        self.assertEqual(play_attack.args, {"card_index": 0, "target_index": 0})

        play_skill = decode_action(4, state)
        self.assertEqual(play_skill.args, {"card_index": 3})

        with self.assertRaises(ValueError):
            decode_action(8, state)

    def test_attack_card_is_invalid_without_enemy(self) -> None:
        state = sample_state()
        state["enemies"] = []

        mask = valid_action_mask(state)

        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertTrue(mask[4])


class ObservationTests(unittest.TestCase):
    def test_observation_has_expected_shape_and_values(self) -> None:
        obs = encode_observation(sample_state())

        self.assertEqual(len(obs), OBSERVATION_SIZE)

        self.assertAlmostEqual(obs[0], 1.0)
        self.assertAlmostEqual(obs[1], 0.0)
        self.assertAlmostEqual(obs[2], 1.0)
        self.assertAlmostEqual(obs[3], 0.30)
        self.assertAlmostEqual(obs[4], 0.05)
        self.assertAlmostEqual(obs[5], 0.24)

        self.assertAlmostEqual(obs[6], 1.0)
        self.assertAlmostEqual(obs[7], 1.0)
        self.assertAlmostEqual(obs[8], 0.0)
        self.assertAlmostEqual(obs[9], 0.0)
        self.assertAlmostEqual(obs[10], 0.0)
        self.assertAlmostEqual(obs[11], 0.0)
        self.assertAlmostEqual(obs[12], 1.0)
        self.assertAlmostEqual(obs[13], 0.0)
        self.assertAlmostEqual(obs[14], 0.0)

        second_enemy_start = 16
        self.assertTrue(all(value == 0.0 for value in obs[second_enemy_start:second_enemy_start + 10]))

        first_card_start = 56
        self.assertAlmostEqual(obs[first_card_start], 1.0)
        self.assertAlmostEqual(obs[first_card_start + 1], 0.0)
        self.assertAlmostEqual(obs[first_card_start + 4], 1.0)
        self.assertAlmostEqual(obs[first_card_start + 8], 1.0)
        self.assertAlmostEqual(obs[first_card_start + 10], 0.06)

        second_card_start = 68
        self.assertAlmostEqual(obs[second_card_start + 1], 0.20)
        self.assertAlmostEqual(obs[second_card_start + 4], 1.0)
        self.assertAlmostEqual(obs[second_card_start + 8], 1.0)
        self.assertAlmostEqual(obs[second_card_start + 10], 0.12)

        defend_card_start = 92
        self.assertAlmostEqual(obs[defend_card_start + 5], 1.0)
        self.assertAlmostEqual(obs[defend_card_start + 7], 1.0)
        self.assertAlmostEqual(obs[defend_card_start + 11], 0.10)

        empty_card_start = 140
        self.assertTrue(all(value == 0.0 for value in obs[empty_card_start:empty_card_start + 12]))


class RewardTests(unittest.TestCase):
    def test_damage_uses_combat_id_when_enemies_reindex(self) -> None:
        previous_state = {
            "decision": "combat_play",
            "player": {"hp": 70},
            "enemies": [
                {"index": 0, "combat_id": 1, "hp": 1},
                {"index": 1, "combat_id": 2, "hp": 24},
                {"index": 2, "combat_id": 3, "hp": 25},
            ],
        }
        next_state = {
            "decision": "combat_play",
            "player": {"hp": 70},
            "enemies": [
                {"index": 0, "combat_id": 2, "hp": 24},
                {"index": 1, "combat_id": 3, "hp": 25},
            ],
        }

        breakdown = reward_breakdown(previous_state, next_state)

        self.assertAlmostEqual(breakdown.damage_dealt, 0.10)
        self.assertAlmostEqual(breakdown.total, 0.09)

    def test_hp_loss_and_terminal_rewards(self) -> None:
        previous_state = {
            "decision": "combat_play",
            "player": {"hp": 70},
            "enemies": [{"index": 0, "combat_id": 1, "hp": 3}],
        }

        damaged_state = {
            "decision": "combat_play",
            "player": {"hp": 65},
            "enemies": [{"index": 0, "combat_id": 1, "hp": 3}],
        }
        win_state = {"decision": "card_reward", "player": {"hp": 70}, "enemies": []}
        loss_state = {"decision": "game_over", "victory": False, "player": {"hp": 0}, "enemies": []}

        self.assertAlmostEqual(reward_breakdown(previous_state, damaged_state).hp_lost, -1.0)
        self.assertAlmostEqual(reward_breakdown(previous_state, win_state).terminal, 10.0)
        self.assertAlmostEqual(reward_breakdown(previous_state, loss_state).terminal, -10.0)
        self.assertAlmostEqual(compute_reward(previous_state, win_state), 10.29)


if __name__ == "__main__":
    unittest.main()
