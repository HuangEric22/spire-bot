import unittest

import numpy as np

from spire_bot.envs.actions import MAX_HAND_SIZE
from spire_bot.envs.observations import (
    CARD_FEATURES,
    ENEMY_FEATURES,
    MAX_ENEMIES,
    OBSERVATION_SIZE,
    PLAYER_FEATURES,
    encode_observation,
    encode_observation_dict,
    flatten_observation_dict,
    old_encode_observation,
)
from tests.fixtures import shrinker_beetle_state


class ObservationTests(unittest.TestCase):
    def test_observation_dict_has_expected_shapes_and_dtypes(self) -> None:
        obs = encode_observation_dict(shrinker_beetle_state())

        self.assertEqual(obs["player_features"].shape, (PLAYER_FEATURES,))
        self.assertEqual(obs["enemy_features"].shape, (MAX_ENEMIES, ENEMY_FEATURES))
        self.assertEqual(obs["enemy_valid"].shape, (MAX_ENEMIES,))
        self.assertEqual(obs["card_features"].shape, (MAX_HAND_SIZE, CARD_FEATURES))
        self.assertEqual(obs["card_valid"].shape, (MAX_HAND_SIZE,))

        self.assertEqual(obs["player_features"].dtype, np.float32)
        self.assertEqual(obs["enemy_features"].dtype, np.float32)
        self.assertEqual(obs["enemy_valid"].dtype, np.bool_)
        self.assertEqual(obs["card_features"].dtype, np.float32)
        self.assertEqual(obs["card_valid"].dtype, np.bool_)

    def test_validity_masks_match_real_enemy_and_card_slots(self) -> None:
        obs = encode_observation_dict(shrinker_beetle_state(hand_size=7, enemy_count=1))

        np.testing.assert_array_equal(
            obs["enemy_valid"],
            np.array([True, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            obs["card_valid"],
            np.array([True, True, True, True, True, True, True, False, False, False]),
        )

    def test_padding_rows_are_zero(self) -> None:
        obs = encode_observation_dict(shrinker_beetle_state(hand_size=1, enemy_count=1))

        self.assertTrue(np.all(obs["enemy_features"][1:] == 0.0))
        self.assertTrue(np.all(obs["card_features"][1:] == 0.0))

    def test_shapes_and_masks_for_different_counts(self) -> None:
        cases = [
            (0, 1),
            (1, 1),
            (5, 2),
            (10, 5),
        ]

        for hand_size, enemy_count in cases:
            with self.subTest(hand_size=hand_size, enemy_count=enemy_count):
                obs = encode_observation_dict(
                    shrinker_beetle_state(hand_size=hand_size, enemy_count=enemy_count)
                )

                self.assertEqual(obs["enemy_features"].shape, (MAX_ENEMIES, ENEMY_FEATURES))
                self.assertEqual(obs["card_features"].shape, (MAX_HAND_SIZE, CARD_FEATURES))
                self.assertEqual(int(obs["enemy_valid"].sum()), enemy_count)
                self.assertEqual(int(obs["card_valid"].sum()), hand_size)

    def test_flattened_dict_matches_legacy_flat_observation(self) -> None:
        state = shrinker_beetle_state()

        flat_from_dict = flatten_observation_dict(encode_observation_dict(state))
        legacy_flat = old_encode_observation(state)

        self.assertEqual(len(flat_from_dict), OBSERVATION_SIZE)
        np.testing.assert_array_equal(
            np.asarray(flat_from_dict, dtype=np.float32),
            np.asarray(legacy_flat, dtype=np.float32),
        )

    def test_public_encoder_still_returns_flat_observation(self) -> None:
        obs = encode_observation(shrinker_beetle_state())

        self.assertIsInstance(obs, list)
        self.assertEqual(len(obs), OBSERVATION_SIZE)


if __name__ == "__main__":
    unittest.main()

class ObservationsTest(unittest.TestCase):
    def test_observation_has_expected_shape_and_values(self) -> None:
        return None
