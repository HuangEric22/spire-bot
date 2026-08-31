import unittest

from spire_bot.envs.actions import (
    ACTION_TYPE_SIZE,
    CARD_INDEX_SIZE,
    NO_CARD_INDEX,
    NO_TARGET_INDEX,
    TARGET_INDEX_SIZE,
    ActionType,
    FactoredAction,
    create_factored_action_masks,
    decode_factored_action,
    is_valid_factored_action,
)
from tests.fixtures import shrinker_beetle_state


class ActionTest(unittest.TestCase):
    def test_factored_masks_have_expected_shapes(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)

        masks = create_factored_action_masks(state)

        self.assertEqual(len(masks.action_mask), ACTION_TYPE_SIZE)
        self.assertEqual(len(masks.card_mask), CARD_INDEX_SIZE)
        self.assertEqual(len(masks.target_mask), CARD_INDEX_SIZE)
        for target_mask in masks.target_mask:
            self.assertEqual(len(target_mask), TARGET_INDEX_SIZE)

    def test_starting_hand_card_mask_marks_only_visible_cards(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)

        masks = create_factored_action_masks(state)

        self.assertTrue(masks.action_mask[ActionType.END_TURN])
        self.assertTrue(masks.action_mask[ActionType.PLAY_CARD])
        for card_index in range(5):
            self.assertTrue(masks.card_mask[card_index])
        for card_index in range(5, CARD_INDEX_SIZE):
            self.assertFalse(masks.card_mask[card_index])

    def test_enemy_target_cards_use_enemy_slots(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=2)

        masks = create_factored_action_masks(state)

        for card_index in [0, 1, 2]:
            self.assertTrue(masks.target_mask[card_index][0])
            self.assertTrue(masks.target_mask[card_index][1])
            self.assertFalse(masks.target_mask[card_index][NO_TARGET_INDEX])

    def test_self_target_cards_use_no_target_slot(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)

        masks = create_factored_action_masks(state)

        for card_index in [3, 4]:
            self.assertFalse(masks.target_mask[card_index][0])
            self.assertTrue(masks.target_mask[card_index][NO_TARGET_INDEX])

    def test_valid_factored_actions(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)
        masks = create_factored_action_masks(state)

        self.assertTrue(
            is_valid_factored_action(
                FactoredAction(ActionType.END_TURN, NO_CARD_INDEX, NO_TARGET_INDEX),
                masks,
            )
        )
        self.assertTrue(
            is_valid_factored_action(
                FactoredAction(ActionType.PLAY_CARD, 0, 0),
                masks,
            )
        )
        self.assertTrue(
            is_valid_factored_action(
                FactoredAction(ActionType.PLAY_CARD, 3, NO_TARGET_INDEX),
                masks,
            )
        )

    def test_invalid_factored_actions(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)
        masks = create_factored_action_masks(state)

        self.assertFalse(
            is_valid_factored_action(
                FactoredAction(ActionType.END_TURN, 0, NO_TARGET_INDEX),
                masks,
            )
        )
        self.assertFalse(
            is_valid_factored_action(
                FactoredAction(ActionType.PLAY_CARD, 0, NO_TARGET_INDEX),
                masks,
            )
        )
        self.assertFalse(
            is_valid_factored_action(
                FactoredAction(ActionType.PLAY_CARD, 3, 0),
                masks,
            )
        )

    def test_decode_factored_end_turn(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)

        action = decode_factored_action(
            FactoredAction(ActionType.END_TURN, NO_CARD_INDEX, NO_TARGET_INDEX),
            state,
        )

        self.assertEqual(action.name, "end_turn")
        self.assertEqual(action.args, {})

    def test_decode_factored_play_card(self) -> None:
        state = shrinker_beetle_state(hand_size=5, enemy_count=1)

        action = decode_factored_action(
            FactoredAction(ActionType.PLAY_CARD, 0, 0),
            state,
        )

        self.assertEqual(action.name, "play_card")
        self.assertEqual(action.args, {"card_index": 0, "target_index": 0})
