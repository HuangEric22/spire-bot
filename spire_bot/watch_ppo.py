"""Watch a trained (or random) policy play Silent combats turn by turn."""

from __future__ import annotations

import argparse

try:
    import numpy as np
    import torch

    from spire_bot.envs.actions import END_TURN_ACTION
    from spire_bot.envs.render import print_state
    from spire_bot.envs.rewards import combat_won, reward_breakdown
    from spire_bot.envs.silent_combat_env import SilentCombatEnv
    from spire_bot.train_ppo import Agent
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with something like:\n"
        "  pip install gymnasium numpy torch\n"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_silent.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--encounter", type=str, default="SHRINKER_BEETLE_WEAK")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample", action="store_true", help="Sample from the masked policy instead of argmax.")
    parser.add_argument("--random", action="store_true", help="Play uniform random valid actions; no checkpoint needed.")
    parser.add_argument("--quiet", action="store_true", help="Skip the turn-by-turn trace; print only episode results and the summary.")
    return parser.parse_args()


def describe_action(action: int, state: dict) -> str:
    if action == END_TURN_ACTION:
        return "end turn"

    hand_index = action - 1
    card = next((c for c in state.get("hand") or [] if c.get("index") == hand_index), {})
    text = f"play [{hand_index}] {card.get('name', '?')}"
    if card.get("target_type") == "AnyEnemy":
        enemies = state.get("enemies") or []
        if enemies:
            text += f" -> enemy [{enemies[0].get('index', 0)}] {enemies[0].get('name', '?')}"
    return text


def choose_action(
    agent: Agent | None,
    obs: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    sample: bool,
) -> int:
    if agent is None:
        return int(rng.choice(np.flatnonzero(mask)))

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits = agent.actor(obs_tensor).masked_fill(~mask_tensor, -1e9)
    if sample:
        return int(torch.distributions.Categorical(logits=logits).sample().item())
    return int(logits.argmax(dim=-1).item())


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    agent = None
    if not args.random:
        agent = Agent()
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        agent.eval()
        print(f"loaded checkpoint {args.checkpoint} (argmax={not args.sample})")
    else:
        print("playing uniform random valid actions")

    env = SilentCombatEnv(encounter=args.encounter, base_seed=f"watch-{args.seed}")
    results = []
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed if episode == 1 else None)
        state = info["state"]
        start_hp = (state.get("player") or {}).get("hp", 0)
        episode_return = 0.0
        steps = 0
        done = False
        won = False

        if not args.quiet:
            print(f"\n######## episode {episode} ########")
        while not done:
            mask = np.asarray(info["action_mask"], dtype=np.bool_)
            action = choose_action(agent, obs, mask, rng, args.sample)
            if not args.quiet:
                print_state(state)
                print(f"valid_actions={np.flatnonzero(mask).tolist()}")
                print(f"chosen: {action} ({describe_action(action, state)})")

            previous_state = state
            obs, reward, terminated, truncated, info = env.step(action)
            state = info["state"]
            steps += 1
            episode_return += float(reward)
            done = terminated or truncated

            if not args.quiet:
                pieces = reward_breakdown(previous_state, state)
                print(
                    f"reward={reward:.2f} "
                    f"(damage={pieces.damage_dealt:.2f} hp_lost={pieces.hp_lost:.2f} "
                    f"step={pieces.step:.2f} terminal={pieces.terminal:.2f})"
                )
            if done:
                won = combat_won(previous_state, state)

        final_hp = ((state or {}).get("player") or {}).get("hp", 0)
        results.append(
            {
                "won": won,
                "return": episode_return,
                "length": steps,
                "hp_remaining": final_hp,
                "hp_lost": start_hp - final_hp,
            }
        )
        print(
            f"\nepisode {episode}: won={won} return={episode_return:.2f} "
            f"length={steps} hp_remaining={final_hp} hp_lost={start_hp - final_hp}"
        )

    env.close()

    wins = sum(r["won"] for r in results)
    count = len(results)
    print("\n######## summary ########")
    print(
        f"episodes={count} wins={wins} win_rate={wins / count:.2f} "
        f"avg_return={np.mean([r['return'] for r in results]):.2f} "
        f"avg_length={np.mean([r['length'] for r in results]):.1f} "
        f"avg_hp_remaining={np.mean([r['hp_remaining'] for r in results]):.1f} "
        f"avg_hp_lost={np.mean([r['hp_lost'] for r in results]):.1f}"
    )


if __name__ == "__main__":
    main()
