"""Minimal masked PPO trainer for the Silent combat proof of concept."""

from __future__ import annotations

import argparse
import random
import time

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical

    from spire_bot.envs.actions import ACTION_SPACE_SIZE
    from spire_bot.envs.observations import OBSERVATION_SIZE
    from spire_bot.envs.silent_combat_env import SilentCombatEnv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing PPO dependencies. Install them with something like:\n"
        "  pip install gymnasium numpy torch\n"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--encounter", type=str, default="SHRINKER_BEETLE_WEAK")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug-resets", action="store_true", help="Print compact simulator state after each env reset.")
    return parser.parse_args()


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def print_reset_debug(label: str, info: dict) -> None:
    state = info.get("state") or {}
    context = state.get("context") or {}
    player = state.get("player") or {}
    hand = state.get("hand") or []
    enemies = state.get("enemies") or []
    raw_action_mask = info.get("action_mask")
    action_mask = np.asarray([] if raw_action_mask is None else raw_action_mask, dtype=np.bool_)
    valid_actions = np.flatnonzero(action_mask).tolist()

    enemy_summary = ", ".join(
        f"{enemy.get('index', '?')}:{enemy.get('id', enemy.get('name', '?'))} "
        f"hp={enemy.get('hp', '?')}/{enemy.get('max_hp', '?')}"
        for enemy in enemies
    )

    print(
        f"[debug:{label}] "
        f"decision={state.get('decision')} "
        f"room_type={state.get('room_type') or context.get('room_type')} "
        f"player_hp={player.get('hp', '?')}/{player.get('max_hp', '?')} "
        f"hand={len(hand)} "
        f"enemies={len(enemies)} "
        f"valid_actions={valid_actions} "
        f"enemy_summary={enemy_summary or 'none'}"
    )


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(OBSERVATION_SIZE, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_SPACE_SIZE), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBSERVATION_SIZE, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        masked_logits = logits.masked_fill(~action_mask.bool(), -1e9)
        distribution = Categorical(logits=masked_logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(obs)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    env = SilentCombatEnv(encounter=args.encounter, base_seed=f"ppo-{args.seed}")
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = max(args.total_timesteps // args.num_steps, 1)

    obs_buf = torch.zeros((args.num_steps, OBSERVATION_SIZE), device=device)
    mask_buf = torch.zeros((args.num_steps, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)
    actions_buf = torch.zeros(args.num_steps, dtype=torch.long, device=device)
    logprobs_buf = torch.zeros(args.num_steps, device=device)
    rewards_buf = torch.zeros(args.num_steps, device=device)
    dones_buf = torch.zeros(args.num_steps, device=device)
    values_buf = torch.zeros(args.num_steps, device=device)

    next_obs, info = env.reset(seed=args.seed)
    if args.debug_resets:
        print_reset_debug("initial_reset", info)
    episode_return = 0.0
    episode_length = 0
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        for step in range(args.num_steps):
            action_mask = info["action_mask"]
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)

            obs_buf[step] = obs_tensor.squeeze(0)
            mask_buf[step] = mask_tensor.squeeze(0)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor, mask_tensor)
                values_buf[step] = value.flatten()

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            rewards_buf[step] = float(reward)
            dones_buf[step] = float(done)

            episode_return += float(reward)
            episode_length += 1
            global_step += 1

            if done:
                print(
                    f"global_step={global_step} "
                    f"episode_return={episode_return:.2f} "
                    f"episode_length={episode_length}"
                )
                next_obs, info = env.reset()
                if args.debug_resets:
                    print_reset_debug("episode_reset", info)
                episode_return = 0.0
                episode_length = 0

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = agent.get_value(next_obs_tensor).reshape(-1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values_buf[t + 1]
                nextnonterminal = 1.0 - dones_buf[t]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        indices = np.arange(batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = indices[start : start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_buf[mb_inds],
                    mask_buf[mb_inds],
                    actions_buf[mb_inds],
                )
                logratio = newlogprob - logprobs_buf[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / max(time.time() - start_time, 1e-6))
        print(
            f"update={update}/{num_updates} "
            f"loss={loss.item():.3f} "
            f"policy_loss={pg_loss.item():.3f} "
            f"value_loss={v_loss.item():.3f} "
            f"entropy={entropy_loss.item():.3f} "
            f"sps={sps}"
        )

    env.close()


if __name__ == "__main__":
    main()
