# Plan: Observe the Model + Barebones PPO Loop

**Status (2026-07-12): done.** All steps below are implemented and verified.
One discovery during implementation: playing Survivor ("Gain Block. Discard 1
card.") pauses the combat on a mid-combat `card_select` decision that used to
end episodes early and score the vanished enemy list as phantom damage reward.
`SilentCombatEnv` now auto-answers those prompts (skip when allowed, else pick
the first required cards); giving the agent control over that choice is a
possible future action-space extension. Post-victory `card_reward` screens are
unrelated and still end the episode as a win, and every env reset still
restarts the simulator process.

Goal for this step: understand what the simulator gives us, add a way to *watch* a
trained model play, and keep the training loop as close to CleanRL `ppo.py` as
possible. No new abstractions, no new dependencies.

## 1. What we learned about the simulator

`external/sts2-cli` runs the real STS2 engine headless (`sts2.dll`, IL patched)
behind a JSON-line stdin/stdout protocol:

- Launch: `dotnet run [--no-build] --project external/sts2-cli/src/Sts2Headless/Sts2Headless.csproj`.
  It prints a `{"type": "ready"}` line, then answers one JSON response per command.
- Commands routed in `src/Sts2Headless/Program.cs:113`: `start_run`, `action`,
  `enter_room`, `get_map`, `set_player`, `set_draw_order`, `load_save`,
  `write_continue_save`, `quit`.
- Combat-relevant actions (`RunSimulator.cs:863`): `play_card`, `end_turn`,
  `use_potion`, plus non-combat ones (`select_map_node`, `choose_option`, ...).
- Every response is a decision point: `combat_play`, `map_select`, `card_reward`,
  `game_over`, etc. Our env treats anything other than `combat_play` as terminal.
- `enter_room` with `type: "combat"` accepts a named `encounter`
  (default `SHRINKER_BEETLE_WEAK`, `RunSimulator.cs:391`), which is how the env
  jumps straight into a fixed combat without walking the map.
- Test hooks we are not using yet but should know exist: `set_player`
  (hp/gold/deck) and `set_draw_order` (force draw order). These make deterministic
  eval scenarios and curriculum combats possible later.
- The interactive player `external/sts2-cli/python/play.py` plus
  `python/game_log.py` (JSONL run logs) show what a human-readable trace of a
  combat looks like — good reference for our watch script output.

Our Python side already wraps this correctly:

- `spire_bot/envs/simulator_client.py` — subprocess + JSON-line client, skips
  non-JSON log lines, fails loudly on EOF with diagnostics.
- `spire_bot/envs/silent_combat_env.py` — one combat per episode, restarts the
  simulator process on every reset (correct but slow: each reset pays `dotnet run`
  startup; this is the known MVP cost, not something to fix in this step).
- `spire_bot/envs/test_simulator.py` — hand-driven smoke script with a good
  `print_state` pretty-printer we should promote to shared code.

## 2. Observing the model: a watch script

There is currently **no way to see a trained policy play** — `train_ppo.py`
never saves weights and there is no eval/rollout viewer. Plan:

### 2a. Promote the pretty-printer to a shared module

- New file `spire_bot/envs/render.py` containing `print_state(state)` (and its
  `_format_stats` / `_format_intents` / `_format_powers` helpers), moved out of
  `spire_bot/envs/test_simulator.py`. `test_simulator.py` imports it instead.
- Pure string formatting over raw simulator JSON — no env or torch imports, so
  it stays usable from tests, the watch script, and debug prints.

### 2b. Save checkpoints from training

Minimal change to `spire_bot/train_ppo.py`:

- `--save-path` argument (default `checkpoints/ppo_silent.pt`).
- `torch.save(agent.state_dict(), path)` once at the end of training (and that's
  it — no schedulers/optimizers/resume logic for now).
- Add `checkpoints/` to `.gitignore`.

### 2c. `spire_bot/watch_ppo.py` — load a checkpoint and watch it play

- Args: `--checkpoint`, `--episodes` (default 3), `--encounter`, `--seed`,
  `--sample` (default off = deterministic argmax over masked logits, per the
  eval guidance in CLAUDE.md), `--random` (masked-uniform baseline, no
  checkpoint needed — useful sanity comparison).
- Loop per step: `print_state(info["state"])`, print the valid-action mask, the
  chosen action decoded to human terms ("play [2] Neutralize -> enemy 0" /
  "end turn"), and the reward breakdown from `spire_bot/envs/rewards.reward_breakdown`
  so shaping is visible turn by turn.
- Per-episode and final summary: win/loss, return, combat length, HP remaining,
  HP lost — the gameplay metrics CLAUDE.md asks us to track.
- Reuses the `Agent` class imported from `train_ppo` (no duplicate model code).

This one script covers both "did training do anything" (metrics vs. `--random`)
and "what is it actually doing" (turn-by-turn trace).

## 3. Training loop: keep it, don't rewrite it

`spire_bot/train_ppo.py` is already a faithful single-env CleanRL `ppo.py`
reduction: rollout buffers, GAE, clipped surrogate + value loss + entropy bonus,
minibatch epochs, grad clipping — with the only intentional additions being
action masking and `--debug-resets`. That *is* the barebones loop we want, so
the plan is small deltas, not a rewrite:

1. Checkpoint saving (2b above).
2. Episode-end print gains gameplay metrics: append `won=<bool>` (derived from
   the terminal state, same logic as `rewards._combat_won`) and
   `hp_remaining=<int>` to the existing `global_step=... episode_return=...` line.
3. Nothing else. No vectorized envs, no LR annealing, no KL early-stop, no
   tensorboard until the env + reward are proven out with the watch script.

Known simplification to keep in mind (not blocking): with `restart_on_reset`,
episodes are slow, so `--num-steps` spanning multiple episodes means most
wall-clock time is `dotnet run` startup. The fix (built binary and/or a
simulator-side `reset_combat`) is the *next* step, per CLAUDE.md performance
guidance — not this one.

## 4. Order of work + verification

1. Extract `render.py`; run `python -m spire_bot.envs.test_simulator` still works.
2. Add checkpoint saving; run the smoke test
   `python -m spire_bot.train_ppo --total-timesteps 64 --num-steps 32 --debug-resets`
   and confirm a `.pt` file appears.
3. Add `watch_ppo.py`; run `python -m spire_bot.watch_ppo --random --episodes 1`
   (no checkpoint) then `--checkpoint checkpoints/ppo_silent.pt --episodes 1`.
   Verify: turn-by-turn output is readable, chosen actions are always valid,
   metrics summary prints.
4. Run `python -m unittest tests.test_env_helpers` (should be untouched/green).
