"""Backtest runner for HIFA portfolio policies.

``run_backtest`` drives an arbitrary policy through a Gymnasium-style
environment until termination, without ever updating gradients, and returns
the full portfolio value / turnover / action history plus the metrics from
:mod:`src.evaluation.metrics`.

The policy argument is intentionally duck-typed so the exact same function
can evaluate:

- A plain Python callable ``obs -> action`` (e.g. a buy-and-hold baseline).
- A Stable-Baselines3 model exposing ``.predict(obs, deterministic=...)``
  (e.g. B4 PPO-no-LLM).
- A ``torch.nn.Module`` (e.g. ``PortfolioPolicyLLM``), called under
  ``torch.no_grad()`` so backtesting never accumulates gradients / autograd
  graphs.

This module also exposes a best-effort CLI (``python -m
src.evaluation.backtest --config configs/phase_a.yaml``). As of writing,
``src/envs`` and `src/agents`` (owned by other in-progress work) only
contain empty ``__init__.py`` files, so the CLI's "load real env + model"
step is wrapped in a broad try/except that prints a clear warning and exits
cleanly instead of crashing. The CLI is NOT required for
``tests/test_metrics.py`` to pass.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.evaluation.metrics import compute_all_metrics

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is a hard requirement of the
    # project (see requirements.txt) but we keep this import defensive so
    # that importing this module never hard-crashes in a minimal test env.
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Hard safety cap on the number of steps a single episode may run for, in
# case an environment never sets terminated/truncated. This should never be
# hit by a well-behaved env; it only guards against a runaway loop.
_MAX_STEPS_SAFETY_CAP = 200_000


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
        except Exception:  # noqa: BLE001 - best-effort reproducibility only
            pass


def _reset_env(env: Any, seed: Optional[int]) -> Any:
    """Call env.reset(), tolerating both the Gymnasium API (obs, info) and
    older Gym APIs (obs only), and environments whose reset() doesn't
    accept a seed kwarg.
    """
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        obs, _info = result
        return obs
    return result


def _step_env(env: Any, action: Any):
    """Call env.step(), tolerating both the 5-tuple Gymnasium API
    (obs, reward, terminated, truncated, info) and the legacy 4-tuple API
    (obs, reward, done, info).
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    elif len(result) == 4:
        obs, reward, done, info = result
        done = bool(done)
    else:
        raise ValueError(
            f"Unexpected env.step() return signature with {len(result)} elements "
            "(expected 4 or 5)."
        )
    return obs, reward, done, info


def _predict_action(policy_or_action_fn: Any, obs: Any, deterministic: bool) -> Any:
    """Dispatch obs -> action across the policy interfaces this evaluation
    pipeline needs to support (see module docstring).
    """
    # 1) Stable-Baselines3 style: model.predict(obs, deterministic=...) -> (action, state)
    if hasattr(policy_or_action_fn, "predict") and callable(policy_or_action_fn.predict):
        result = policy_or_action_fn.predict(obs, deterministic=deterministic)
        action = result[0] if isinstance(result, tuple) else result
        return action

    # 2) torch.nn.Module: never let gradients accumulate during backtesting.
    if _TORCH_AVAILABLE and isinstance(policy_or_action_fn, torch.nn.Module):
        policy_or_action_fn.eval()
        with torch.no_grad():
            output = policy_or_action_fn(obs)
        if isinstance(output, tuple):
            output = output[0]
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()
        return output

    # 3) Plain callable (e.g. buy-and-hold, a lambda, a hand-wrapped LLM adapter).
    if callable(policy_or_action_fn):
        try:
            return policy_or_action_fn(obs, deterministic=deterministic)
        except TypeError:
            return policy_or_action_fn(obs)

    raise TypeError(
        f"policy_or_action_fn of type {type(policy_or_action_fn)} is neither callable, "
        "nor a torch.nn.Module, nor exposes a .predict() method."
    )


def _to_plain_action(action: Any) -> Any:
    """Normalize an action to something JSON/CSV-friendly for history logs."""
    if _TORCH_AVAILABLE and torch is not None and torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    if isinstance(action, np.ndarray):
        return action.tolist()
    return action


def run_backtest(
    policy_or_action_fn: Any,
    env: Any,
    seed: Optional[int] = None,
    deterministic: bool = True,
    benchmark_values: Optional[list[float]] = None,
) -> dict:
    """Run a policy through an environment end-to-end, without any gradient
    updates, and collect the full backtest history + metrics.

    Args:
        policy_or_action_fn: See module docstring for supported policy
            interfaces (plain callable / SB3 .predict / torch.nn.Module).
        env: A Gymnasium-style environment (must support reset()/step()).
        seed: Optional seed, passed to env.reset() and used to seed
            python/numpy/torch RNGs for reproducibility.
        deterministic: Forwarded to policies that support it (SB3-style, or
            any callable accepting a ``deterministic`` kwarg).
        benchmark_values: Optional benchmark value history forwarded to
            :func:`compute_all_metrics`.

    Returns:
        dict with keys ``portfolio_value_history``, ``turnover_history``,
        ``action_history`` and ``metrics`` (the output of
        :func:`compute_all_metrics`).
    """
    _seed_everything(seed)

    obs = _reset_env(env, seed)

    portfolio_value_history: list[float] = []
    turnover_history: list[float] = []
    action_history: list[Any] = []

    # If the env exposes portfolio_value_history right after reset (as
    # PortfolioEnvPhaseA does, seeded with initial_cash), start from there
    # so the returned history includes the initial value.
    env_history = getattr(env, "portfolio_value_history", None)
    if isinstance(env_history, list) and len(env_history) > 0:
        portfolio_value_history = [float(v) for v in env_history]

    done = False
    steps = 0
    while not done and steps < _MAX_STEPS_SAFETY_CAP:
        action = _predict_action(policy_or_action_fn, obs, deterministic)
        action_history.append(_to_plain_action(action))

        obs, _reward, done, info = _step_env(env, action)
        steps += 1

        if isinstance(info, dict) and "portfolio_value" in info:
            portfolio_value_history.append(float(info["portfolio_value"]))
        if isinstance(info, dict) and "turnover" in info:
            turnover_history.append(float(info["turnover"]))

    if steps >= _MAX_STEPS_SAFETY_CAP:
        logger.warning(
            "run_backtest: hit the safety cap of %d steps without env "
            "signalling done/terminated/truncated. Stopping early.",
            _MAX_STEPS_SAFETY_CAP,
        )

    # Prefer the env's own tracked history if it is available and non-trivial
    # (covers envs that track it internally but don't echo it via info every
    # step).
    env_history = getattr(env, "portfolio_value_history", None)
    if isinstance(env_history, list) and len(env_history) > len(portfolio_value_history):
        portfolio_value_history = [float(v) for v in env_history]

    env_turnover_history = getattr(env, "turnover_history", None)
    if isinstance(env_turnover_history, list) and len(env_turnover_history) > len(
        turnover_history
    ):
        turnover_history = [float(v) for v in env_turnover_history]

    metrics = compute_all_metrics(portfolio_value_history, benchmark_values)

    return {
        "portfolio_value_history": portfolio_value_history,
        "turnover_history": turnover_history,
        "action_history": action_history,
        "metrics": metrics,
    }


def run_backtest_multi_seed(
    policy_fn_factory: Callable[[int], Any],
    env_factory: Callable[[int], Any],
    seeds: list[int],
    deterministic: bool = True,
    benchmark_values: Optional[list[float]] = None,
) -> list[dict]:
    """Run ``run_backtest`` across multiple seeds.

    ``policy_fn_factory`` and ``env_factory`` are factories (seed -> policy,
    seed -> env) rather than pre-built objects, because a policy may need to
    be re-initialized per seed (fresh weights / RNG state) and an env may
    hold per-episode state that shouldn't be reused across independent runs.

    Args:
        policy_fn_factory: Callable seed -> policy_or_action_fn.
        env_factory: Callable seed -> env.
        seeds: List of seeds to evaluate.
        deterministic: Forwarded to :func:`run_backtest`.
        benchmark_values: Forwarded to :func:`run_backtest`.

    Returns:
        List of ``run_backtest`` result dicts, one per seed, in the same
        order as ``seeds``.
    """
    results = []
    for seed in seeds:
        policy = policy_fn_factory(seed)
        env = env_factory(seed)
        result = run_backtest(
            policy,
            env,
            seed=seed,
            deterministic=deterministic,
            benchmark_values=benchmark_values,
        )
        results.append(result)
    return results


# --------------------------------------------------------------------------- #
# Best-effort CLI
# --------------------------------------------------------------------------- #


def _load_yaml_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _try_build_env_and_policy(config: dict, checkpoint: Optional[str]):
    """Best-effort attempt to build a real env + trained policy from the
    project's (still-in-progress) env/agent/training modules.

    Intentionally broad exception handling: `src/envs`, `src/agents` and
    `src/training` are being developed in parallel by other work and may not
    exist, may not expose the expected names, or the checkpoint file may not
    exist yet. None of that should crash this CLI -- we print a clear
    warning and let the caller fall back to a no-op run.

    Returns:
        (env, policy) tuple, or (None, None) if anything is missing.
    """
    try:
        from src.envs.portfolio_env import PortfolioEnvPhaseA  # type: ignore
    except Exception as exc:
        print(
            f"[WARN] Could not import PortfolioEnvPhaseA from src.envs.portfolio_env "
            f"({type(exc).__name__}: {exc}). The env module is likely still under "
            "development. Skipping real backtest run."
        )
        return None, None

    data_cfg = config.get("data", {})
    processed_path = data_cfg.get("processed_path")
    if not processed_path or not Path(processed_path).exists():
        print(
            f"[WARN] Processed data file not found ({processed_path!r}). "
            "Run the data pipeline (fetch_ohlcv.py / build_indicators.py) first. "
            "Skipping real backtest run."
        )
        return None, None

    try:
        import pandas as pd

        state_df = pd.read_parquet(processed_path)
        tickers = data_cfg.get("tickers", [])
        env_cfg = config.get("env", {})
        env = PortfolioEnvPhaseA(
            state_df,
            tickers,
            initial_cash=env_cfg.get("initial_cash", 1_000_000_000),
            transaction_cost=env_cfg.get("transaction_cost", 0.002),
            price_limit=env_cfg.get("price_limit", 0.07),
        )
    except Exception as exc:
        print(
            f"[WARN] Failed to construct PortfolioEnvPhaseA ({type(exc).__name__}: {exc}). "
            "Skipping real backtest run."
        )
        return None, None

    policy = None
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            print(
                f"[WARN] Checkpoint path does not exist: {checkpoint}. "
                "Falling back to a random/no-op policy is not meaningful here, "
                "so no policy will be evaluated."
            )
        else:
            try:
                from src.agents.super_agent_head import PortfolioPolicyLLM  # type: ignore

                model_cfg = config.get("model", {})
                policy = PortfolioPolicyLLM(
                    base_model_name=model_cfg.get("base_model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                    n_assets=len(tickers),
                )
                if _TORCH_AVAILABLE:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    policy.load_state_dict(state_dict)
            except Exception as exc:
                print(
                    f"[WARN] Failed to load policy checkpoint from {checkpoint} "
                    f"({type(exc).__name__}: {exc}). The agent module / checkpoint "
                    "format may not be finalized yet."
                )
                policy = None
    else:
        print("[WARN] No --checkpoint given; skipping policy load.")

    if policy is None:
        return env, None
    return env, policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a HIFA backtest (best-effort CLI; see module docstring)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/phase_a.yaml", help="Path to a phase YAML config."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional path to a trained policy checkpoint."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None, help="Override the seeds list from the config."
    )
    args = parser.parse_args()

    try:
        config = _load_yaml_config(args.config)
    except FileNotFoundError:
        print(f"[WARN] Config file not found: {args.config}. Proceeding with an empty config.")
        config = {}
    except Exception as exc:
        print(f"[WARN] Failed to parse YAML config {args.config} ({type(exc).__name__}: {exc}).")
        config = {}

    seeds = args.seeds or config.get("training", {}).get("seeds", [0])

    env, policy = _try_build_env_and_policy(config, args.checkpoint)

    if env is None or policy is None:
        print(
            "[INFO] backtest CLI finished in best-effort/no-op mode: real env and/or "
            "trained policy were not available (see warnings above). This is expected "
            "while src/envs, src/agents and src/training are still under development. "
            "The programmatic API (run_backtest / run_backtest_multi_seed) is fully "
            "functional and covered by tests/test_metrics.py."
        )
        return

    def policy_fn_factory(_seed: int):
        return policy

    def env_factory(_seed: int):
        return env

    results = run_backtest_multi_seed(policy_fn_factory, env_factory, seeds)
    for seed, result in zip(seeds, results):
        print(f"[seed={seed}] metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
