from __future__ import annotations

from pathlib import Path

from env_wrappers import LBFWrapper
from scenario_registry import load_registry
from shared_lbf_env import make_lbf_env


PROJECT = Path(__file__).resolve().parent


def init(env_name, args):
    if not env_name.startswith("lbf_"):
        raise ValueError("Formal CAHC accepts only registered lbf_ scenarios.")
    scenario = env_name[4:-3] if env_name.endswith("-v0") else env_name[4:]
    registry = load_registry(PROJECT / "scenario_registry.json")
    allowed = (
        registry["formal_scenario_ids"]
        + registry["supplementary_scenario_ids"]
        + registry["reviewer_probe_ids"]
    )
    if scenario not in allowed:
        raise ValueError(f"Unknown formal scenario {scenario!r}; allowed={allowed}")
    env = LBFWrapper(make_lbf_env(scenario), args)
    args.nagents = env.n_agents
    args.nfriendly = env.n_agents
    args.nenemies = 0
    args.enemy_comm = False
    return env
