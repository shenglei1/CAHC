import sys
import os
import importlib
from env_wrappers import LBFWrapper


def init(env_name, args, final_init=True):
    if not env_name.startswith("lbf_"):
        raise ValueError("HyperComm formal runs accept only registered lbf_ scenarios.")
    scenario = env_name[4:-3] if env_name.endswith("-v0") else env_name[4:]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Formal support files are bundled with HyperComm so the complete folder can
    # be copied to Windows without relying on a particular parent directory.
    support_dir = current_dir
    sys.path.insert(0, support_dir)
    try:
        registry_module = importlib.import_module("scenario_registry")
        registry = registry_module.load_registry(
            os.path.join(support_dir, "scenario_registry.json")
        )
        experiment_ids = (
            registry["formal_scenario_ids"]
            + registry["supplementary_scenario_ids"]
            + registry["reviewer_probe_ids"]
        )
        if scenario not in experiment_ids:
            raise ValueError(
                f"Unknown formal scenario '{scenario}'. Allowed: {experiment_ids}"
            )
        formal_env = importlib.import_module("shared_lbf_env")
        env = formal_env.make_lbf_env(scenario)
    finally:
        sys.path.pop(0)
    env = LBFWrapper(env, args)
    args.nagents = env.n_agents
    args.nfriendly = args.nagents
    args.nenemies = 0
    args.enemy_comm = False
    return env
