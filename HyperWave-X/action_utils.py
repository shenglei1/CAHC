from __future__ import annotations

import numpy as np
import torch


def parse_action_args(args) -> None:
    if len(args.num_actions) != args.dim_actions:
        raise ValueError("num_actions and dim_actions are inconsistent.")
    if any(int(value) <= 0 for value in args.num_actions):
        raise ValueError("Formal HyperWave-X supports discrete actions only.")
    args.continuous = False
    args.naction_heads = [int(value) for value in args.num_actions]


def select_action(args, action_out):
    sampled = []
    for log_probabilities in action_out:
        probabilities = log_probabilities.exp()
        sampled.append(
            torch.stack(
                [torch.multinomial(row, 1).detach() for row in probabilities[0]]
            )
        )
    return torch.stack(sampled)


def translate_action(args, env, action):
    arrays = [head.squeeze(-1).detach().cpu().numpy() for head in action]
    if len(arrays) != 1:
        raise ValueError("The frozen LBF protocol has exactly one action head.")
    actual = [np.asarray(arrays[0], dtype=np.int64)]
    return arrays, actual
