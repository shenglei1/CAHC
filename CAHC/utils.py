from __future__ import annotations

import numbers

import numpy as np


def merge_stat(source: dict, destination: dict) -> None:
    for key, value in source.items():
        if key not in destination:
            destination[key] = value
        elif isinstance(value, numbers.Number) or isinstance(value, np.ndarray):
            destination[key] = destination[key] + value
        elif isinstance(destination[key], list) and isinstance(value, list):
            destination[key].extend(value)
        elif isinstance(destination[key], list):
            destination[key].append(value)
        else:
            destination[key] = [destination[key], value]
