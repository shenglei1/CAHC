from itertools import product
import math


def paired_sign_flip(differences):
    values = tuple(float(x) for x in differences)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Finite paired differences required')
    observed = abs(sum(values))
    extreme = sum(abs(sum(s*x for s,x in zip(signs,values))) >= observed - 1e-12
                  for signs in product((-1,1), repeat=len(values)))
    return extreme / (2 ** len(values))


def holm(p_values):
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in p_values):
        raise ValueError('Invalid p-value')
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [None]*len(order)
    previous = 0.0
    for rank,i in enumerate(order):
        previous = max(previous, min(1.0, (len(order)-rank)*p_values[i]))
        adjusted[i] = previous
    return adjusted
