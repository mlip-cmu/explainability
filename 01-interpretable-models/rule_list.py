"""A small greedy learner for rule lists: IF ... THEN arrest, ELSE IF ..., ELSE no arrest."""

from itertools import combinations

import pandas as pd


def learn_rule_list(C: pd.DataFrame, y: pd.Series, max_rules: int = 3, min_support: int = 100):
    """Each rule is one or two yes/no conditions. Take the rule that corrects most errors."""
    candidates = [(c,) for c in C.columns] + list(combinations(C.columns, 2))
    remaining = pd.Series(True, index=C.index)
    rules = []
    for _ in range(max_rules):
        best, best_gain = None, 0
        for rule in candidates:
            covered = remaining & C[list(rule)].all(axis=1)
            gain = (2 * y[covered] - 1).sum()  # arrests minus non-arrests among the covered
            if covered.sum() >= min_support and gain > best_gain:
                best, best_gain = rule, gain
        if best is None:
            break
        covered = remaining & C[list(best)].all(axis=1)
        rules.append((best, int(covered.sum()), float(y[covered].mean())))
        remaining &= ~covered
    return rules


def predict_rule_list(rules, C: pd.DataFrame) -> pd.Series:
    prediction = pd.Series(0, index=C.index)
    for rule, _, _ in rules:
        prediction[C[list(rule)].all(axis=1)] = 1
    return prediction
