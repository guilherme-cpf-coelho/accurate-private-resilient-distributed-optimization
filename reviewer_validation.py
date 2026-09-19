#!/usr/bin/env python3
"""Reproducible reviewer-requested validation for Access-2026-30726.

The script implements the finite-time detector disclosed in the revised
manuscript, tests its tolerance/threshold sensitivity, probes attacks outside
the limiting-regime theorem, validates augmented EXTRA on a smooth
non-quadratic objective, and sweeps honest-but-curious coalition size using the
canonical-query observability distance.

Run from this directory with

    python reviewer_validation.py --output results

All random generators use recorded deterministic seeds.  CSV files contain the
underlying numbers and reviewer_requested_validation.pdf is the manuscript
figure generated from them.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


SetKey = FrozenSet[int]


@dataclass(frozen=True)
class DetectorConfig:
    """All numerical constants used by the finite-time detector."""

    eps_match: float = 1.0e-3
    tau_cap: float = 0.70
    tau_struct: float = 0.60
    tau_weak: float = 0.15
    min_anchor_members: int = 3
    min_anchor_sizes: int = 2


NOMINAL_CONFIG = DetectorConfig()


_GRAPH_CACHE: Dict[Tuple[int, float, int], np.ndarray] = {}


def powerset_upto(nodes: Sequence[int], f: int) -> List[SetKey]:
    return [
        frozenset(s)
        for size in range(f + 1)
        for s in itertools.combinations(nodes, size)
    ]


def connected(adj: np.ndarray, removed: Iterable[int] = ()) -> bool:
    removed = set(removed)
    remaining = [i for i in range(len(adj)) if i not in removed]
    if not remaining:
        return False
    seen = {remaining[0]}
    stack = [remaining[0]]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(adj[i]):
            j = int(j)
            if j not in removed and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(remaining)


def vertex_connectivity_at_least(adj: np.ndarray, kappa: int) -> bool:
    """Exact small-graph test: removal of fewer than kappa vertices stays connected."""

    n = len(adj)
    if np.min(adj.sum(axis=1)) < kappa:
        return False
    for r in range(kappa):
        for removed in itertools.combinations(range(n), r):
            if not connected(adj, removed):
                return False
    return True


def er_graph(n: int, p: float, rng: np.random.Generator, kappa: int) -> np.ndarray:
    for _ in range(20000):
        upper = rng.random((n, n)) < p
        adj = np.triu(upper, 1).astype(float)
        adj += adj.T
        if vertex_connectivity_at_least(adj, kappa):
            return adj
    raise RuntimeError(f"failed to draw a graph with connectivity >= {kappa} at p={p}")


def complete_graph(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=float) - np.eye(n)


def cycle_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[(i + 1) % n, i] = 1.0
    return adj


def path_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def small_world_graph(
    n: int, degree: int, rewire_probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Undirected Watts--Strogatz-style graph without an external graph package."""

    if degree % 2 or degree >= n:
        raise ValueError("degree must be even and smaller than n")
    for _ in range(100):
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            for offset in range(1, degree // 2 + 1):
                j = (i + offset) % n
                adj[i, j] = adj[j, i] = 1.0
        original = [(i, (i + offset) % n) for i in range(n) for offset in range(1, degree // 2 + 1)]
        for i, j in original:
            if rng.random() >= rewire_probability or adj[i, j] == 0.0:
                continue
            available = [k for k in range(n) if k != i and adj[i, k] == 0.0]
            if not available:
                continue
            replacement = int(rng.choice(available))
            adj[i, j] = adj[j, i] = 0.0
            adj[i, replacement] = adj[replacement, i] = 1.0
        if connected(adj):
            return adj
    raise RuntimeError("failed to generate a connected small-world graph")


def average_clustering(adj: np.ndarray) -> float:
    values = []
    for i in range(len(adj)):
        neighbors = np.flatnonzero(adj[i])
        if len(neighbors) < 2:
            values.append(0.0)
            continue
        edges = np.sum(adj[np.ix_(neighbors, neighbors)]) / 2.0
        values.append(edges / math.comb(len(neighbors), 2))
    return float(np.mean(values))


def metropolis_weights(adj: np.ndarray) -> np.ndarray:
    n = len(adj)
    degree = adj.sum(axis=1)
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.flatnonzero(adj[i]):
            j = int(j)
            weights[i, j] = 1.0 / (1.0 + max(degree[i], degree[j]))
        weights[i, i] = 1.0 - weights[i].sum()
    return weights


def harmonic_observer_value(
    adj: np.ndarray,
    active: Sequence[int],
    boundaries: Mapping[int, float],
    observer: int,
) -> float:
    """Discrete-harmonic extension of boundary broadcasts on an induced graph."""

    if observer in boundaries:
        return float(boundaries[observer])
    free = [i for i in active if i not in boundaries]
    boundary = [i for i in active if i in boundaries]
    index = {node: k for k, node in enumerate(active)}
    local_adj = adj[np.ix_(active, active)]
    lap = np.diag(local_adj.sum(axis=1)) - local_adj
    free_idx = [index[i] for i in free]
    boundary_idx = [index[i] for i in boundary]
    rhs = -lap[np.ix_(free_idx, boundary_idx)] @ np.array(
        [boundaries[i] for i in boundary], dtype=float
    )
    values = np.linalg.solve(lap[np.ix_(free_idx, free_idx)], rhs)
    return float(values[free.index(observer)])


def limiting_candidates(
    adj: np.ndarray,
    y: np.ndarray,
    adversaries: SetKey,
    attacks: Mapping[int, float],
    observer: int,
    f: int,
) -> Dict[SetKey, float]:
    n = len(adj)
    honest = frozenset(range(n)) - adversaries
    candidates: Dict[SetKey, float] = {}
    for excluded in powerset_upto(list(range(n)), f):
        if observer in excluded:
            continue
        active = [i for i in range(n) if i not in excluded]
        surviving = adversaries - excluded
        if not surviving:
            surviving_honest = list(honest - excluded)
            candidates[excluded] = float(np.mean(y[surviving_honest]))
        else:
            boundary_values = {i: attacks[i] for i in surviving}
            if np.ptp(list(boundary_values.values())) <= 1.0e-14:
                candidates[excluded] = float(next(iter(boundary_values.values())))
            else:
                candidates[excluded] = harmonic_observer_value(
                    adj, active, boundary_values, observer
                )
    return candidates


def diameter_clusters(
    candidates: Mapping[SetKey, float], eps_match: float
) -> List[List[Tuple[SetKey, float]]]:
    """One-dimensional complete-linkage clusters with maximum diameter eps_match."""

    ordered = sorted(candidates.items(), key=lambda item: (item[1], tuple(item[0])))
    clusters: List[List[Tuple[SetKey, float]]] = []
    for item in ordered:
        if not clusters or item[1] - clusters[-1][0][1] > eps_match:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def expected_witness_count(n: int, f: int, x_size: int, y_size: int) -> int:
    remaining = n - 1 - x_size - y_size
    total = 0
    for size in range(x_size, f + 1):
        choose = size - x_size
        if 0 <= choose <= remaining:
            total += math.comb(remaining, choose)
    return max(total, 1)


def robust_decode(
    cluster: Sequence[Tuple[SetKey, float]], n: int, observer: int, tau_cap: float
) -> Tuple[SetKey, SetKey]:
    subsets = [entry[0] for entry in cluster]
    universe = set(range(n)) - {observer}
    presence = {i: sum(i in subset for subset in subsets) / len(subsets) for i in universe}
    x_set = frozenset(i for i, ratio in presence.items() if ratio >= tau_cap)
    # Only the intersection is relaxed.  Y retains its mathematical definition
    # as the complement of the union; relaxing both sides would label honest
    # agents that occur in only a few large witnesses as adversarial.
    y_set = frozenset(i for i, ratio in presence.items() if ratio == 0.0)
    return x_set, y_set


def candidate_nearest(
    candidates: Mapping[SetKey, float], key: SetKey
) -> float | None:
    return candidates.get(key)


def finite_time_detector(
    candidates: Mapping[SetKey, float],
    n: int,
    observer: int,
    f: int,
    config: DetectorConfig = NOMINAL_CONFIG,
) -> SetKey:
    """Exact implementation order disclosed in the revised manuscript.

    Order: complete-linkage clustering; robust X/Y decoding; structural
    acceptance; value-disambiguated weak hints; total-collusion fallback;
    singleton incompatibility fallback; adversary-free default.
    """

    clusters = diameter_clusters(candidates, config.eps_match)
    accepted: List[Tuple[SetKey, float, float, int]] = []
    weak: List[Tuple[SetKey, float, float]] = []

    for cluster in clusters:
        sizes = {len(key) for key, _ in cluster}
        if len(cluster) < config.min_anchor_members or len(sizes) < config.min_anchor_sizes:
            continue
        x_set, y_set = robust_decode(cluster, n, observer, config.tau_cap)
        decoded = x_set | y_set
        expected = expected_witness_count(n, f, len(x_set), len(y_set))
        # Count only members compatible with the decoded witness template:
        # every X element must be excluded and every Y element must survive.
        compatible = [
            (key, value)
            for key, value in cluster
            if x_set.issubset(key) and key.isdisjoint(y_set)
        ]
        support_ratio = len(compatible) / expected
        mean_value = float(np.mean([value for _, value in cluster]))
        # A decoded adversary set larger than the declared resilience budget is
        # structurally impossible and is rejected before consolidation.
        if decoded and len(decoded) <= f and support_ratio >= config.tau_struct:
            accepted.append((decoded, mean_value, support_ratio, len(compatible)))
        elif decoded and len(decoded) <= f and support_ratio >= config.tau_weak:
            weak.append((decoded, mean_value, support_ratio))

    if accepted:
        # True single-anchor families decode the same full adversary set.  Rank
        # decoded proposals first by the number of independently valued anchors
        # that agree, then by structural coverage.  This prevents a repeated
        # multi-boundary harmonic level from being unioned into the estimate.
        groups: Dict[SetKey, List[Tuple[float, float, int]]] = {}
        for decoded, value, ratio, compatible_count in accepted:
            groups.setdefault(decoded, []).append((value, ratio, compatible_count))
        best = max(
            groups,
            key=lambda key: (
                len(groups[key]),
                max(entry[1] for entry in groups[key]),
                max(entry[2] for entry in groups[key]),
                len(key),
            ),
        )
        estimate = set(best)
        anchor_values = [entry[0] for entry in groups[best]]

        # Weak hints are only needed in the sparse case where every validated
        # proposal is a singleton.  They must be separately supported,
        # disjoint, value-disambiguated, and respect the declared budget.
        sparse_singletons = all(len(key) == 1 for key in groups)
        if sparse_singletons:
            for decoded, cluster_value, support_ratio in sorted(
                weak, key=lambda entry: entry[2], reverse=True
            ):
                if len(decoded) != 1 or not estimate.isdisjoint(decoded):
                    continue
                decoded_value = candidate_nearest(candidates, decoded)
                if decoded_value is None:
                    continue
                if all(abs(decoded_value - value) > config.eps_match for value in anchor_values):
                    estimate.update(decoded)
                if len(estimate) >= f:
                    break
        if len(estimate) <= f:
            return frozenset(estimate)

    # Total-collusion fallback.  It is enabled only for a dominant repeated
    # cluster, preventing a chance pair in the fault-free case from triggering it.
    dominant = max(clusters, key=len)
    if len(dominant) >= max(3, math.ceil(config.tau_struct * len(candidates))):
        malicious_value = float(np.median([value for _, value in dominant]))
        incompatible = [
            key
            for key, value in candidates.items()
            if abs(value - malicious_value) > config.eps_match
        ]
        if incompatible:
            min_size = min(map(len, incompatible))
            smallest = sorted(
                (key for key in incompatible if len(key) == min_size),
                key=lambda key: tuple(key),
            )
            if len(smallest) == 1 and min_size <= f:
                return smallest[0]

    # Singleton incompatibility fallback, used only when one singleton removal
    # is uniquely separated from the dominant candidate value.
    baseline = candidates.get(frozenset())
    singleton = []
    if baseline is not None:
        for i in range(n):
            key = frozenset({i})
            if key in candidates and abs(candidates[key] - baseline) > config.eps_match:
                singleton.append(key)
    if len(singleton) == 1 and f >= 1:
        return singleton[0]

    return frozenset()


REGIMES = (
    ("none", 0, "none"),
    ("one-distinct", 1, "distinct"),
    ("two-distinct", 2, "distinct"),
    ("three-distinct", 3, "distinct"),
    ("two-colluding", 2, "colluding"),
    ("three-colluding", 3, "colluding"),
    ("three-partial", 3, "partial"),
)


def attack_values(
    adversaries: SetKey, mode: str, rng: np.random.Generator
) -> Dict[int, float]:
    nodes = sorted(adversaries)
    if not nodes:
        return {}
    if mode == "colluding":
        value = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 8.0))
        return {i: value for i in nodes}
    if mode == "partial":
        common = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 8.0))
        distinct = float(-np.sign(common) * rng.uniform(3.0, 8.0))
        return {nodes[0]: common, nodes[1]: common, nodes[2]: distinct}
    values: Dict[int, float] = {}
    for position, node in enumerate(nodes):
        sign = -1.0 if position % 2 else 1.0
        values[node] = float(sign * rng.uniform(3.0, 8.0) + 0.25 * position)
    return values


def battery(
    config: DetectorConfig,
    noise_std: float = 1.5e-4,
    seeds_per_cell: int = 3,
    master_seed: int = 20260719,
) -> List[dict]:
    rows: List[dict] = []
    n, f = 10, 3
    for p_index, p in enumerate(np.arange(0.4, 1.01, 0.1)):
        for regime_index, (regime, count, mode) in enumerate(REGIMES):
            for replicate in range(seeds_per_cell):
                seed = master_seed + 10000 * p_index + 100 * regime_index + replicate
                graph_key = (seed, round(float(p), 1), f + 1)
                if graph_key not in _GRAPH_CACHE:
                    _GRAPH_CACHE[graph_key] = er_graph(
                        n, float(p), np.random.default_rng(seed), f + 1
                    )
                adj = _GRAPH_CACHE[graph_key]
                rng = np.random.default_rng(seed + 17)
                y = rng.normal(0.0, 1.5, n)
                adversaries = frozenset(rng.choice(n, size=count, replace=False).tolist())
                attacks = attack_values(adversaries, mode, rng)
                strict_success = True
                for observer in sorted(set(range(n)) - adversaries):
                    values = limiting_candidates(adj, y, adversaries, attacks, observer, f)
                    noisy = {
                        key: float(value + rng.normal(0.0, noise_std))
                        for key, value in values.items()
                    }
                    estimate = finite_time_detector(noisy, n, observer, f, config)
                    if estimate != adversaries:
                        strict_success = False
                        break
                rows.append(
                    {
                        "seed": seed,
                        "p": round(float(p), 1),
                        "regime": regime,
                        "replicate": replicate,
                        "success_all_honest": int(strict_success),
                    }
                )
    return rows


def threshold_sensitivity() -> List[dict]:
    rows: List[dict] = []
    sweeps = {
        "eps_match": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "tau_cap": [0.50, 0.60, 0.70, 0.80, 0.90],
        "tau_struct": [0.40, 0.50, 0.60, 0.70, 0.80],
        "tau_weak": [0.00, 0.10, 0.15, 0.20, 0.30],
    }
    for parameter, values in sweeps.items():
        for value in values:
            kwargs = asdict(NOMINAL_CONFIG)
            kwargs[parameter] = value
            config = DetectorConfig(**kwargs)
            trials = battery(config, seeds_per_cell=1, master_seed=20260800)
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "successes": sum(row["success_all_honest"] for row in trials),
                    "trials": len(trials),
                    "success_rate": np.mean([row["success_all_honest"] for row in trials]),
                }
            )
    return rows


def time_varying_candidates(
    adj: np.ndarray,
    y: np.ndarray,
    adversaries: SetKey,
    base_attacks: Mapping[int, float],
    observer: int,
    f: int,
    kind: str,
    severity: float,
    rng: np.random.Generator,
    horizon: int = 400,
    window: int = 40,
) -> Dict[SetKey, float]:
    """Finite-time, first-order response model for attacks outside Assumption 2."""

    n = len(adj)
    honest = frozenset(range(n)) - adversaries
    exclusions = [s for s in powerset_upto(list(range(n)), f) if observer not in s]
    phases = {a: rng.uniform(0.0, 2.0 * np.pi) for a in adversaries}
    adversary_list = sorted(adversaries)
    attack_series = np.empty((len(adversary_list), horizon), dtype=float)
    time = np.arange(horizon, dtype=float)
    for row, adversary in enumerate(adversary_list):
        base = base_attacks[adversary]
        if kind == "drift":
            attack_series[row] = base + severity * time
        elif kind == "oscillation":
            attack_series[row] = base + 2.0 * np.sin(
                2.0 * np.pi * severity * time + phases[adversary]
            )
        else:
            attack_series[row] = base

    targets = np.empty((len(exclusions), horizon), dtype=float)
    copy_offsets = rng.normal(0.0, severity, len(exclusions))
    for row, excluded in enumerate(exclusions):
        surviving = sorted(adversaries - excluded)
        if not surviving:
            targets[row] = float(np.mean(y[list(honest - excluded)]))
        else:
            indices = [adversary_list.index(a) for a in surviving]
            # The stress graph is complete.  Its harmonic extension is the
            # arithmetic mean of surviving boundary values at every free node.
            targets[row] = np.mean(attack_series[indices], axis=0)
            if kind == "copy-dependent":
                targets[row] += copy_offsets[row]

    betas = np.array(
        [0.035 + 0.010 * (len(excluded) / max(f, 1)) for excluded in exclusions]
    )
    states = targets[:, 0] + rng.normal(0.0, 0.5, len(exclusions))
    tail: List[np.ndarray] = []
    for t in range(horizon):
        states += betas * (targets[:, t] - states)
        if t >= horizon - window:
            tail.append(states.copy())
    means = np.mean(np.vstack(tail), axis=0)
    return {key: float(means[row]) for row, key in enumerate(exclusions)}


def out_of_model_stress() -> List[dict]:
    rows: List[dict] = []
    levels = {
        "drift": [0.0, 2.5e-5, 1.0e-4, 5.0e-4, 2.0e-3],
        "oscillation": [0.0, 2.5e-4, 1.0e-3, 5.0e-3, 2.0e-2],
        "copy-dependent": [0.0, 2.5e-4, 1.0e-3, 5.0e-3, 2.0e-2],
    }
    n, f, trials = 10, 3, 30
    for kind_index, (kind, severities) in enumerate(levels.items()):
        for level_index, severity in enumerate(severities):
            successes = 0
            errors: List[float] = []
            for replicate in range(trials):
                rng = np.random.default_rng(20261000 + 10000 * kind_index + 100 * level_index + replicate)
                adj = complete_graph(n)
                y = rng.normal(0.0, 1.5, n)
                adversaries = frozenset(rng.choice(n, size=2, replace=False).tolist())
                base = attack_values(adversaries, "distinct", rng)
                honest = frozenset(range(n)) - adversaries
                all_correct = True
                trial_errors: List[float] = []
                for observer in sorted(honest):
                    values = time_varying_candidates(
                        adj, y, adversaries, base, observer, f, kind, severity, rng
                    )
                    estimate = finite_time_detector(values, n, observer, f, NOMINAL_CONFIG)
                    all_correct &= estimate == adversaries
                    selected = values.get(estimate, values[frozenset()])
                    trial_errors.append(abs(selected - float(np.mean(y[list(honest)]))))
                successes += int(all_correct)
                errors.append(max(trial_errors))
            rows.append(
                {
                    "attack": kind,
                    "level_index": level_index,
                    "severity": severity,
                    "successes": successes,
                    "trials": trials,
                    "success_rate": successes / trials,
                    "median_max_abs_error": float(np.median(errors)),
                }
            )
    return rows


def augmented_matrix(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(weights)
    matrix = np.zeros((4 * n, 4 * n), dtype=float)
    for i in range(n):
        matrix[4 * i, 0::4] = weights[i]
        matrix[4 * i, 4 * i + 1] = 2.0
        matrix[4 * i, 4 * i + 3] = 1.0
        matrix[4 * i + 1, 4 * i] = 1.0
        matrix[4 * i + 2, 4 * i] = 1.0
        matrix[4 * i + 3, 4 * i + 2] = 1.0
    matrix /= matrix.sum(axis=1, keepdims=True)
    eigval, eigvec = np.linalg.eig(matrix.T)
    left = np.real(eigvec[:, np.argmin(np.abs(eigval - 1.0))])
    left *= np.sign(left.sum())
    left /= left.sum()
    splitters = np.zeros((4 * n, n), dtype=float)
    for i in range(n):
        scale = 1.0 / left[4 * i + 1 : 4 * i + 4].sum()
        splitters[4 * i + 1 : 4 * i + 4, i] = scale
    return matrix, left, splitters


def run_augmented_extra_nonquadratic(
    seed: int, iterations: int = 2500, step_size: float = 0.01
) -> Tuple[np.ndarray, float, float, float]:
    rng = np.random.default_rng(seed)
    n = 6
    centers = rng.normal(0.0, 1.0, n)

    def gradient(physical: np.ndarray) -> np.ndarray:
        residual = physical - centers
        return residual + 0.08 * residual**3

    optimum = brentq(
        lambda value: float(np.sum(gradient(np.full(n, value)))), -20.0, 20.0
    )
    weights = metropolis_weights(complete_graph(n))
    matrix, _, splitters = augmented_matrix(weights)
    identity = np.eye(4 * n)
    half = 0.5 * (identity + matrix)
    previous = np.zeros(4 * n)
    previous_gradient = gradient(previous[0::4])
    current = matrix @ previous - step_size * splitters @ previous_gradient
    errors = [float(np.max(np.abs(previous[0::4] - optimum)))]
    for _ in range(1, iterations):
        current_gradient = gradient(current[0::4])
        following = (
            (identity + matrix) @ current
            - half @ previous
            - step_size * splitters @ (current_gradient - previous_gradient)
        )
        previous, current = current, following
        previous_gradient = current_gradient
        errors.append(float(np.max(np.abs(current[0::4] - optimum))))
    final_error = float(np.max(np.abs(current[0::4] - optimum)))
    disagreement = float(np.ptp(current[0::4]))
    return np.array(errors), optimum, final_error, disagreement


def nonquadratic_validation() -> Tuple[List[dict], np.ndarray]:
    rows: List[dict] = []
    trajectories = []
    for replicate in range(30):
        seed = 20262000 + replicate
        errors, optimum, final_error, disagreement = run_augmented_extra_nonquadratic(seed)
        trajectories.append(errors)
        rows.append(
            {
                "seed": seed,
                "optimum": optimum,
                "final_max_abs_error": final_error,
                "final_disagreement": disagreement,
            }
        )
    return rows, np.vstack(trajectories)


def modal_pencil_radius(matrix: np.ndarray) -> float:
    eigenvalues = list(np.linalg.eigvals(matrix))
    consensus_index = int(np.argmin(np.abs(np.asarray(eigenvalues) - 1.0)))
    eigenvalues.pop(consensus_index)
    roots = []
    for value in eigenvalues:
        roots.extend(np.roots([1.0, -(1.0 + value), 0.5 * (1.0 + value)]))
    return float(np.max(np.abs(roots)))


def quadratic_iterations(
    adj: np.ndarray,
    centers: np.ndarray,
    step_size: float = 0.008,
    tolerance: float = 1.0e-6,
    maximum_iterations: int = 8000,
) -> Tuple[int, float]:
    n = len(adj)
    optimum = float(np.mean(centers))
    matrix, _, splitters = augmented_matrix(metropolis_weights(adj))
    identity = np.eye(4 * n)
    half = 0.5 * (identity + matrix)
    previous = np.zeros(4 * n)
    previous_gradient = previous[0::4] - centers
    current = matrix @ previous - step_size * splitters @ previous_gradient
    for iteration in range(1, maximum_iterations + 1):
        error = float(np.max(np.abs(current[0::4] - optimum)))
        if error <= tolerance:
            return iteration, error
        current_gradient = current[0::4] - centers
        following = (
            (identity + matrix) @ current
            - half @ previous
            - step_size * splitters @ (current_gradient - previous_gradient)
        )
        previous, current = current, following
        previous_gradient = current_gradient
    return maximum_iterations, float(np.max(np.abs(current[0::4] - optimum)))


def network_structure_validation() -> List[dict]:
    rows: List[dict] = []
    probabilities = [0.0, 0.1, 0.3, 0.6, 1.0]
    for level, probability in enumerate(probabilities):
        for replicate in range(10):
            seed = 20263000 + 100 * level + replicate
            rng = np.random.default_rng(seed)
            adj = small_world_graph(20, 4, probability, rng)
            centers = rng.normal(0.0, 1.0, 20)
            matrix, _, _ = augmented_matrix(metropolis_weights(adj))
            iterations, final_error = quadratic_iterations(adj, centers)
            rows.append(
                {
                    "seed": seed,
                    "rewire_probability": probability,
                    "average_clustering": average_clustering(adj),
                    "base_slem": float(
                        sorted(np.abs(np.linalg.eigvals(metropolis_weights(adj))))[-2]
                    ),
                    "augmented_modal_radius": modal_pencil_radius(matrix),
                    "iterations_to_1e-6": iterations,
                    "final_error": final_error,
                }
            )
    return rows


def encoding_map(matrix: np.ndarray, left: np.ndarray) -> np.ndarray:
    n = len(matrix) // 4
    coefficients = np.array([0.0, 1.3, 0.8, 1.7], dtype=float)
    coefficients *= 4.0 / coefficients.sum()
    encoding = np.zeros((4 * n, n), dtype=float)
    for i in range(n):
        block = coefficients.copy()
        block /= n * np.dot(left[4 * i : 4 * i + 4], block)
        encoding[4 * i : 4 * i + 4, i] = block
    return encoding


def rowspace_distance(rows: np.ndarray, query: np.ndarray, rtol: float = 1e-9) -> float:
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    if len(singular) == 0 or singular[0] == 0.0:
        return float(np.linalg.norm(query))
    rank = int(np.sum(singular > rtol * singular[0]))
    basis = right[:rank].T
    residual = query - basis @ (basis.T @ query)
    return float(np.linalg.norm(residual))


def coalition_distance(
    matrix: np.ndarray, encoding: np.ndarray, target: int, coalition: SetKey
) -> float:
    dimension = len(matrix)
    identity = np.eye(dimension)
    lifted = np.block(
        [[identity + matrix, -0.5 * (identity + matrix)], [identity, np.zeros_like(identity)]]
    )
    visible = list(range(0, dimension, 4))
    for agent in coalition:
        visible.extend([4 * agent + 1, 4 * agent + 2, 4 * agent + 3])
    output_now = np.eye(dimension)[visible]
    output = np.hstack([output_now, np.zeros_like(output_now)])
    blocks = []
    power = np.eye(2 * dimension)
    for _ in range(2 * dimension):
        blocks.append(output @ power)
        power = power @ lifted
    observability = np.vstack(blocks)
    unit = np.eye(encoding.shape[1])[:, target]
    query_now = encoding @ np.linalg.solve(encoding.T @ encoding, unit)
    query = np.concatenate([query_now, np.zeros(dimension)])
    return rowspace_distance(observability, query)


def coalition_sweep() -> List[dict]:
    rows: List[dict] = []
    topologies = {
        "cycle": cycle_graph(6),
        "path": path_graph(6),
        "complete": complete_graph(6),
    }
    for topology, adj in topologies.items():
        matrix, left, _ = augmented_matrix(metropolis_weights(adj))
        encoding = encoding_map(matrix, left)
        for coalition_size in range(6):
            distances = []
            for target in range(6):
                available = [i for i in range(6) if i != target]
                for members in itertools.combinations(available, coalition_size):
                    distances.append(
                        coalition_distance(matrix, encoding, target, frozenset(members))
                    )
            rows.append(
                {
                    "topology": topology,
                    "coalition_size": coalition_size,
                    "cases": len(distances),
                    "minimum_distance": float(np.min(distances)),
                    "median_distance": float(np.median(distances)),
                    "maximum_distance": float(np.max(distances)),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    output: Path,
    sensitivity: Sequence[dict],
    stress: Sequence[dict],
    trajectories: np.ndarray,
    coalition: Sequence[dict],
) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), constrained_layout=True)

    eps_rows = [row for row in sensitivity if row["parameter"] == "eps_match"]
    axes[0, 0].semilogx(
        [row["value"] for row in eps_rows],
        [100 * row["success_rate"] for row in eps_rows],
        marker="o",
        color="#1f77b4",
    )
    axes[0, 0].axvline(NOMINAL_CONFIG.eps_match, color="black", ls="--", lw=0.8)
    axes[0, 0].set(xlabel=r"matching tolerance $\epsilon_m$", ylabel="strict recovery (%)")
    axes[0, 0].set_title("(a) Detector tolerance sensitivity")
    axes[0, 0].grid(alpha=0.25)

    for attack, marker in [("drift", "o"), ("oscillation", "s"), ("copy-dependent", "^")]:
        subset = [row for row in stress if row["attack"] == attack]
        axes[0, 1].plot(
            [row["level_index"] for row in subset],
            [100 * row["success_rate"] for row in subset],
            marker=marker,
            label=attack,
        )
    axes[0, 1].set(xlabel="severity index (0 is convergent)", ylabel="strict recovery (%)")
    axes[0, 1].set_title("(b) Out-of-model attacks")
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.25)

    median = np.median(trajectories, axis=0)
    lower = np.quantile(trajectories, 0.1, axis=0)
    upper = np.quantile(trajectories, 0.9, axis=0)
    x = np.arange(1, len(median) + 1)
    axes[1, 0].semilogy(x, median, color="#2ca02c", label="median")
    axes[1, 0].fill_between(x, lower, upper, color="#2ca02c", alpha=0.2, label="10--90%")
    axes[1, 0].set(xlabel="iteration", ylabel="maximum optimization error")
    axes[1, 0].set_title("(c) Smooth quartic objectives (30 trials)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.25)

    for topology, marker in [("cycle", "o"), ("path", "s"), ("complete", "^")]:
        subset = [row for row in coalition if row["topology"] == topology]
        axes[1, 1].plot(
            [row["coalition_size"] for row in subset],
            [row["minimum_distance"] for row in subset],
            marker=marker,
            label=topology,
        )
    axes[1, 1].set(xlabel="coalition size $|P|$", ylabel=r"minimum $d^{\rm ex}_{P,K}$")
    axes[1, 1].set_title("(d) Initial-state privacy vs. coalition size")
    axes[1, 1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1, 1].set_ylim(0.105, 0.120)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(alpha=0.25)

    fig.savefig(output / "reviewer_requested_validation.pdf", bbox_inches="tight")
    fig.savefig(output / "reviewer_requested_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def near_honest_separation_sweep(
    trials_per_level: int = 30, master_seed: int = 20269000
) -> List[dict]:
    """Single adversary broadcasting at a fixed separation from the honest limit.

    This reproduces, with the disclosed nominal constants, the near-honest
    stress behaviour reported during peer review: detection remains correct for
    separations well above eps_match and fails only once the separation falls
    within the clustering tolerance itself.
    """

    n, f = 10, 3
    separations = [
        0.5,
        0.1,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.0012,
        0.001,
        0.0005,
    ]
    rows: List[dict] = []
    for separation in separations:
        successes = 0
        for replicate in range(trials_per_level):
            seed = master_seed + replicate
            rng = np.random.default_rng(seed)
            adj = er_graph(n, 0.7, np.random.default_rng(seed), f + 1)
            y = rng.normal(0.0, 1.5, n)
            adversary = int(rng.integers(0, n))
            adversaries = frozenset({adversary})
            honest = sorted(set(range(n)) - adversaries)
            honest_limit = float(np.mean(y[honest]))
            attacks = {adversary: honest_limit + separation}
            correct = True
            for observer in honest:
                values = limiting_candidates(adj, y, adversaries, attacks, observer, f)
                noisy = {
                    key: float(value + rng.normal(0.0, 1.5e-4))
                    for key, value in values.items()
                }
                estimate = finite_time_detector(noisy, n, observer, f, NOMINAL_CONFIG)
                if estimate != adversaries:
                    correct = False
                    break
            successes += int(correct)
        rows.append(
            {
                "separation": separation,
                "successes": successes,
                "trials": trials_per_level,
                "success_rate": successes / trials_per_level,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    nominal = battery(NOMINAL_CONFIG)
    sensitivity = threshold_sensitivity()
    stress = out_of_model_stress()
    nonquadratic, trajectories = nonquadratic_validation()
    structure = network_structure_validation()
    coalition = coalition_sweep()
    separation = near_honest_separation_sweep()

    write_csv(args.output / "nominal_battery.csv", nominal)
    write_csv(args.output / "detector_sensitivity.csv", sensitivity)
    write_csv(args.output / "out_of_model_attacks.csv", stress)
    write_csv(args.output / "nonquadratic_validation.csv", nonquadratic)
    write_csv(args.output / "network_structure.csv", structure)
    write_csv(args.output / "coalition_sweep.csv", coalition)
    write_csv(args.output / "near_honest_separation.csv", separation)
    np.save(args.output / "nonquadratic_error_trajectories.npy", trajectories)
    make_figure(args.output, sensitivity, stress, trajectories, coalition)

    summary = {
        "detector": asdict(NOMINAL_CONFIG),
        "nominal_battery": {
            "successes": int(sum(row["success_all_honest"] for row in nominal)),
            "trials": len(nominal),
        },
        "nonquadratic": {
            "trials": len(nonquadratic),
            "median_final_error": float(np.median([r["final_max_abs_error"] for r in nonquadratic])),
            "maximum_final_error": float(np.max([r["final_max_abs_error"] for r in nonquadratic])),
            "maximum_disagreement": float(np.max([r["final_disagreement"] for r in nonquadratic])),
        },
        "coalition": {
            "maximum_tested_size": 5,
            "minimum_distance": float(np.min([r["minimum_distance"] for r in coalition])),
        },
        "near_honest_separation": {
            "smallest_perfect_separation": float(
                min(r["separation"] for r in separation if r["successes"] == r["trials"])
            ),
            "largest_failing_separation": float(
                max(
                    (r["separation"] for r in separation if r["successes"] == 0),
                    default=float("nan"),
                )
            ),
        },
        "network_structure": {
            "trials": len(structure),
            "clustering_iteration_correlation": float(
                np.corrcoef(
                    [r["average_clustering"] for r in structure],
                    [r["iterations_to_1e-6"] for r in structure],
                )[0, 1]
            ),
            "modal_radius_iteration_correlation": float(
                np.corrcoef(
                    [r["augmented_modal_radius"] for r in structure],
                    [r["iterations_to_1e-6"] for r in structure],
                )[0, 1]
            ),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
