# Reviewer-requested validation code

This repository contains supplementary code and reproducible experiments for the paper **“Accurate, Private, and Resilient Distributed Optimization via Network Augmentation.”**

The implementation specifies all finite-time detector constants and their exact application order. It includes sensitivity analyses, stress tests involving drift, oscillatory, and copy-dependent attacks outside the theoretical assumptions, an augmented-EXTRA experiment with smooth non-quadratic objectives, and an honest-but-curious coalition-size sweep.

## Reproduce

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python reviewer_validation.py --output results
```

The run is deterministic.  Seeds are included in the generated CSV files.  The
script writes:

- `nominal_battery.csv`: the 147-run strict randomized battery;
- `detector_sensitivity.csv`: one-at-a-time sweeps around all four detector
  constants;
- `out_of_model_attacks.csv`: empirical tests deliberately outside Assumption 2;
- `nonquadratic_validation.csv` and `nonquadratic_error_trajectories.npy`:
  augmented EXTRA on strongly convex smooth quartic objectives;
- `network_structure.csv`: clustering coefficient, base SLEM, augmented modal
  radius, and measured iterations on small-world networks;
- `coalition_sweep.csv`: canonical-query privacy distances for every target and
  every coalition through size `n-1` on three topologies;
- `near_honest_separation.csv`: detection of a single adversary broadcasting at
  a fixed separation from the honest limit, reproducing the near-honest stress
  boundary observed during peer review;
- `reviewer_requested_validation.pdf`/`.png`: the four-panel manuscript figure;
- `summary.json`: machine-readable headline results.

## Detector specification

The nominal finite-time constants are:

| Constant | Value | Role |
|---|---:|---|
| `eps_match` | `1e-3` | Maximum diameter of a one-dimensional complete-linkage limit cluster |
| `tau_cap` | `0.70` | Fractional presence threshold for the robust intersection `X(C)`; `Y(C)` remains the exact complement of the union |
| `tau_struct` | `0.60` | Minimum observed/expected witness count for a validated anchor |
| `tau_weak` | `0.15` | Minimum observed/expected count for a retained weak hint |

The code applies them in this order: complete-linkage clustering; robust
intersection and exact complement-of-union decoding; compatible-witness
structural acceptance; agreement-ranked proposal consolidation;
value-disambiguated weak hints for sparse singleton proposals; total-collusion
fallback; singleton incompatibility fallback; and finally the adversary-free
default. This is the same order stated in the revised paper.

## Scope

The formal theorem applies only when exclusion-copy trajectories have
copy-independent limits.  The drift, oscillation, and copy-dependent panels are
stress tests, not theorem-backed guarantees.  Privacy here protects the original
initial state under the knowledge assumptions stated in the paper; it is not a
differential-privacy claim for local costs or the full gradient trajectory.
