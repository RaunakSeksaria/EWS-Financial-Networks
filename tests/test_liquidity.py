"""Smoke tests for the liquidity-fragmentation model and dataset utilities."""

import numpy as np
import pandas as pd
from liquid_model import (
    LiquidityParams,
    find_critical_kappa,
    giant_component_fraction,
    run_simulation,
)
from load_dataset import create_train_val_test_split


def test_run_simulation_short_produces_transition():
    params = LiquidityParams()
    # Shrink the problem so the ODE integrates quickly.
    params.N = 40
    params.T_max = 200
    params.sample_points = 200

    kappa_vals, gc_vals, x_history = run_simulation(params, seed=0)

    assert len(kappa_vals) == len(gc_vals)
    assert x_history.shape[1] == params.N
    # Funding cost ramps monotonically upward.
    assert kappa_vals[-1] > kappa_vals[0]
    # Giant component fraction is a valid order parameter in [0, 1].
    assert gc_vals.min() >= 0.0 and gc_vals.max() <= 1.0


def test_giant_component_fraction_bounds():
    A_full = np.ones((10, 10)) - np.eye(10)
    A_empty = np.zeros((10, 10))
    assert giant_component_fraction(A_full) == 1.0
    assert 0.0 <= giant_component_fraction(A_empty) <= 0.2


def test_find_critical_kappa_detects_drop():
    kappa = np.linspace(0.3, 0.7, 50)
    gc = np.ones(50)
    gc[25:] = 0.1  # sharp fragmentation halfway through
    kappa_c = find_critical_kappa(kappa, gc, method="threshold")
    assert kappa_c is not None
    assert kappa[24] <= kappa_c <= kappa[26]


def test_split_has_no_simulation_leakage():
    # Ten simulations, five windows each.
    metadata = pd.DataFrame(
        {"sim_id": np.repeat(np.arange(10), 5), "kappa_c_label": np.random.rand(50)}
    )
    train, val, test = create_train_val_test_split(metadata, seed=0)

    train_sims = set(train["sim_id"])
    val_sims = set(val["sim_id"])
    test_sims = set(test["sim_id"])

    assert train_sims.isdisjoint(val_sims)
    assert train_sims.isdisjoint(test_sims)
    assert val_sims.isdisjoint(test_sims)
    assert len(train) + len(val) + len(test) == len(metadata)
