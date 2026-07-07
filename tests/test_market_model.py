"""Smoke tests for the synthetic correlated-market generator."""

import torch

import training_data as td


def test_correlation_matrix_is_valid():
    N, K = 20, 3
    cluster_assignments = torch.randint(0, K, (N,))
    intra_corrs = torch.full((K,), 0.7)

    corr = td.build_correlation_matrix(N, K, cluster_assignments, intra_corrs, epsilon=0.3)

    assert corr.shape == (N, N)
    assert torch.allclose(torch.diag(corr), torch.ones(N))
    assert torch.allclose(corr, corr.T)
    assert corr.max() <= 1.0 + 1e-6


def test_generate_universe_returns_consistent_windows():
    torch.manual_seed(0)
    samples, label = td.generate_universe_data(N=20, K=3, T=30, w=10)

    # A tiny universe may fail to tip; either outcome must be internally consistent.
    if not samples:
        assert label is None
        return

    assert 0.0 < label <= 1.0
    for window in samples:
        assert window.shape == (20, 10)


def test_create_dataset_labels_align_with_samples():
    torch.manual_seed(1)
    samples, labels = td.create_training_dataset(num_universes=3, T=30, w=10)
    assert len(samples) == len(labels)
