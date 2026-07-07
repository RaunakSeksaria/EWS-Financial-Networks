"""Synthetic correlated-market universe generator.

Simulates stylized equity markets of N stocks grouped into K sectors. The
inter-sector correlation epsilon acts as the control parameter: as it ramps
up, the correlation matrix eventually loses positive-definiteness, which is
taken as the critical transition. Pre-transition return windows are extracted
as training samples, all labelled with the universe's critical epsilon.
"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal

torch.set_default_dtype(torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_correlation_matrix(N, K, cluster_assignments, intra_corrs, epsilon):
    """Build the (N, N) correlation matrix for one epsilon step.

    Args:
        N: Total number of stocks.
        K: Total number of sectors.
        cluster_assignments: (N,) tensor mapping each stock to a sector index.
        intra_corrs: (K,) tensor of intra-sector correlations.
        epsilon: Inter-sector correlation.

    Returns:
        (N, N) correlation matrix with unit diagonal.
    """
    # Pairs in the same sector get the smaller of the two sector correlations;
    # pairs in different sectors get epsilon.
    cluster_matrix = cluster_assignments.unsqueeze(0) == cluster_assignments.unsqueeze(1)
    intra_corr_map = intra_corrs[cluster_assignments]
    intra_matrix = torch.min(intra_corr_map.unsqueeze(0), intra_corr_map.unsqueeze(1))
    epsilon_matrix = torch.full((N, N), epsilon, device=device)

    corr_matrix = torch.where(cluster_matrix, intra_matrix, epsilon_matrix)
    corr_matrix.fill_diagonal_(1.0)
    return corr_matrix


def generate_universe_data(N, K, T, w, epsilon_start=0.1, epsilon_end=0.9, epsilon_step=0.01):
    """Simulate one universe and return its samples and tipping-point label.

    Epsilon is ramped from epsilon_start to epsilon_end. At each step, T
    timesteps of returns are drawn from a multivariate normal with the
    corresponding correlation matrix. The first epsilon at which the matrix
    stops being positive-definite (Cholesky failure) is the critical
    transition epsilon_c; all data generated before it is windowed into
    (N, w) samples that share the label epsilon_c.

    Args:
        N: Number of stocks in this universe.
        K: Number of sectors.
        T: Timesteps generated per epsilon step.
        w: Sliding-window length.
        epsilon_start: Initial inter-sector correlation.
        epsilon_end: Maximum inter-sector correlation.
        epsilon_step: Ramp increment.

    Returns:
        (samples, epsilon_c): a list of (N, w) tensors and the float label,
        or ([], None) if the universe never tipped or tipped too early to
        yield at least one full window.
    """
    cluster_assignments = torch.randint(0, K, (N,), device=device)
    # Intra-sector correlations are drawn uniformly from [0.5, 1.0].
    intra_corrs = (torch.rand(K, device=device) * 0.5) + 0.5

    all_time_series_data = []
    epsilon_steps = torch.arange(epsilon_start, epsilon_end + epsilon_step, epsilon_step, device=device)
    epsilon_c = None

    for epsilon_val in epsilon_steps:
        epsilon_val = epsilon_val.item()
        corr_matrix = build_correlation_matrix(N, K, cluster_assignments, intra_corrs, epsilon_val)

        # Positive-definiteness check: a failed Cholesky factorization marks
        # the critical transition and ends the simulation.
        try:
            torch.linalg.cholesky(corr_matrix)
        except RuntimeError:
            epsilon_c = epsilon_val
            break

        try:
            dist = MultivariateNormal(
                loc=torch.zeros(N, device=device), covariance_matrix=corr_matrix
            )
        except ValueError:
            # Numerical instability short of outright Cholesky failure is
            # treated as the transition as well.
            epsilon_c = epsilon_val
            break

        # (T, N) sample transposed to (N, T).
        returns = dist.sample((T,)).T
        all_time_series_data.append(returns)

    if epsilon_c is None or len(all_time_series_data) == 0:
        return [], None

    full_pre_series = torch.cat(all_time_series_data, dim=1)
    total_len = full_pre_series.shape[1]
    if total_len < w:
        return [], None

    samples = [full_pre_series[:, t : t + w] for t in range(total_len - w + 1)]
    return samples, epsilon_c


def create_training_dataset(num_universes, T=100, w=20):
    """Build the full training pool from many randomized universes.

    Args:
        num_universes: Number of universes to simulate.
        T: Timesteps generated per epsilon step.
        w: Sliding-window length.

    Returns:
        (samples, labels): a pooled list of (N, w) tensors and the parallel
        list of epsilon_c labels.
    """
    all_samples_in_pool = []
    all_labels_in_pool = []

    print(f"Generating {num_universes} universes...")

    for i in range(num_universes):
        N_rand = np.random.randint(50, 201)
        K_rand = np.random.randint(3, 11)

        samples, label = generate_universe_data(N=N_rand, K=K_rand, T=T, w=w)
        if samples:
            all_samples_in_pool.extend(samples)
            all_labels_in_pool.extend([label] * len(samples))

        if (i + 1) % 100 == 0:
            print(
                f"  {i + 1}/{num_universes} universes done, "
                f"{len(all_samples_in_pool)} samples so far"
            )

    print(f"Done: {len(all_samples_in_pool)} samples from {num_universes} universes.")
    return all_samples_in_pool, all_labels_in_pool


if __name__ == "__main__":
    # Small smoke run; increase num_universes to 1000+ for a real dataset.
    samples, labels = create_training_dataset(num_universes=10, T=100, w=20)

    if samples:
        print(f"First sample shape: {samples[0].shape}, label: {labels[0]}")
        print(f"Last sample shape:  {samples[-1].shape}, label: {labels[-1]}")
