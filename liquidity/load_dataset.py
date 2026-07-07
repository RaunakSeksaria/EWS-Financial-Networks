"""
Utility functions for loading and inspecting the training dataset

Example usage:
    from load_dataset import load_dataset, get_window
    
    # Load full dataset
    metadata, windows_path = load_dataset('training_data')
    
    # Load a specific window
    window_data, adjacency, mask, label = get_window(metadata, windows_path, 0)
    
    # Filter by network size
    small_nets = metadata[metadata['N'] == 50]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_dataset(data_dir: str = 'training_data') -> Tuple[pd.DataFrame, Path]:
    """
    Load the training dataset metadata
    
    Args:
        data_dir: path to training data directory
    
    Returns:
        metadata: DataFrame with all window metadata
        windows_path: Path object to windows directory
    """
    data_path = Path(data_dir)
    metadata = pd.read_csv(data_path / 'metadata.csv')
    windows_path = data_path / 'windows'
    
    print(f"Loaded dataset with {len(metadata)} windows from {len(metadata['sim_id'].unique())} simulations")
    return metadata, windows_path


def get_window(
    metadata: pd.DataFrame,
    windows_path: Path,
    idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load a specific window by index
    
    Args:
        metadata: metadata DataFrame
        windows_path: path to windows directory
        idx: row index in metadata
    
    Returns:
        window_data: (time, nodes) time-series data
        adjacency: (nodes, nodes) adjacency matrix
        mask: (nodes,) observation mask
        label: scalar kappa_c label
    """
    row = metadata.iloc[idx]
    window_id = row['window_id']
    
    # Load NPZ file
    data = np.load(windows_path / f"{window_id}.npz")
    
    window_data = data['window_data']
    adjacency = data['adjacency']
    mask = data['mask']
    label = row['kappa_c_label']
    
    return window_data, adjacency, mask, label


def get_simulation_windows(
    metadata: pd.DataFrame,
    windows_path: Path,
    sim_id: str
) -> Tuple[list, list, np.ndarray, list]:
    """
    Get all windows from a specific simulation
    
    Args:
        metadata: metadata DataFrame
        windows_path: path to windows directory
        sim_id: simulation ID (e.g., 'sim_00001')
    
    Returns:
        window_data_list: list of window arrays
        labels: list of labels (all same for one simulation)
        adjacency: the shared adjacency matrix
        lead_fractions: list of lead fractions for each window
    """
    sim_windows = metadata[metadata['sim_id'] == sim_id]
    
    window_data_list = []
    labels = []
    lead_fractions = []
    adjacency = None
    
    for idx, row in sim_windows.iterrows():
        window_id = row['window_id']
        data = np.load(windows_path / f"{window_id}.npz")
        
        window_data_list.append(data['window_data'])
        labels.append(row['kappa_c_label'])
        lead_fractions.append(row['lead_fraction'])
        
        if adjacency is None:
            adjacency = data['adjacency']
    
    return window_data_list, labels, adjacency, lead_fractions


def dataset_summary(metadata: pd.DataFrame):
    """Print summary statistics about the dataset"""
    
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal windows: {len(metadata)}")
    print(f"Unique simulations: {len(metadata['sim_id'].unique())}")
    print(f"Windows per simulation: {len(metadata) / len(metadata['sim_id'].unique()):.1f}")
    
    print(f"\n--- Network Properties ---")
    print(f"Network sizes: {sorted(metadata['N'].unique())}")
    print(f"Topology types: {metadata['topology_type'].unique().tolist()}")
    
    print(f"\n--- Labels ---")
    print(f"κ_c range: [{metadata['kappa_c_label'].min():.4f}, {metadata['kappa_c_label'].max():.4f}]")
    print(f"κ_c mean ± std: {metadata['kappa_c_label'].mean():.4f} ± {metadata['kappa_c_label'].std():.4f}")
    
    print(f"\n--- Label Methods ---")
    print(metadata.groupby('sim_id')['label_method'].first().value_counts())
    
    print(f"\n--- Lead Fractions ---")
    print(metadata['lead_fraction'].value_counts().sort_index())
    
    print(f"\n--- GC Drop Statistics ---")
    sim_gc = metadata.groupby('sim_id')['gc_drop'].first()
    print(f"Mean: {sim_gc.mean():.3f}")
    print(f"Std: {sim_gc.std():.3f}")
    print(f"Min: {sim_gc.min():.3f}")
    print(f"Max: {sim_gc.max():.3f}")
    
    print("=" * 70)


def create_train_val_test_split(
    metadata: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at simulation level (no leakage between splits)
    
    Args:
        metadata: full metadata DataFrame
        train_frac: fraction for training
        val_frac: fraction for validation
        test_frac: fraction for testing
        seed: random seed
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    # Get unique simulations
    sim_ids = metadata['sim_id'].unique()
    n_sims = len(sim_ids)
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(sim_ids)
    
    # Split
    n_train = int(n_sims * train_frac)
    n_val = int(n_sims * val_frac)
    
    train_sims = sim_ids[:n_train]
    val_sims = sim_ids[n_train:n_train + n_val]
    test_sims = sim_ids[n_train + n_val:]
    
    # Create split DataFrames
    train_df = metadata[metadata['sim_id'].isin(train_sims)].copy()
    val_df = metadata[metadata['sim_id'].isin(val_sims)].copy()
    test_df = metadata[metadata['sim_id'].isin(test_sims)].copy()
    
    print(f"Split dataset:")
    print(f"  Train: {len(train_df)} windows from {len(train_sims)} simulations")
    print(f"  Val:   {len(val_df)} windows from {len(val_sims)} simulations")
    print(f"  Test:  {len(test_df)} windows from {len(test_sims)} simulations")
    
    return train_df, val_df, test_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Load dataset
    metadata, windows_path = load_dataset('training_data')
    
    # Print summary
    dataset_summary(metadata)
    
    print("\n--- Example: Loading a single window ---")
    window_data, adjacency, mask, label = get_window(metadata, windows_path, 0)
    print(f"Window shape: {window_data.shape}")
    print(f"Adjacency shape: {adjacency.shape}")
    print(f"Label (κ_c): {label:.4f}")
    
    print("\n--- Example: Creating train/val/test split ---")
    train_df, val_df, test_df = create_train_val_test_split(metadata)
    
    print("\n--- Example: Get all windows from one simulation ---")
    first_sim_id = metadata['sim_id'].iloc[0]
    windows, labels, adj, lead_fracs = get_simulation_windows(metadata, windows_path, first_sim_id)
    print(f"Simulation {first_sim_id}: {len(windows)} windows")
    print(f"Lead fractions: {sorted(set(lead_fracs))}")
