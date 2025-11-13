# Training Data Generation for Liquidity Fragmentation Prediction

This directory contains tools for generating training datasets to predict network fragmentation from time-series observations.

## Overview

The pipeline generates supervised learning datasets where:
- **Input**: Short time-series windows of node states (20 timesteps × N nodes) + adjacency matrix
- **Output**: Scalar label κ_c (control parameter value at fragmentation)
- **Goal**: Learn to predict when a network will fragment from short observations

## Files

### Core Scripts

1. **`liquid-model.py`** - Base ODE simulator for liquidity fragmentation dynamics
2. **`generate_training_data.py`** - Main data generation pipeline
3. **`load_dataset.py`** - Utilities for loading and inspecting generated data

### Generated Data Structure

```
training_data/
├── config.json              # Generation configuration
├── metadata.csv             # All window metadata and labels
├── windows/                 # Individual window NPZ files
│   ├── sim_00001_win_000.npz
│   ├── sim_00001_win_001.npz
│   └── ...
└── diagnostics/
    └── dataset_diagnostics.png  # QA plots
```

## Quick Start

### 1. Generate Dataset

```python
from generate_training_data import DataGenConfig, generate_training_dataset

# Configure generation
config = DataGenConfig(
    n_simulations=100,          # Number of accepted simulations
    window_length=20,           # Timesteps per window
    window_stride=5,            # Overlapping stride
    network_sizes=[50, 100],    # Node counts to vary
    topology_types=['ER_sparse', 'ER_dense', 'scale_free']
)

# Run generation (takes ~5 minutes for 100 sims)
metadata = generate_training_dataset(config)
```

**Output**: 
- ~5000 windows from 100 simulations
- ~10 simulations rejected for weak transitions (GC drop < 0.3)

### 2. Load and Inspect Data

```python
from load_dataset import load_dataset, get_window, dataset_summary

# Load metadata
metadata, windows_path = load_dataset('training_data')

# Print summary statistics
dataset_summary(metadata)

# Load a specific window
window_data, adjacency, mask, label = get_window(metadata, windows_path, 0)
# window_data: (20, N) array
# adjacency: (N, N) array  
# mask: (N,) boolean array (all True for now)
# label: scalar κ_c value
```

### 3. Create Train/Val/Test Split

```python
from load_dataset import create_train_val_test_split

# Split at simulation level (no leakage)
train_df, val_df, test_df = create_train_val_test_split(
    metadata, 
    train_frac=0.7, 
    val_frac=0.15, 
    test_frac=0.15
)

# Each DataFrame contains windows only from simulations in that split
```

## Dataset Details

### What Each Training Example Contains

**Inputs:**
- `window_data`: (20, N) time-series of node liquidity states
- `adjacency`: (N, N) directed adjacency matrix (same for all windows from one simulation)
- `mask`: (N,) observation mask (currently all ones, for future partial observability)

**Label:**
- `kappa_c_label`: Scalar control parameter value at network fragmentation

**Metadata** (for analysis, not training):
- Network parameters: topology type, size, connection probability
- Dynamics parameters: ε, β, c, η, κ₀  
- Diagnostics: GC drop magnitude, initial/final GC values
- Window info: lead fraction, end κ value, which simulation

### Parameter Variations

The pipeline sweeps across:

1. **Network Topology**:
   - Erdős-Rényi sparse (p ~ 0.03-0.05)
   - Erdős-Rényi dense (p ~ 0.06-0.08)
   - Scale-free (Barabási-Albert)

2. **Network Size**: N ∈ {50, 100}

3. **Dynamics Parameters** (randomized around defaults):
   - Coupling strength ε ∈ [0.2, 0.4]
   - Linear drain β ∈ [0.4, 0.6]
   - Cubic saturation c ∈ [0.08, 0.12]
   - Control ramp rate η ∈ [1.5e-4, 2.5e-4]
   - Initial funding cost κ₀ ∈ [0.25, 0.35]

4. **Label Method** (random choice per simulation):
   - Threshold method: κ where GC drops below 50% of initial
   - Derivative method: κ at maximum negative dGC/dκ

### Window Extraction Strategy

From each simulation trajectory:
- Extract windows at **5 lead distances**: [0.70, 0.80, 0.90, 0.95, 0.98] × κ_c
- Use **overlapping windows** with stride=5 timesteps
- Each simulation produces ~50-60 windows at varying distances from transition

This creates diversity in "how far ahead" the model needs to predict.

## Quality Control

Simulations are **rejected** if:
- Integration fails
- GC drop < 0.3 (too weak/unclear transition)
- κ_c cannot be detected

**Acceptance rate**: ~90% (10 rejected per 100 accepted in typical runs)

## Statistics from Sample Run (100 simulations)

```
Total windows: 5,088
Simulations: 100 accepted, 10 rejected
Windows per simulation: ~51

Network sizes: 50 (47%), 100 (53%)
Topologies: Scale-free (36%), ER_sparse (33%), ER_dense (31%)

κ_c labels:
  Mean: 0.502 ± 0.072
  Range: [0.333, 0.666]

GC drops:
  Mean: 0.972 ± 0.058
  Range: [0.720, 1.000]
```

## Converting to PyTorch Tensors

```python
import torch
import numpy as np

# Load window
window_data, adjacency, mask, label = get_window(metadata, windows_path, 0)

# Convert to tensors
X = torch.FloatTensor(window_data)  # (20, N)
A = torch.FloatTensor(adjacency)    # (N, N)
y = torch.FloatTensor([label])      # (1,)

# Optional: normalize node states
X_norm = (X - X.mean()) / (X.std() + 1e-8)
```

## Extending the Dataset

### Add More Topologies

Edit `DataGenConfig.topology_types` and implement new cases in `generate_network()`:

```python
# In generate_training_data.py
config = DataGenConfig(
    topology_types=['ER_sparse', 'ER_dense', 'scale_free', 'small_world', 'core_periphery']
)
```

### Add Partial Observability

Modify window extraction to create random observation masks:

```python
# In extract_windows() or when saving
mask = np.random.rand(N) > 0.3  # 30% nodes hidden
np.savez_compressed(path, window_data=window_data, adjacency=A, mask=mask)
```

### Add Heterogeneous Node Parameters

Vary local parameters (α₀, β, c) per node instead of globally.

## Notes

- **Window overlap**: Helps create more training data, but note windows from same simulation are correlated
- **Lead fractions**: Earlier windows (0.7×κ_c) are harder to predict than later ones (0.98×κ_c)
- **Label method randomization**: Adds robustness - model learns both threshold and derivative-based transitions
- **Adjacency reuse**: Same network used for all windows in a simulation (stored redundantly for convenience)

## Performance

- **Generation time**: ~3s per simulation (100 sims in ~5 minutes)
- **Storage**: ~1.4 MB metadata CSV + ~200 MB windows for 100 simulations
- Each window NPZ: ~40-160 KB depending on network size

## Future Enhancements

1. **Parallelization**: Use multiprocessing to generate simulations in parallel
2. **Noise variations**: Add observational noise and dynamical shocks
3. **Heterogeneity**: Per-node parameter variations
4. **Alternative networks**: Core-periphery, small-world, community structure
5. **Temporal resolution**: Vary sampling cadence across simulations
6. **Label noise**: Intentionally perturb some labels to test robustness

## Citation

Based on liquidity fragmentation model from `liquid-model.py`. See model documentation for ODE dynamics and phase transition details.
