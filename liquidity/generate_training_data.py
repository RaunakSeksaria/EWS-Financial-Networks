"""
Training Data Generation Pipeline for Liquidity Fragmentation Prediction

Generates datasets by:
1. Running many simulations with varied network topologies and parameters
2. Extracting overlapping time windows at different lead distances before κ_c
3. Storing windows + adjacencies + labels for supervised learning

Output:
- windows/ directory with NPZ files (time-series arrays, adjacencies, masks)
- metadata.csv with simulation parameters and labels
- diagnostics/ with QA plots and statistics
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import from existing model
from pathlib import Path
import importlib.util

# Use absolute path based on script location for robustness
script_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "liquid_model", 
    script_dir / "liquid_model.py"
)
liquid_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(liquid_model)

LiquidityParams = liquid_model.LiquidityParams
LiquidityODE = liquid_model.LiquidityODE
compute_active_adjacency = liquid_model.compute_active_adjacency
giant_component_fraction = liquid_model.giant_component_fraction
find_critical_kappa = liquid_model.find_critical_kappa

from scipy.integrate import solve_ivp


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class DataGenConfig:
    """Configuration for training data generation"""
    
    # Output paths
    output_dir: str = "training_data"
    windows_dir: str = "windows"
    metadata_file: str = "metadata.csv"
    diagnostics_dir: str = "diagnostics"
    
    # Dataset size
    n_simulations: int = 100  # Start small for testing
    
    # Window extraction
    window_length: int = 20  # time steps per window
    window_stride: int = 5   # stride for overlapping windows
    lead_fractions: List[float] = None  # fractions of kappa_c to extract windows
    
    # Topology variations
    network_sizes: List[int] = None
    topology_types: List[str] = None
    
    # Parameter ranges (will be sampled)
    p_edge_range: Tuple[float, float] = (0.03, 0.08)  # ER connection prob
    sf_gamma_range: Tuple[float, float] = (2.0, 3.0)  # Scale-free exponent
    
    # Dynamics parameters (varied slightly around defaults)
    epsilon_range: Tuple[float, float] = (0.2, 0.4)  # coupling strength
    beta_range: Tuple[float, float] = (0.4, 0.6)     # linear drain
    c_range: Tuple[float, float] = (0.08, 0.12)      # cubic saturation
    
    # Control ramp variations
    eta_range: Tuple[float, float] = (1.5e-4, 2.5e-4)  # ramp rate
    kappa0_range: Tuple[float, float] = (0.25, 0.35)   # initial funding cost
    
    # Simulation settings
    T_max: float = 1800
    sample_points: int = 3000
    noise_sd: float = 2e-3
    
    # Quality control
    min_gc_drop: float = 0.3  # minimum GC drop to accept simulation
    
    def __post_init__(self):
        if self.lead_fractions is None:
            self.lead_fractions = [0.70, 0.80, 0.90, 0.95, 0.98]
        if self.network_sizes is None:
            self.network_sizes = [50, 100]
        if self.topology_types is None:
            self.topology_types = ["ER_sparse", "ER_dense", "scale_free"]


# ============================================================================
# NETWORK GENERATION
# ============================================================================
def generate_network(topology_type: str, N: int, seed: int, **kwargs) -> np.ndarray:
    """
    Generate network adjacency matrix based on topology type
    
    Args:
        topology_type: "ER_sparse", "ER_dense", "scale_free", etc.
        N: number of nodes
        seed: random seed
        **kwargs: additional topology-specific parameters
    
    Returns:
        Adjacency matrix (N, N)
    """
    np.random.seed(seed)
    
    if topology_type == "ER_sparse":
        p = kwargs.get('p_edge', 0.05)
        G = nx.erdos_renyi_graph(N, p, directed=True, seed=seed)
    
    elif topology_type == "ER_dense":
        p = kwargs.get('p_edge', 0.08)
        G = nx.erdos_renyi_graph(N, p, directed=True, seed=seed)
    
    elif topology_type == "scale_free":
        # Use Barabasi-Albert and add direction
        m = max(1, int(N * 0.03))  # edges to attach per new node
        G = nx.barabasi_albert_graph(N, m, seed=seed)
        # Convert to directed
        G = G.to_directed()
    
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    A = nx.to_numpy_array(G)
    return A


# ============================================================================
# SIMULATION WITH FULL TRAJECTORY
# ============================================================================
def run_single_simulation(
    sim_id: int,
    params: LiquidityParams,
    A: np.ndarray,
    seed: int,
    verbose: bool = False
) -> Dict:
    """
    Run a single simulation and return full trajectory + diagnostics
    
    Returns:
        Dictionary with:
            - t_vals: time array
            - kappa_vals: control parameter values
            - gc_vals: giant component values
            - x_history: node states (time, nodes)
            - kappa_c_thresh: critical kappa (threshold method)
            - kappa_c_deriv: critical kappa (derivative method)
            - success: whether simulation succeeded
            - gc_drop: magnitude of GC drop
    """
    np.random.seed(seed)
    N = A.shape[0]
    
    # Initial conditions
    x0 = np.random.normal(0.8, 0.05, N)
    x0 = np.clip(x0, 0.5, 1.2)
    
    # Time points
    t_eval = np.linspace(0, params.T_max, params.sample_points)
    
    # Kappa function
    def kappa_func(time):
        return params.kappa0 + params.eta * time
    
    # Create ODE system
    ode_system = LiquidityODE(A, kappa_func, params)
    
    if verbose:
        print(f"  [Sim {sim_id}] Integrating ODEs...")
    
    # Solve
    sol = solve_ivp(
        ode_system,
        (0, params.T_max),
        x0,
        t_eval=t_eval,
        method='RK45',
        max_step=10.0,
        rtol=1e-4,
        atol=1e-6
    )
    
    if not sol.success:
        return {
            'success': False,
            'message': sol.message
        }
    
    # Extract results
    t_vals = sol.t
    x_history = sol.y.T  # (time_points, N)
    
    # Compute GC over time
    if verbose:
        print(f"  [Sim {sim_id}] Computing giant component...")
    
    kappa_vals = []
    gc_vals = []
    
    for ti, xi in zip(t_vals, x_history):
        kappa = kappa_func(ti)
        A_active = compute_active_adjacency(xi, A, params)
        gc = giant_component_fraction(A_active)
        
        kappa_vals.append(kappa)
        gc_vals.append(gc)
    
    kappa_vals = np.array(kappa_vals)
    gc_vals = np.array(gc_vals)
    
    # Find critical points
    kappa_c_thresh = find_critical_kappa(kappa_vals, gc_vals, method='threshold', threshold=0.5)
    kappa_c_deriv = find_critical_kappa(kappa_vals, gc_vals, method='derivative')
    
    # Compute GC drop magnitude
    gc_initial = gc_vals[0]
    gc_final = gc_vals[-1]
    gc_drop = gc_initial - gc_final
    
    return {
        'success': True,
        't_vals': t_vals,
        'kappa_vals': kappa_vals,
        'gc_vals': gc_vals,
        'x_history': x_history,
        'kappa_c_thresh': kappa_c_thresh,
        'kappa_c_deriv': kappa_c_deriv,
        'gc_initial': gc_initial,
        'gc_final': gc_final,
        'gc_drop': gc_drop
    }


# ============================================================================
# WINDOW EXTRACTION
# ============================================================================
def extract_windows(
    t_vals: np.ndarray,
    kappa_vals: np.ndarray,
    x_history: np.ndarray,
    kappa_c: float,
    window_length: int,
    window_stride: int,
    lead_fractions: List[float]
) -> List[Dict]:
    """
    Extract overlapping windows at specified lead distances before kappa_c
    
    Args:
        t_vals: time array
        kappa_vals: control parameter values
        x_history: node trajectories (time, nodes)
        kappa_c: critical kappa value
        window_length: number of time steps per window
        window_stride: stride for overlapping windows
        lead_fractions: fractions of kappa_c to center windows around
    
    Returns:
        List of window dictionaries with:
            - window_data: (window_length, nodes) array
            - window_start_idx: start index in full trajectory
            - window_end_idx: end index
            - window_end_kappa: kappa value at window end
            - lead_fraction: which lead fraction this window corresponds to
    """
    windows = []
    
    # Find index closest to kappa_c
    if kappa_c is None:
        return windows
    
    kappa_c_idx = np.argmin(np.abs(kappa_vals - kappa_c))
    
    for lead_frac in lead_fractions:
        # Target kappa value for this lead distance
        target_kappa = lead_frac * kappa_c
        
        # Find closest index
        target_idx = np.argmin(np.abs(kappa_vals - target_kappa))
        
        # Extract windows around this point with stride
        # Start from earlier time and stride towards target
        start_search = max(0, target_idx - window_length * 3)
        end_search = min(len(t_vals), target_idx + window_length)
        
        for window_start in range(start_search, end_search - window_length, window_stride):
            window_end = window_start + window_length
            
            if window_end >= len(t_vals):
                break
            
            window_data = x_history[window_start:window_end, :]
            
            # CRITICAL: Ensure window ends before kappa_c (pre-transition data only)
            window_end_kappa = kappa_vals[window_end - 1]
            if window_end_kappa >= kappa_c:
                continue  # Skip windows that extend past critical point
            
            windows.append({
                'window_data': window_data,
                'window_start_idx': window_start,
                'window_end_idx': window_end,
                'window_end_kappa': window_end_kappa,
                'lead_fraction': lead_frac,
                'target_kappa': target_kappa
            })
    
    return windows


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================
def generate_training_dataset(config: DataGenConfig, verbose: bool = True):
    """
    Main pipeline: generate simulations, extract windows, save dataset
    """
    
    # Create output directories
    output_path = Path(config.output_dir)
    windows_path = output_path / config.windows_dir
    diagnostics_path = output_path / config.diagnostics_dir
    
    output_path.mkdir(exist_ok=True)
    windows_path.mkdir(exist_ok=True)
    diagnostics_path.mkdir(exist_ok=True)
    
    # Save config
    with open(output_path / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print("=" * 70)
    print("TRAINING DATA GENERATION PIPELINE")
    print("=" * 70)
    print(f"Target simulations: {config.n_simulations}")
    print(f"Window length: {config.window_length} steps")
    print(f"Lead fractions: {config.lead_fractions}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    # Storage for metadata
    metadata_records = []
    
    # Counters
    sim_count = 0
    accepted_count = 0
    rejected_count = 0
    total_windows = 0
    
    # Statistics
    gc_drops = []
    kappa_c_values = []
    
    # Progress bar
    pbar = tqdm(total=config.n_simulations, desc="Generating simulations")
    
    while accepted_count < config.n_simulations:
        sim_count += 1
        
        # Randomly select parameters
        N = np.random.choice(config.network_sizes)
        topology_type = np.random.choice(config.topology_types)
        
        # Create base parameters
        params = LiquidityParams()
        params.N = N
        params.T_max = config.T_max
        params.sample_points = config.sample_points
        params.noise_sd = config.noise_sd
        
        # Randomize dynamics parameters
        params.epsilon = np.random.uniform(*config.epsilon_range)
        params.beta = np.random.uniform(*config.beta_range)
        params.c = np.random.uniform(*config.c_range)
        params.eta = np.random.uniform(*config.eta_range)
        params.kappa0 = np.random.uniform(*config.kappa0_range)
        
        # Generate network
        if topology_type.startswith("ER"):
            p_edge = np.random.uniform(*config.p_edge_range)
            A = generate_network(topology_type, N, seed=sim_count, p_edge=p_edge)
            topology_param = p_edge
        elif topology_type == "scale_free":
            A = generate_network(topology_type, N, seed=sim_count)
            topology_param = None
        else:
            A = generate_network(topology_type, N, seed=sim_count)
            topology_param = None
        
        # Run simulation
        result = run_single_simulation(
            sim_id=sim_count,
            params=params,
            A=A,
            seed=sim_count * 42,
            verbose=False
        )
        
        if not result['success']:
            rejected_count += 1
            pbar.set_postfix({'accepted': accepted_count, 'rejected': rejected_count})
            continue
        
        # Check quality: GC drop must exceed threshold
        if result['gc_drop'] < config.min_gc_drop:
            rejected_count += 1
            pbar.set_postfix({'accepted': accepted_count, 'rejected': rejected_count})
            continue
        
        # Choose label method randomly (epsilon column)
        use_threshold = np.random.choice([True, False])
        kappa_c_label = result['kappa_c_thresh'] if use_threshold else result['kappa_c_deriv']
        
        if kappa_c_label is None:
            rejected_count += 1
            pbar.set_postfix({'accepted': accepted_count, 'rejected': rejected_count})
            continue
        
        # Simulation accepted!
        accepted_count += 1
        
        # Extract windows
        windows = extract_windows(
            t_vals=result['t_vals'],
            kappa_vals=result['kappa_vals'],
            x_history=result['x_history'],
            kappa_c=kappa_c_label,
            window_length=config.window_length,
            window_stride=config.window_stride,
            lead_fractions=config.lead_fractions
        )
        
        # Save windows and adjacency
        sim_id_str = f"sim_{accepted_count:05d}"
        
        for win_idx, window in enumerate(windows):
            window_id = f"{sim_id_str}_win_{win_idx:03d}"
            
            # Save window data
            np.savez_compressed(
                windows_path / f"{window_id}.npz",
                window_data=window['window_data'],
                adjacency=A,
                mask=np.ones(N, dtype=bool)  # all observed for now
            )
            
            # Record metadata
            metadata_records.append({
                'window_id': window_id,
                'sim_id': sim_id_str,
                'sim_count': accepted_count,
                'topology_type': topology_type,
                'topology_param': topology_param,
                'N': N,
                'epsilon': params.epsilon,
                'beta': params.beta,
                'c': params.c,
                'eta': params.eta,
                'kappa0': params.kappa0,
                'kappa_c_thresh': result['kappa_c_thresh'],
                'kappa_c_deriv': result['kappa_c_deriv'],
                'kappa_c_label': kappa_c_label,
                'label_method': 'threshold' if use_threshold else 'derivative',
                'window_end_kappa': window['window_end_kappa'],
                'lead_fraction': window['lead_fraction'],
                'gc_initial': result['gc_initial'],
                'gc_final': result['gc_final'],
                'gc_drop': result['gc_drop'],
                'seed': sim_count * 42
            })
        
        total_windows += len(windows)
        
        # Update statistics
        gc_drops.append(result['gc_drop'])
        kappa_c_values.append(kappa_c_label)
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix({
            'accepted': accepted_count,
            'rejected': rejected_count,
            'windows': total_windows
        })
    
    pbar.close()
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(output_path / config.metadata_file, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total simulations attempted: {sim_count}")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected (weak transition): {rejected_count}")
    print(f"Total windows extracted: {total_windows}")
    print(f"Windows per simulation: {total_windows / accepted_count:.1f}")
    print()
    print(f"GC drop statistics:")
    print(f"  Mean: {np.mean(gc_drops):.3f}")
    print(f"  Std: {np.std(gc_drops):.3f}")
    print(f"  Min: {np.min(gc_drops):.3f}")
    print(f"  Max: {np.max(gc_drops):.3f}")
    print()
    print(f"κ_c statistics:")
    print(f"  Mean: {np.mean(kappa_c_values):.4f}")
    print(f"  Std: {np.std(kappa_c_values):.4f}")
    print(f"  Min: {np.min(kappa_c_values):.4f}")
    print(f"  Max: {np.max(kappa_c_values):.4f}")
    print()
    print(f"Metadata saved to: {output_path / config.metadata_file}")
    print(f"Windows saved to: {windows_path}")
    print("=" * 70)
    
    # Create diagnostic plots
    create_diagnostic_plots(metadata_df, gc_drops, kappa_c_values, diagnostics_path)
    
    return metadata_df


# ============================================================================
# DIAGNOSTIC VISUALIZATION
# ============================================================================
def create_diagnostic_plots(metadata_df: pd.DataFrame, gc_drops: List, kappa_c_values: List, save_dir: Path):
    """Create QA/diagnostic plots for the dataset"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Distribution of κ_c labels
    axes[0, 0].hist(kappa_c_values, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('κ_c', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Distribution of Critical Points', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. GC drop distribution
    axes[0, 1].hist(gc_drops, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('GC Drop', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Distribution of GC Drops', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Topology distribution
    topology_counts = metadata_df.groupby('sim_id')['topology_type'].first().value_counts()
    axes[0, 2].bar(range(len(topology_counts)), topology_counts.values, color='green', alpha=0.7)
    axes[0, 2].set_xticks(range(len(topology_counts)))
    axes[0, 2].set_xticklabels(topology_counts.index, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Count', fontsize=11)
    axes[0, 2].set_title('Network Topology Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Network size distribution
    size_counts = metadata_df.groupby('sim_id')['N'].first().value_counts().sort_index()
    axes[1, 0].bar(size_counts.index, size_counts.values, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Network Size (N)', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Network Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Windows per lead fraction
    lead_counts = metadata_df['lead_fraction'].value_counts().sort_index()
    axes[1, 1].bar(lead_counts.index, lead_counts.values, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Lead Fraction', fontsize=11)
    axes[1, 1].set_ylabel('Window Count', fontsize=11)
    axes[1, 1].set_title('Windows by Lead Distance', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Label method distribution
    method_counts = metadata_df.groupby('sim_id')['label_method'].first().value_counts()
    axes[1, 2].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Label Method Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dataset_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Diagnostic plots saved to {save_dir / 'dataset_diagnostics.png'}")
    plt.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Create configuration
    config = DataGenConfig(
        n_simulations=100,  # Start with 100 for testing
        window_length=20,
        window_stride=5,
        network_sizes=[50, 100],
        topology_types=["ER_sparse", "ER_dense", "scale_free"]
    )
    
    # Generate dataset
    metadata = generate_training_dataset(config, verbose=True)
    
    print("\n✓ Training data generation complete!")
    print(f"✓ Load windows with: np.load('training_data/windows/<window_id>.npz')")
    print(f"✓ Load metadata with: pd.read_csv('training_data/metadata.csv')")
