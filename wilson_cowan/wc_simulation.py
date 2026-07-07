import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from tqdm import tqdm

# Import constants from config
from config import (
    TAU, MU, WEIGHT_MIN, WEIGHT_MAX, MAX_ITER, DT, ABS_TOL,
    EPSILON_MIN, EPSILON_MAX, EPSILON_STEP
)

# --- Wilson-Cowan Dynamics ---

def sigmoid(x, tau=TAU, mu=MU):
    """Logistic sigmoid: 1/(1 + e^(-τ(x - μ)))"""
    return 1.0 / (1.0 + np.exp(-tau * (x - mu)))

def rhs(x: np.ndarray, epsilon: float, W_sparse: sp.csr_matrix) -> np.ndarray:
    """Wilson-Cowan ODE matching paper: dx/dt = -x + (1-ε) * W @ sigmoid(x)"""
    weighted_input = (1.0 - epsilon) * (W_sparse @ sigmoid(x))
    return -x + weighted_input

def relax_to_steady_state(epsilon: float, x0: np.ndarray, W_sparse: sp.csr_matrix) -> np.ndarray:
    """Relax to steady state using explicit Euler integration."""
    x = x0.copy()
    for _ in range(MAX_ITER):
        dx = rhs(x, epsilon, W_sparse)
        x_new = x + DT * dx
        np.clip(x_new, 0.0, None, out=x_new)
        if np.linalg.norm(x_new - x) < ABS_TOL:
            return x_new
        x = x_new
    return x # Return last state if no convergence

def run_single_universe(sim_rng: np.random.Generator) -> dict:
    """
    Generates one random network ("universe") and runs a full epsilon sweep.
    Returns a dictionary with all data needed for training.
    """
    
    # --- 1. Generate Network (as per paper Sec III C) ---
    N = sim_rng.integers(300, 701)
    mean_degree = sim_rng.uniform(3.0, 6.0)
    p_er = mean_degree / N
    
    G_nx = nx.erdos_renyi_graph(n=N, p=p_er, seed=sim_rng, directed=True)
    
    W = np.zeros((N, N))
    for (u, v) in G_nx.edges():
        W[v, u] = sim_rng.uniform(WEIGHT_MIN, WEIGHT_MAX) 
    W_sparse = sp.csr_matrix(W, dtype=float)
    
    edge_index = torch.tensor(list(G_nx.edges), dtype=torch.long).t().contiguous()

    # --- 2. Setup Simulation ---
    epsilon_values = np.arange(EPSILON_MIN, EPSILON_MAX + EPSILON_STEP, EPSILON_STEP)
    num_eps = len(epsilon_values)
    
    full_states = [] 
    mean_activity_trace = [] 
    
    # --- 3. Find Initial Bistable States ---
    x_init_low = sim_rng.uniform(low=0.0, high=0.01, size=N)
    x_init_high = sim_rng.uniform(low=50.0, high=80.0, size=N)
    
    x_low_0 = relax_to_steady_state(0.0, x_init_low, W_sparse)
    x_high_0 = relax_to_steady_state(0.0, x_init_high, W_sparse)
    
    # Store the initial mean activity
    mean_high_0 = np.mean(x_high_0)

    if np.mean(x_low_0) > (mean_high_0 - 1.0):
        return None 

    x_current = x_high_0.copy()
    transitioned = False
    transition_idx = -1
    
    # --- 4. Run Epsilon Sweep ---
    for i, eps in enumerate(epsilon_values):
        if not transitioned:
            x_star = relax_to_steady_state(eps, x_current, W_sparse)
            mean_act = np.mean(x_star)
            
            activity_jump = mean_act - mean_activity_trace[-1] if i > 0 else 0.0
            jump_threshold = -10.0
            
            if (i > 0 and activity_jump < jump_threshold):
                transitioned = True
                transition_idx = i-1
                x_star_low = relax_to_steady_state(eps, x_low_0, W_sparse) 
                x_star = x_star_low.copy()
                mean_act = np.mean(x_star)
                x_current = x_star_low.copy()
            else:
                x_current = x_star.copy()
        
        else:
            x_star = relax_to_steady_state(eps, x_current, W_sparse)
            mean_act = np.mean(x_star)
            x_current = x_star.copy()
            
        full_states.append(x_star)
        mean_activity_trace.append(mean_act)
            
    # --- 5. Find Critical Epsilon (eps_c) ---
    eps_c = np.nan
    idx_cross = None
    
    if transitioned:
        idx_cross = transition_idx
    else:
        min_jump = 0.0
        jump_threshold = -5.0
        
        for i in range(1, len(mean_activity_trace)):
            jump = mean_activity_trace[i] - mean_activity_trace[i-1]
            if jump < min_jump and jump < jump_threshold:
                min_jump = jump
                idx_cross = i
                break
    
    if idx_cross is None:
        return None
        
    eps_c = epsilon_values[idx_cross]
    
    return {
        'N': N,
        'edge_index': edge_index,
        'eps_values': epsilon_values,
        'full_states': np.array(full_states),
        'eps_c': eps_c,
        'transition_idx': idx_cross,
        'mean_high_0': mean_high_0  # <-- ADD THIS LINE
    }

def generate_universes(num_universes, seed):
    """
    Main function to generate G universes.
    """
    print(f"Generating {num_universes} simulation universes...")
    universe_rng = np.random.default_rng(seed)
    universes = []
    for _ in tqdm(range(num_universes)):
        universe_data = run_single_universe(universe_rng)
        if universe_data is not None:
            universes.append(universe_data)

    print(f"Successfully generated {len(universes)} universes with valid transitions.")
    return universes