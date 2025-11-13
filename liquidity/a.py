import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
class LiquidityParams:
    """Parameters for the liquidity fragmentation ODE model"""
    def __init__(self):
        # Network
        self.N = 100                    # number of nodes
        self.p_edge = 0.05             # ER connection probability
        
        # Local dynamics
        self.alpha0 = 1.0              # baseline inflow
        self.beta = 0.5                # linear drain
        self.c = 0.1                   # cubic saturation
        
        # Coupling
        self.epsilon = 0.9             # coupling strength (increased for stronger feedback)
        
        # Sigmoid thresholds & sharpness
        self.x_L = 0.6                 # lender threshold
        self.x_R = 0.5                 # risky borrower threshold
        self.lam = 10.0                # lender sigmoid sharpness (sharper transitions)
        self.gamma = 10.0              # risk sigmoid sharpness (sharper transitions)
        
        # Control parameter ramp
        self.kappa0 = 0.0              # initial funding cost
        self.eta = 2e-4                # ramp rate (faster for shorter sim)
        
        # Simulation
        self.T_max = 3000              # total time (reduced)
        self.sample_points = 150       # number of points to sample
        self.noise_sd = 5e-4           # dynamical noise (reduced)
        
        # Active link threshold
        self.tau_w = 0.15              # threshold for active edges

# ============================================================================
# COUPLING KERNELS
# ============================================================================
def sigmoid(z):
    """Numerically stable sigmoid"""
    z = np.clip(z, -20, 20)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

def H_lend(x, x_L, lam):
    """Lender propensity: high when x > x_L"""
    return sigmoid(lam * (x - x_L))

def R_risk(x, x_R, gamma):
    """Borrower risk: high when x < x_R"""
    return sigmoid(gamma * (x_R - x))

# ============================================================================
# ODE DYNAMICS (VECTORIZED)
# ============================================================================
class LiquidityODE:
    def __init__(self, A, kappa_func, params):
        self.A = A
        self.A_sparse = (A > 0).astype(float)  # Binary adjacency
        self.kappa_func = kappa_func
        self.params = params
        
    def __call__(self, t, x):
        """
        dx_i/dt = alpha(kappa) - beta*x_i - c*x_i^3 
                  - epsilon * sum_j A_ij * H_lend(x_i) * R_risk(x_j)
        """
        # Clip state to prevent runaway
        x = np.clip(x, -2, 3)
        
        kappa = self.kappa_func(t)
        alpha = self.params.alpha0 - kappa
        
        # Local dynamics
        dx = alpha - self.params.beta * x - self.params.c * x**3
        
        # Coupling term (vectorized)
        H = H_lend(x, self.params.x_L, self.params.lam)  # (N,)
        R = R_risk(x, self.params.x_R, self.params.gamma)  # (N,)
        
        # Matrix multiplication: A[i,j] * H[i] * R[j]
        # For each i: sum_j A[i,j] * R[j], then multiply by H[i]
        coupling = H * (self.A_sparse @ R)
        
        dx -= self.params.epsilon * coupling
        
        # Add small noise
        dx += np.random.normal(0, self.params.noise_sd, len(x))
        
        return dx

# ============================================================================
# ACTIVE NETWORK & GIANT COMPONENT
# ============================================================================
def compute_active_adjacency(x, A, params):
    """
    Compute active adjacency based on effective weights:
    w_ij = H_lend(x_i) * (1 - R_risk(x_j))
    Edge is active if w_ij > tau_w
    """
    x = np.clip(x, -2, 3)
    H = H_lend(x, params.x_L, params.lam)
    R = R_risk(x, params.x_R, params.gamma)
    
    # Vectorized: w[i,j] = H[i] * (1 - R[j]) where A[i,j] = 1
    w = np.outer(H, (1 - R))
    A_active = ((A > 0) & (w > params.tau_w)).astype(float)
    
    return A_active

def giant_component_fraction(A):
    """Return fraction of nodes in giant component"""
    if A.sum() == 0:
        return 0.0
    
    G = nx.from_numpy_array(A, create_using=nx.Graph())
    if G.number_of_nodes() == 0:
        return 0.0
    
    components = list(nx.connected_components(G))
    if len(components) == 0:
        return 0.0
    
    largest = max(components, key=len)
    return len(largest) / G.number_of_nodes()

# ============================================================================
# SIMULATION
# ============================================================================
def run_simulation(params: LiquidityParams, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full simulation with kappa ramp
    
    Returns:
        kappa_vals: array of kappa values
        gc_vals: giant component fractions
        x_history: state history (sample_points, N)
    """
    np.random.seed(seed)
    
    # Generate network
    G = nx.erdos_renyi_graph(params.N, params.p_edge, directed=True, seed=seed)
    A = nx.to_numpy_array(G)
    
    # Initial conditions (healthy state)
    x0 = np.random.normal(0.8, 0.05, params.N)
    x0 = np.clip(x0, 0.5, 1.2)
    
    # Time points to evaluate
    t_eval = np.linspace(0, params.T_max, params.sample_points)
    
    # Kappa function
    def kappa_func(time):
        return params.kappa0 + params.eta * time
    
    # Create ODE system
    ode_system = LiquidityODE(A, kappa_func, params)
    
    print("Integrating ODEs...")
    # Solve with adaptive method
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
        print(f"Warning: Integration failed: {sol.message}")
    
    # Extract results
    t = sol.t
    x_history = sol.y.T  # (time_points, N)
    
    # Compute metrics
    print("Computing giant component...")
    kappa_vals = []
    gc_vals = []
    
    for i, (ti, xi) in enumerate(zip(t, x_history)):
        kappa = kappa_func(ti)
        A_active = compute_active_adjacency(xi, A, params)
        gc = giant_component_fraction(A_active)
        
        kappa_vals.append(kappa)
        gc_vals.append(gc)
        
        if i % 10 == 0:
            print(f"  t = {ti:.1f}, κ = {kappa:.4f}, GC = {gc:.3f}, mean(x) = {xi.mean():.3f}")
    
    return np.array(kappa_vals), np.array(gc_vals), x_history

# ============================================================================
# CRITICAL POINT DETECTION
# ============================================================================
def find_critical_kappa(kappa_vals, gc_vals, method='threshold', threshold=0.5):
    """
    Find kappa_c where transition occurs
    
    method: 'threshold' - drop below threshold*initial
            'derivative' - maximum derivative
    """
    if len(gc_vals) < 3:
        return None
    
    if method == 'threshold':
        gc_initial = gc_vals[0]
        target = threshold * gc_initial
        idx = np.where(gc_vals < target)[0]
        if len(idx) == 0:
            return None
        return kappa_vals[idx[0]]
    
    elif method == 'derivative':
        # Find maximum negative derivative
        dgc = np.diff(gc_vals)
        idx = np.argmin(dgc)
        return kappa_vals[idx]
    
    return None

# ============================================================================
# PLOTTING
# ============================================================================
def plot_transition(kappa_vals, gc_vals, kappa_c=None, save_path='transition.png'):
    """
    Plot giant component vs kappa (the epsilon-graph equivalent)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Main transition plot
    ax1.plot(kappa_vals, gc_vals, 'b-', linewidth=2.5, label='Giant Component', alpha=0.8)
    
    if kappa_c is not None:
        ax1.axvline(kappa_c, color='r', linestyle='--', linewidth=2, 
                   label=f'κc = {kappa_c:.4f}', alpha=0.7)
        ax1.plot(kappa_c, gc_vals[np.argmin(np.abs(kappa_vals - kappa_c))], 
                'ro', markersize=10, label='Transition Point')
    
    ax1.set_xlabel('Funding Cost κ', fontsize=13)
    ax1.set_ylabel('Giant Component Fraction', fontsize=13)
    ax1.set_title('Liquidity Fragmentation Phase Transition', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 1.05])
    
    # Derivative to show transition sharpness
    if len(kappa_vals) > 1:
        dgc_dkappa = np.gradient(gc_vals, kappa_vals)
        ax2.plot(kappa_vals, -dgc_dkappa, 'g-', linewidth=2, label='-dGC/dκ')
        
        if kappa_c is not None:
            ax2.axvline(kappa_c, color='r', linestyle='--', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Funding Cost κ', fontsize=13)
        ax2.set_ylabel('Transition Rate', fontsize=13)
        ax2.set_title('Order Parameter Derivative', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LIQUIDITY FRAGMENTATION NETWORK ODE SIMULATOR")
    print("=" * 70)
    
    # Initialize parameters
    params = LiquidityParams()
    
    print("\nParameters:")
    print(f"  Network: N = {params.N}, p = {params.p_edge}")
    print(f"  Local dynamics: α₀ = {params.alpha0}, β = {params.beta}, c = {params.c}")
    print(f"  Coupling: ε = {params.epsilon}")
    print(f"  Thresholds: x_L = {params.x_L}, x_R = {params.x_R}")
    print(f"  Sigmoids: λ = {params.lam}, γ = {params.gamma}")
    print(f"  Control: κ₀ = {params.kappa0}, η = {params.eta}")
    print(f"  Time: T_max = {params.T_max}, points = {params.sample_points}")
    print()
    
    # Run simulation
    print("Starting simulation...\n")
    kappa_vals, gc_vals, x_history = run_simulation(params, seed=42)
    
    # Find critical point (try both methods)
    kappa_c_thresh = find_critical_kappa(kappa_vals, gc_vals, method='threshold', threshold=0.5)
    kappa_c_deriv = find_critical_kappa(kappa_vals, gc_vals, method='derivative')
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    if kappa_c_thresh is not None:
        print(f"  Critical point (50% drop): κc ≈ {kappa_c_thresh:.4f}")
    if kappa_c_deriv is not None:
        print(f"  Critical point (max derivative): κc ≈ {kappa_c_deriv:.4f}")
    print(f"  Initial GC: {gc_vals[0]:.3f}")
    print(f"  Final GC: {gc_vals[-1]:.3f}")
    print(f"  Final mean(x): {x_history[-1].mean():.3f}")
    print("=" * 70)
    
    # Use threshold method for visualization
    kappa_c = kappa_c_thresh if kappa_c_thresh is not None else kappa_c_deriv
    
    # Plot
    plot_transition(kappa_vals, gc_vals, kappa_c)
    
    print("\n✓ Simulation complete!")