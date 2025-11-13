"""
Validation script to prove the liquidity model has non-trivial network feedback
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from a import LiquidityParams, LiquidityODE, compute_active_adjacency, giant_component_fraction

def run_comparison(params, seed=42):
    """
    Compare three scenarios:
    1. Full network coupling (ε > 0)
    2. No coupling (ε = 0) - just local dynamics
    3. Random network (different topology)
    
    If network feedback matters, trajectories should differ significantly.
    """
    np.random.seed(seed)
    
    # Generate networks
    G_original = nx.erdos_renyi_graph(params.N, params.p_edge, directed=True, seed=seed)
    A_original = nx.to_numpy_array(G_original)
    
    G_random = nx.erdos_renyi_graph(params.N, params.p_edge, directed=True, seed=seed+1)
    A_random = nx.to_numpy_array(G_random)
    
    # Same initial conditions for all
    x0 = np.random.normal(0.8, 0.05, params.N)
    x0 = np.clip(x0, 0.5, 1.2)
    
    t_eval = np.linspace(0, params.T_max, params.sample_points)
    
    def kappa_func(t):
        return params.kappa0 + params.eta * t
    
    results = {}
    
    # Scenario 1: Full coupling (original network)
    print("Running: Full coupling with original network...")
    ode1 = LiquidityODE(A_original, kappa_func, params)
    sol1 = solve_ivp(ode1, (0, params.T_max), x0, t_eval=t_eval, 
                     method='RK45', max_step=10.0, rtol=1e-4, atol=1e-6)
    results['full_original'] = {
        't': sol1.t,
        'x': sol1.y.T,
        'A': A_original
    }
    
    # Scenario 2: No coupling (ε = 0)
    print("Running: No coupling (isolated nodes)...")
    params_nocoupling = LiquidityParams()
    params_nocoupling.epsilon = 0.0
    ode2 = LiquidityODE(A_original, kappa_func, params_nocoupling)
    sol2 = solve_ivp(ode2, (0, params.T_max), x0, t_eval=t_eval,
                     method='RK45', max_step=10.0, rtol=1e-4, atol=1e-6)
    results['no_coupling'] = {
        't': sol2.t,
        'x': sol2.y.T,
        'A': A_original
    }
    
    # Scenario 4: Full coupling but NO contagion (contagion_strength = 0)
    print("Running: Coupling without contagion...")
    params_nocontagion = LiquidityParams()
    params_nocontagion.contagion_strength = 0.0
    ode4 = LiquidityODE(A_original, kappa_func, params_nocontagion)
    sol4 = solve_ivp(ode4, (0, params.T_max), x0, t_eval=t_eval,
                     method='RK45', max_step=10.0, rtol=1e-4, atol=1e-6)
    results['no_contagion'] = {
        't': sol4.t,
        'x': sol4.y.T,
        'A': A_original
    }
    
    # Scenario 3: Different network topology
    print("Running: Full coupling with different network...")
    ode3 = LiquidityODE(A_random, kappa_func, params)
    sol3 = solve_ivp(ode3, (0, params.T_max), x0, t_eval=t_eval,
                     method='RK45', max_step=10.0, rtol=1e-4, atol=1e-6)
    results['full_random'] = {
        't': sol3.t,
        'x': sol3.y.T,
        'A': A_random
    }
    
    return results, kappa_func, params

def analyze_differences(results, kappa_func, params):
    """Compute metrics to show network effects"""
    
    metrics = {}
    
    for name, data in results.items():
        t = data['t']
        x_history = data['x']
        A = data['A']
        
        kappa_vals = [kappa_func(ti) for ti in t]
        gc_vals = []
        mean_x = []
        std_x = []
        
        for xi in x_history:
            A_active = compute_active_adjacency(xi, A, params)
            gc = giant_component_fraction(A_active)
            gc_vals.append(gc)
            mean_x.append(xi.mean())
            std_x.append(xi.std())
        
        # Find when GC drops below 0.5
        gc_initial = gc_vals[0]
        idx = np.where(np.array(gc_vals) < 0.5 * gc_initial)[0]
        kappa_c = kappa_vals[idx[0]] if len(idx) > 0 else None
        
        metrics[name] = {
            'kappa': np.array(kappa_vals),
            'gc': np.array(gc_vals),
            'mean_x': np.array(mean_x),
            'std_x': np.array(std_x),
            'kappa_c': kappa_c
        }
    
    return metrics

def plot_validation(metrics, save_path='validation.png'):
    """Plot comparison to show network feedback matters"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'full_original': 'blue',
        'no_coupling': 'red',
        'full_random': 'green',
        'no_contagion': 'orange'
    }
    
    labels = {
        'full_original': 'Full Model (Coupling + Contagion)',
        'no_coupling': 'No Coupling (ε=0)',
        'full_random': 'Full Model (Random Net)',
        'no_contagion': 'Coupling Only (No Contagion)'
    }
    
    # Plot 1: Giant Component
    ax = axes[0, 0]
    for name, m in metrics.items():
        ax.plot(m['kappa'], m['gc'], '-', linewidth=2.5, 
                color=colors[name], label=labels[name], alpha=0.7)
        if m['kappa_c'] is not None:
            ax.axvline(m['kappa_c'], color=colors[name], linestyle='--', alpha=0.5)
    ax.set_xlabel('Funding Cost κ', fontsize=11)
    ax.set_ylabel('Giant Component Fraction', fontsize=11)
    ax.set_title('Giant Component vs κ', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean liquidity
    ax = axes[0, 1]
    for name, m in metrics.items():
        ax.plot(m['kappa'], m['mean_x'], '-', linewidth=2.5,
                color=colors[name], label=labels[name], alpha=0.7)
    ax.set_xlabel('Funding Cost κ', fontsize=11)
    ax.set_ylabel('Mean Liquidity <x>', fontsize=11)
    ax.set_title('Average Node State', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Std deviation (heterogeneity)
    ax = axes[1, 0]
    for name, m in metrics.items():
        ax.plot(m['kappa'], m['std_x'], '-', linewidth=2.5,
                color=colors[name], label=labels[name], alpha=0.7)
    ax.set_xlabel('Funding Cost κ', fontsize=11)
    ax.set_ylabel('Std Dev of Liquidity', fontsize=11)
    ax.set_title('Node Heterogeneity (Network Feedback)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Critical points comparison
    ax = axes[1, 1]
    kappa_cs = [m['kappa_c'] for m in metrics.values() if m['kappa_c'] is not None]
    names = [labels[name] for name, m in metrics.items() if m['kappa_c'] is not None]
    colors_bar = [colors[name] for name, m in metrics.items() if m['kappa_c'] is not None]
    
    if len(kappa_cs) > 0:
        bars = ax.bar(range(len(kappa_cs)), kappa_cs, color=colors_bar, alpha=0.7)
        ax.set_xticks(range(len(kappa_cs)))
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Critical κc', fontsize=11)
        ax.set_title('Transition Point Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, kappa_cs)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Validation plot saved to {save_path}")
    plt.show()

def print_analysis(metrics):
    """Print quantitative analysis"""
    print("\n" + "="*70)
    print("NETWORK FEEDBACK VALIDATION (WITH CONTAGION)")
    print("="*70)
    
    full_orig = metrics['full_original']
    no_coup = metrics['no_coupling']
    full_rand = metrics['full_random']
    no_cont = metrics.get('no_contagion')
    
    print("\n1. CRITICAL POINTS:")
    print(f"   Full model (coupling + contagion):  κc = {full_orig['kappa_c']:.4f}")
    print(f"   No coupling (ε=0):                  κc = {no_coup['kappa_c']:.4f}" if no_coup['kappa_c'] else "   No coupling: No transition detected")
    if no_cont and no_cont['kappa_c']:
        print(f"   Coupling only (no contagion):       κc = {no_cont['kappa_c']:.4f}")
        contagion_effect = abs(full_orig['kappa_c'] - no_cont['kappa_c'])
        print(f"   → Contagion shifts κc by: {contagion_effect:.4f}")
    print(f"   Full model (random network):        κc = {full_rand['kappa_c']:.4f}")
    
    kc_diff = abs(full_orig['kappa_c'] - full_rand['kappa_c']) if full_rand['kappa_c'] else 0
    print(f"   → Network topology changes κc by: {kc_diff:.4f}")
    
    print("\n2. FINAL HETEROGENEITY (std of x at end):")
    print(f"   Full model (original):     σ = {full_orig['std_x'][-1]:.4f}")
    print(f"   No coupling (ε=0):         σ = {no_coup['std_x'][-1]:.4f}")
    if no_cont:
        print(f"   No contagion:              σ = {no_cont['std_x'][-1]:.4f}")
    print(f"   Full model (random):       σ = {full_rand['std_x'][-1]:.4f}")
    
    print("\n3. INTERPRETATION:")
    if kc_diff > 0.02:
        print("   ✓ Network topology MATTERS (different κc)")
        print("   ✓ Model captures network-dependent fragmentation")
    else:
        print("   ⚠ Network topology has weak effect")
    
    if full_orig['std_x'][-1] > no_coup['std_x'][-1] * 1.5:
        print("   ✓ Coupling creates HETEROGENEITY")
        print("   ✓ Not all nodes decline uniformly")
    else:
        print("   ⚠ Coupling may be too weak")
    
    if no_cont and no_cont['kappa_c']:
        contagion_effect = abs(full_orig['kappa_c'] - no_cont['kappa_c'])
        if contagion_effect > 0.02:
            print("   ✓ CONTAGION mechanism has significant effect")
            print("   ✓ Cascading failures amplify the transition")
        else:
            print("   ⚠ Contagion effect is weak")
    
    print("\n4. VERDICT:")
    if kc_diff > 0.02 and full_orig['std_x'][-1] > no_coup['std_x'][-1] * 1.2:
        print("   ✅ MODEL IS NON-TRIVIAL - network feedback is real")
        if no_cont and no_cont['kappa_c'] and abs(full_orig['kappa_c'] - no_cont['kappa_c']) > 0.02:
            print("   ✅ CONTAGION adds meaningful cascading dynamics")
    else:
        print("   ⚠️  MODEL MAY BE TOO WEAK - consider increasing parameters")
    
    print("="*70)

if __name__ == "__main__":
    print("="*70)
    print("LIQUIDITY MODEL VALIDATION (WITH CONTAGION)")
    print("Testing if network feedback is non-trivial...")
    print("="*70)
    
    params = LiquidityParams()
    print(f"\nUsing: ε = {params.epsilon}, λ = {params.lam}, γ = {params.gamma}")
    print(f"Contagion: strength = {params.contagion_strength}, x_crit = {params.x_critical}")
    print("Running 4 scenarios (this may take a moment)...\n")
    
    results, kappa_func, params = run_comparison(params, seed=42)
    metrics = analyze_differences(results, kappa_func, params)
    
    print_analysis(metrics)
    plot_validation(metrics)
    
    print("\n✓ Validation complete!")
