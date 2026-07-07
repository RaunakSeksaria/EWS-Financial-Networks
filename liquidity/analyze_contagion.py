"""
Detailed analysis of contagion mechanism effectiveness
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from liquid_model import (
    LiquidityODE,
    LiquidityParams,
    compute_active_adjacency,
    contagion_multiplier,
    giant_component_fraction,
)
from scipy.integrate import solve_ivp


def run_single_sim(contagion_strength=1.5, seed=42):
    """Run simulation and track contagion activation"""
    np.random.seed(seed)
    params = LiquidityParams()
    params.contagion_strength = contagion_strength

    # Network
    G = nx.erdos_renyi_graph(params.N, params.p_edge, directed=True, seed=seed)
    A = nx.to_numpy_array(G)

    # Initial conditions
    x0 = np.random.normal(0.8, 0.05, params.N)
    x0 = np.clip(x0, 0.5, 1.2)

    t_eval = np.linspace(0, params.T_max, params.sample_points)

    def kappa_func(t):
        return params.kappa0 + params.eta * t

    # Solve
    ode_system = LiquidityODE(A, kappa_func, params)
    sol = solve_ivp(
        ode_system,
        (0, params.T_max),
        x0,
        t_eval=t_eval,
        method="RK45",
        max_step=10.0,
        rtol=1e-4,
        atol=1e-6,
    )

    # Compute metrics
    results = {
        "t": sol.t,
        "x": sol.y.T,
        "kappa": [],
        "gc": [],
        "mean_x": [],
        "num_critical": [],  # nodes below x_critical
        "contagion_active": [],  # mean contagion multiplier
    }

    for ti, xi in zip(sol.t, sol.y.T):
        kappa = kappa_func(ti)
        A_active = compute_active_adjacency(xi, A, params)
        gc = giant_component_fraction(A_active)

        # Count critical nodes
        num_crit = np.sum(xi < params.x_critical)

        # Compute average contagion activation
        C = contagion_multiplier(xi, params.x_critical, params.mu)
        contagion_avg = C.mean()

        results["kappa"].append(kappa)
        results["gc"].append(gc)
        results["mean_x"].append(xi.mean())
        results["num_critical"].append(num_crit)
        results["contagion_active"].append(contagion_avg)

    for k in ["kappa", "gc", "mean_x", "num_critical", "contagion_active"]:
        results[k] = np.array(results[k])

    return results, params


def plot_contagion_analysis(results_list, labels, params, save_path="contagion_analysis.png"):
    """Plot detailed contagion mechanism analysis"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["blue", "red", "green", "orange"]

    # Plot 1: Giant Component
    ax = axes[0, 0]
    for res, label, color in zip(results_list, labels, colors):
        ax.plot(res["kappa"], res["gc"], "-", linewidth=2.5, label=label, color=color, alpha=0.7)
    ax.set_xlabel("Funding Cost κ", fontsize=11)
    ax.set_ylabel("Giant Component", fontsize=11)
    ax.set_title("Giant Component vs κ", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean liquidity
    ax = axes[0, 1]
    for res, label, color in zip(results_list, labels, colors):
        ax.plot(
            res["kappa"], res["mean_x"], "-", linewidth=2.5, label=label, color=color, alpha=0.7
        )
    ax.axhline(
        params.x_critical,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"x_crit = {params.x_critical}",
    )
    ax.axhline(params.x_R, color="gray", linestyle=":", linewidth=1.5, label=f"x_R = {params.x_R}")
    ax.set_xlabel("Funding Cost κ", fontsize=11)
    ax.set_ylabel("Mean Liquidity <x>", fontsize=11)
    ax.set_title("Average Node State", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Number of critical nodes
    ax = axes[0, 2]
    for res, label, color in zip(results_list, labels, colors):
        ax.plot(
            res["kappa"],
            res["num_critical"],
            "-",
            linewidth=2.5,
            label=label,
            color=color,
            alpha=0.7,
        )
    ax.set_xlabel("Funding Cost κ", fontsize=11)
    ax.set_ylabel("# Critical Nodes (x < x_crit)", fontsize=11)
    ax.set_title("Contagion Trigger Count", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Contagion activation
    ax = axes[1, 0]
    for res, label, color in zip(results_list, labels, colors):
        ax.plot(
            res["kappa"],
            res["contagion_active"],
            "-",
            linewidth=2.5,
            label=label,
            color=color,
            alpha=0.7,
        )
    ax.set_xlabel("Funding Cost κ", fontsize=11)
    ax.set_ylabel("Mean Contagion Multiplier", fontsize=11)
    ax.set_title("Contagion Mechanism Activation", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 5: Transition sharpness (derivative of GC)
    ax = axes[1, 1]
    for res, label, color in zip(results_list, labels, colors):
        dgc = np.gradient(res["gc"], res["kappa"])
        ax.plot(res["kappa"], -dgc, "-", linewidth=2.5, label=label, color=color, alpha=0.7)
    ax.set_xlabel("Funding Cost κ", fontsize=11)
    ax.set_ylabel("-dGC/dκ", fontsize=11)
    ax.set_title("Transition Sharpness (Higher = More Abrupt)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 6: State distribution at transition
    ax = axes[1, 2]
    for res, label, color in zip(results_list, labels, colors):
        # Find transition point
        idx = np.where(res["gc"] < 0.5)[0]
        if len(idx) > 0:
            trans_idx = max(0, idx[0] - 2)  # Just before transition
            x_at_trans = res["x"][trans_idx]
            ax.hist(x_at_trans, bins=20, alpha=0.5, label=label, color=color)
    ax.axvline(params.x_critical, color="black", linestyle=":", linewidth=2)
    ax.axvline(params.x_R, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Node Liquidity x", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Node State Distribution at Transition", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Analysis plot saved to {save_path}")
    plt.show()


def print_metrics(results_list, labels):
    """Print quantitative comparison"""
    print("\n" + "=" * 70)
    print("CONTAGION MECHANISM DETAILED ANALYSIS")
    print("=" * 70)

    for res, label in zip(results_list, labels):
        print(f"\n{label}:")

        # Find transition
        idx = np.where(res["gc"] < 0.5)[0]
        if len(idx) > 0:
            kc = res["kappa"][idx[0]]
            print(f"  κc = {kc:.4f}")

            # Transition sharpness (max derivative)
            dgc = np.gradient(res["gc"], res["kappa"])
            max_deriv = np.abs(dgc).max()
            print(f"  Max |dGC/dκ| = {max_deriv:.2f} (sharpness)")

            # Nodes in critical state at transition
            trans_idx = max(0, idx[0] - 1)
            n_crit = res["num_critical"][trans_idx]
            print(f"  Critical nodes at transition = {n_crit:.0f} / 100")

            # Contagion activation at transition
            cong_act = res["contagion_active"][trans_idx]
            print(f"  Contagion activation = {cong_act:.3f}")
        else:
            print("  No transition detected")

    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("CONTAGION MECHANISM ANALYSIS")
    print("=" * 70)

    # Run with different contagion strengths
    print("\nRunning simulations with varying contagion strength...")

    strengths = [0.0, 0.5, 1.5, 2.5]
    labels = [f"Contagion={s}" for s in strengths]
    results_list = []

    for strength in strengths:
        print(f"  Running with contagion_strength = {strength}...")
        res, params = run_single_sim(contagion_strength=strength, seed=42)
        results_list.append(res)

    print_metrics(results_list, labels)
    plot_contagion_analysis(results_list, labels, params)

    print("\n✓ Analysis complete!")
