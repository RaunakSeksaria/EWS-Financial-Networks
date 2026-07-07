import json

import networkx as nx
import numpy as np
import pandas as pd


class FinancialNetworkSimulator:
    """
    Simulates dynamics on a financial network of banks and firms.
    """

    def __init__(
        self,
        n_firms: int = 100,
        n_banks: int = 10,
        topology: str = "barabasi_albert",
        seed: int = 42,
    ):
        """
        Initialize the financial network simulator.

        Parameters:
        -----------
        n_firms : int
            Number of firm nodes
        n_banks : int
            Number of bank nodes
        topology : str
            Network topology: 'erdos_renyi', 'barabasi_albert', 'watts_strogatz'
        seed : int
            Random seed for reproducibility
        """
        self.n_firms = n_firms
        self.n_banks = n_banks
        self.topology = topology
        self.seed = seed
        np.random.seed(seed)

        # Global parameters
        self.params = {
            "v": 3.0,  # capital-output ratio
            "delta": 0.02,  # depreciation rate
            "a": 1.0,  # productivity
            "theta": 0.2,  # markup
            "omega": 0.05,  # wage adjustment weight
            "r_L": 0.05,  # loan interest rate
            "r_D": 0.02,  # deposit interest rate
            "tau_P": 10.0,  # price adjustment time
            "P": 1.0,  # initial price level
            "epsilon_D": 0.1,  # contagion strength
            "D_crit": 0.0,  # critical deposit threshold
            "K_crit": 0.1,  # critical capital threshold
            # Investment function parameters
            "I0": 0.05,
            "I_alpha": 0.1,
            "I_beta": 10.0,
            "I_pi0": 0.05,
            # Loan repayment time parameters
            "tau0": 5.0,
            "tau_gamma": 2.0,
            # Wage pressure parameters
            "phi0": 0.1,
            "lambda_star": 1.0,
        }

        self.G = None
        self.time_series = []

    def build_network(self, m: int = 3, p: float = 0.1, k: int = 4, beta: float = 0.3):
        """
        Construct the financial network.

        Parameters:
        -----------
        m : int
            Parameter for Barabasi-Albert (edges to attach)
        p : float
            Probability for Erdos-Renyi
        k : int
            Mean degree for Watts-Strogatz
        beta : float
            Rewiring probability for Watts-Strogatz
        """
        # Create firm network based on topology
        if self.topology == "erdos_renyi":
            G_firms = nx.erdos_renyi_graph(self.n_firms, p, seed=self.seed)
        elif self.topology == "barabasi_albert":
            G_firms = nx.barabasi_albert_graph(self.n_firms, m, seed=self.seed)
        elif self.topology == "watts_strogatz":
            G_firms = nx.watts_strogatz_graph(self.n_firms, k, beta, seed=self.seed)
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        # Create full network with firms and banks
        self.G = nx.DiGraph()

        # Add firm nodes with attributes
        for i in range(self.n_firms):
            firm_id = f"F{i}"
            self.G.add_node(
                firm_id,
                node_type="firm",
                K=np.random.lognormal(3.0, 0.5),  # capital
                D=np.random.lognormal(2.0, 0.5),  # deposits
                W=np.random.uniform(0.8, 1.2),  # wage
                pi=0.05,  # initial profit rate
                N=np.random.uniform(80, 120),  # labor force
                defaulted=False,
            )

        # Add bank nodes with attributes
        for j in range(self.n_banks):
            bank_id = f"B{j}"
            self.G.add_node(
                bank_id,
                node_type="bank",
                V=np.random.lognormal(4.0, 0.5),  # vaults
                E=np.random.lognormal(3.5, 0.5),  # equity
                defaulted=False,
            )

        # Add firm-firm edges (contagion network)
        for u, v in G_firms.edges():
            firm_u = f"F{u}"
            firm_v = f"F{v}"
            self.G.add_edge(
                firm_u, firm_v, edge_type="contagion", A=np.random.uniform(0.01, 0.1)
            )  # contagion weight

        # Add bank-firm lending edges
        for j in range(self.n_banks):
            bank_id = f"B{j}"
            # Each bank lends to a random subset of firms
            n_connections = np.random.randint(self.n_firms // 4, self.n_firms // 2)
            connected_firms = np.random.choice(self.n_firms, n_connections, replace=False)

            for i in connected_firms:
                firm_id = f"F{i}"
                K_f = self.G.nodes[firm_id]["K"]
                # Initial loan proportional to firm capital
                L_initial = np.random.uniform(0.1, 0.3) * K_f

                self.G.add_edge(
                    bank_id,
                    firm_id,
                    edge_type="loan",
                    L=L_initial,  # loan amount
                    nu=0.0,  # allocation share (normalized later)
                    rho=np.random.uniform(0.3, 0.7),
                )  # recovery fraction

        # Normalize allocation shares nu for each firm
        self._normalize_allocation_shares()

        print(f"Network created: {self.n_firms} firms, {self.n_banks} banks")
        print(
            f"Firm-firm edges: {sum(1 for u, v, d in self.G.edges(data=True) if d['edge_type'] == 'contagion')}"
        )
        print(
            f"Bank-firm edges: {sum(1 for u, v, d in self.G.edges(data=True) if d['edge_type'] == 'loan')}"
        )

    def _normalize_allocation_shares(self):
        """Normalize nu allocation shares so they sum to 1 for each firm."""
        firm_nodes = [n for n, d in self.G.nodes(data=True) if d["node_type"] == "firm"]

        for firm_id in firm_nodes:
            # Get all banks lending to this firm
            lending_banks = [
                u for u, v, d in self.G.in_edges(firm_id, data=True) if d["edge_type"] == "loan"
            ]

            if len(lending_banks) > 0:
                # Assign random weights and normalize
                weights = np.random.uniform(0.5, 1.5, len(lending_banks))
                weights = weights / weights.sum()

                for bank_id, nu_val in zip(lending_banks, weights):
                    self.G[bank_id][firm_id]["nu"] = nu_val

    def save_network_spec(self, filename: str = "network_spec.json"):
        """Save network topology and parameters to JSON."""
        spec = {
            "n_firms": self.n_firms,
            "n_banks": self.n_banks,
            "topology": self.topology,
            "seed": self.seed,
            "parameters": self.params,
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
        }

        with open(filename, "w") as f:
            json.dump(spec, f, indent=2)

        print(f"Network specification saved to {filename}")

    # ==================== FUNCTIONAL FORMS ====================

    def investment_rate(self, pi: float) -> float:
        """
        Investment rate as sigmoid function of profit rate.
        I(π) = I0 + α / (1 + exp(-β(π - π0)))
        """
        I0 = self.params["I0"]
        alpha = self.params["I_alpha"]
        beta = self.params["I_beta"]
        pi0 = self.params["I_pi0"]

        return I0 + alpha / (1.0 + np.exp(-beta * (pi - pi0)))

    def loan_repayment_time(self, pi: float) -> float:
        """
        Loan repayment time as function of profit rate.
        τ_L(π) = τ0 / (1 + γ·π)
        """
        tau0 = self.params["tau0"]
        gamma = self.params["tau_gamma"]

        return tau0 / (1.0 + gamma * pi)

    def wage_pressure(self, lambda_f: float) -> float:
        """
        Wage pressure function.
        Φ(λ) = φ0 · (λ - λ*)
        """
        phi0 = self.params["phi0"]
        lambda_star = self.params["lambda_star"]

        return phi0 * (lambda_f - lambda_star)

    # ==================== SIMULATION DYNAMICS ====================

    def compute_firm_quantities(self, firm_id: str) -> dict[str, float]:
        """Compute local quantities for a firm."""
        node = self.G.nodes[firm_id]

        K_f = node["K"]
        D_f = node["D"]
        W_f = node["W"]
        N_f = node["N"]

        v = self.params["v"]
        a = self.params["a"]
        r_L = self.params["r_L"]
        r_D = self.params["r_D"]
        P = self.params["P"]

        # Output
        Y_f = K_f / v

        # Labor demand
        L_lab_f = Y_f / a

        # Employment rate
        lambda_f = L_lab_f / N_f

        # Total loans from all banks
        total_loans = sum(
            self.G[bank_id][firm_id]["L"]
            for bank_id in self.G.predecessors(firm_id)
            if self.G[bank_id][firm_id]["edge_type"] == "loan"
        )

        # Profit rate: π = (PY - WL - r_L·L + r_D·D) / (PK)
        pi_f = (P * Y_f - W_f * L_lab_f - r_L * total_loans + r_D * D_f) / (P * K_f)

        return {
            "Y": Y_f,
            "L_lab": L_lab_f,
            "lambda": lambda_f,
            "pi": pi_f,
            "total_loans": total_loans,
        }

    def step(self, dt: float = 0.01):
        """
        Execute one time step of the simulation.

        Parameters:
        -----------
        dt : float
            Time step size
        """
        P = self.params["P"]
        r_L = self.params["r_L"]
        r_D = self.params["r_D"]
        delta = self.params["delta"]
        omega = self.params["omega"]
        tau_P = self.params["tau_P"]
        a = self.params["a"]
        theta = self.params["theta"]
        epsilon_D = self.params["epsilon_D"]

        # Storage for updates
        node_updates = {}
        edge_updates = {}

        # === 1. Update firms ===
        firm_nodes = [
            n for n, d in self.G.nodes(data=True) if d["node_type"] == "firm" and not d["defaulted"]
        ]

        total_W = 0  # For average wage calculation

        for firm_id in firm_nodes:
            node = self.G.nodes[firm_id]
            quants = self.compute_firm_quantities(firm_id)

            Y_f = quants["Y"]
            L_lab_f = quants["L_lab"]
            lambda_f = quants["lambda"]
            pi_f = quants["pi"]

            K_f = node["K"]
            D_f = node["D"]
            W_f = node["W"]

            # Investment rate
            I_f = self.investment_rate(pi_f)

            # Loan repayment time
            tau_L_f = self.loan_repayment_time(pi_f)

            # Capital dynamics: dK/dt = I(π)Y - δK
            dK = (I_f * Y_f - delta * K_f) * dt

            # Deposits dynamics: dD/dt = r_D·D - W·L_lab - I·PY + ΣL/τ_L
            repayment_inflow = sum(
                self.G[bank_id][firm_id]["L"] / tau_L_f
                for bank_id in self.G.predecessors(firm_id)
                if self.G[bank_id][firm_id]["edge_type"] == "loan"
            )

            dD = (r_D * D_f - W_f * L_lab_f - I_f * P * Y_f + repayment_inflow) * dt

            # Wage dynamics: dW/dt = W(Φ(λ) + ω·dP/P)
            # Note: dP/P computed later, for now use zero (or store from previous step)
            dP_over_P = 0  # Placeholder, will update in price dynamics
            dW = W_f * (self.wage_pressure(lambda_f) + omega * dP_over_P) * dt

            # Store updates
            node_updates[firm_id] = {"K": K_f + dK, "D": D_f + dD, "W": W_f + dW, "pi": pi_f}

            total_W += W_f

            # === Update bank-firm loan edges ===
            for bank_id in self.G.predecessors(firm_id):
                edge_data = self.G[bank_id][firm_id]
                if edge_data["edge_type"] != "loan":
                    continue

                L_bf = edge_data["L"]
                nu_bf = edge_data["nu"]

                # dL/dt = ν·I(π)·PY - L/τ_L(π) + r_L·L
                dL = (nu_bf * I_f * P * Y_f - L_bf / tau_L_f + r_L * L_bf) * dt

                edge_updates[(bank_id, firm_id)] = {"L": L_bf + dL}

        # === 2. Update banks ===
        bank_nodes = [
            n for n, d in self.G.nodes(data=True) if d["node_type"] == "bank" and not d["defaulted"]
        ]

        for bank_id in bank_nodes:
            node = self.G.nodes[bank_id]
            V_b = node["V"]

            dV = 0
            for firm_id in self.G.successors(bank_id):
                edge_data = self.G[bank_id][firm_id]
                if edge_data["edge_type"] != "loan":
                    continue

                L_bf = edge_data["L"]
                nu_bf = edge_data["nu"]

                firm_quants = self.compute_firm_quantities(firm_id)
                pi_f = firm_quants["pi"]
                Y_f = firm_quants["Y"]
                I_f = self.investment_rate(pi_f)
                tau_L_f = self.loan_repayment_time(pi_f)

                # dV/dt = Σ_f (L/τ_L - ν·I·PY + r_L·L)
                dV += (L_bf / tau_L_f - nu_bf * I_f * P * Y_f + r_L * L_bf) * dt

            node_updates[bank_id] = {"V": V_b + dV}

        # === 3. Update price level ===
        avg_W = total_W / len(firm_nodes) if firm_nodes else 1.0
        P_target = (avg_W / a) / (1 - theta)
        dP = ((P_target - P) / tau_P) * dt

        self.params["P"] = P + dP

        # === 4. Apply contagion ===
        for firm_id in firm_nodes:
            contagion_loss = 0
            for neighbor_id in self.G.neighbors(firm_id):
                edge_data = self.G[firm_id][neighbor_id]
                if edge_data["edge_type"] == "contagion":
                    neighbor = self.G.nodes[neighbor_id]
                    if neighbor["defaulted"]:
                        A_ij = edge_data["A"]
                        contagion_loss += epsilon_D * A_ij

            if firm_id in node_updates:
                node_updates[firm_id]["D"] -= contagion_loss

        # === 5. Apply updates ===
        for node_id, updates in node_updates.items():
            for key, value in updates.items():
                self.G.nodes[node_id][key] = value

        for (u, v), updates in edge_updates.items():
            for key, value in updates.items():
                self.G[u][v][key] = value

        # === 6. Check defaults ===
        D_crit = self.params["D_crit"]
        K_crit = self.params["K_crit"]

        for firm_id in firm_nodes:
            node = self.G.nodes[firm_id]
            if node["D"] < D_crit or node["K"] < K_crit:
                node["defaulted"] = True

                # Apply losses to connected banks
                for bank_id in self.G.predecessors(firm_id):
                    edge_data = self.G[bank_id][firm_id]
                    if edge_data["edge_type"] == "loan":
                        L_bf = edge_data["L"]
                        rho = edge_data["rho"]
                        loss = (1 - rho) * L_bf
                        self.G.nodes[bank_id]["V"] -= loss
                        self.G[bank_id][firm_id]["L"] = 0  # Write off loan

    def compute_observables(self, t: float) -> dict[str, float]:
        """Compute system-level observables for recording."""
        firm_nodes = [n for n, d in self.G.nodes(data=True) if d["node_type"] == "firm"]
        defaulted_firms = [n for n in firm_nodes if self.G.nodes[n]["defaulted"]]

        # Order parameter: fraction defaulted
        phi = len(defaulted_firms) / len(firm_nodes) if firm_nodes else 0.0

        # Profit statistics (for non-defaulted firms)
        active_firms = [n for n in firm_nodes if not self.G.nodes[n]["defaulted"]]
        if active_firms:
            profits = [self.G.nodes[n]["pi"] for n in active_firms]
            mean_pi = np.mean(profits)
            var_pi = np.var(profits)
            skew_pi = np.nan if len(profits) < 3 else pd.Series(profits).skew()
        else:
            mean_pi = var_pi = skew_pi = 0.0

        # Total system capital
        total_K = sum(self.G.nodes[n]["K"] for n in active_firms)

        # Average deposits
        avg_D = np.mean([self.G.nodes[n]["D"] for n in active_firms]) if active_firms else 0.0

        return {
            "t": t,
            "phi": phi,
            "mean_pi": mean_pi,
            "var_pi": var_pi,
            "skew_pi": skew_pi,
            "total_K": total_K,
            "avg_D": avg_D,
            "n_defaulted": len(defaulted_firms),
            "P": self.params["P"],
        }

    def run_simulation(
        self, T: float = 100.0, dt: float = 0.01, record_interval: int = 10
    ) -> pd.DataFrame:
        """
        Run the full simulation.

        Parameters:
        -----------
        T : float
            Total simulation time
        dt : float
            Time step
        record_interval : int
            Record observables every N steps

        Returns:
        --------
        pd.DataFrame
            Time series of observables
        """
        n_steps = int(T / dt)
        t = 0.0

        self.time_series = []

        print(f"Running simulation for T={T}, dt={dt} ({n_steps} steps)")

        for step in range(n_steps):
            # Execute dynamics
            self.step(dt)
            t += dt

            # Record observables
            if step % record_interval == 0:
                obs = self.compute_observables(t)
                self.time_series.append(obs)

                # Progress output
                if step % (n_steps // 10) == 0:
                    print(f"  t={t:.1f} | φ={obs['phi']:.3f} | mean_π={obs['mean_pi']:.4f}")

            # Early stopping if collapse
            if obs["phi"] >= 0.95:
                print(f"Collapse detected at t={t:.1f}")
                break

        df = pd.DataFrame(self.time_series)
        print(f"Simulation complete. Recorded {len(df)} time points.")

        return df

    def save_timeseries(self, df: pd.DataFrame, run_id: str, output_dir: str = "simulation_runs"):
        """Save time series data to CSV."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        df["run_id"] = run_id
        filename = f"{output_dir}/run_{run_id}.csv"
        df.to_csv(filename, index=False)
        print(f"Time series saved to {filename}")


# ==================== ENSEMBLE GENERATION ====================


def generate_ensemble(
    n_runs: int = 10,
    n_firms: int = 100,
    n_banks: int = 10,
    T: float = 100.0,
    dt: float = 0.01,
    output_dir: str = "simulation_runs",
    seed_base: int = 42,
) -> pd.DataFrame:
    """
    Generate ensemble of simulation runs with varying parameters.

    Parameters:
    -----------
    n_runs : int
        Number of simulation runs
    n_firms, n_banks : int
        Network size
    T : float
        Simulation time
    dt : float
        Time step
    output_dir : str
        Directory to save results
    seed_base : int
        Base seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Combined metadata for all runs
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    all_timeseries = []

    topologies = ["erdos_renyi", "barabasi_albert", "watts_strogatz"]

    for run in range(n_runs):
        print(f"\n{'=' * 60}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'=' * 60}")

        # Vary topology and parameters
        topology = topologies[run % len(topologies)]
        seed = seed_base + run

        # Create simulator with varying parameters
        sim = FinancialNetworkSimulator(
            n_firms=n_firms, n_banks=n_banks, topology=topology, seed=seed
        )

        # Vary some key parameters randomly
        sim.params["epsilon_D"] = np.random.uniform(0.05, 0.2)
        sim.params["r_L"] = np.random.uniform(0.03, 0.07)
        sim.params["delta"] = np.random.uniform(0.01, 0.03)

        # Build network
        if topology == "erdos_renyi":
            sim.build_network(p=np.random.uniform(0.05, 0.15))
        elif topology == "barabasi_albert":
            sim.build_network(m=np.random.randint(2, 5))
        else:  # watts_strogatz
            sim.build_network(k=np.random.randint(4, 8), beta=np.random.uniform(0.2, 0.5))

        # Run simulation
        df = sim.run_simulation(T=T, dt=dt, record_interval=10)

        # Save individual run
        run_id = f"{run:03d}"
        sim.save_timeseries(df, run_id, output_dir)

        # Store metadata
        metadata.append(
            {
                "run_id": run_id,
                "topology": topology,
                "seed": seed,
                "n_firms": n_firms,
                "n_banks": n_banks,
                "epsilon_D": sim.params["epsilon_D"],
                "r_L": sim.params["r_L"],
                "delta": sim.params["delta"],
                "final_phi": df["phi"].iloc[-1],
                "max_phi": df["phi"].max(),
                "T_final": df["t"].iloc[-1],
            }
        )

        all_timeseries.append(df)

    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(f"{output_dir}/metadata.csv", index=False)
    print(f"\nMetadata saved to {output_dir}/metadata.csv")

    return meta_df


# ==================== FEATURE EXTRACTION ====================


def extract_features(df: pd.DataFrame, window_size: int = 50) -> pd.DataFrame:
    """
    Extract early-warning features from time series using sliding windows.

    Parameters:
    -----------
    df : pd.DataFrame
        Time series from one simulation run
    window_size : int
        Size of sliding window

    Returns:
    --------
    pd.DataFrame
        Features for each window
    """
    features = []

    for i in range(len(df) - window_size):
        window = df.iloc[i : i + window_size]

        # Compute features within window
        phi_series = window["phi"].values
        pi_series = window["mean_pi"].values

        # Autocorrelation (AR1)
        if len(phi_series) > 1:
            ar1_phi = np.corrcoef(phi_series[:-1], phi_series[1:])[0, 1]
        else:
            ar1_phi = 0.0

        # Variance
        var_phi = np.var(phi_series)
        var_pi = np.var(pi_series)

        # Trend (linear regression slope)
        t_vals = np.arange(len(phi_series))
        if len(phi_series) > 1:
            trend_phi = np.polyfit(t_vals, phi_series, 1)[0]
        else:
            trend_phi = 0.0

        # Skewness
        skew_phi = pd.Series(phi_series).skew() if len(phi_series) > 2 else 0.0

        # Label: time to collapse (if it happens)
        final_phi = df["phi"].iloc[-1]
        t_current = window["t"].iloc[-1]
        t_final = df["t"].iloc[-1]

        if final_phi > 0.8:  # Collapse occurred
            time_to_collapse = t_final - t_current
            near_transition = 1 if time_to_collapse < 20.0 else 0
        else:
            time_to_collapse = np.inf
            near_transition = 0

        features.append(
            {
                "run_id": df["run_id"].iloc[0],
                "t": t_current,
                "phi_mean": np.mean(phi_series),
                "phi_var": var_phi,
                "phi_ar1": ar1_phi,
                "phi_trend": trend_phi,
                "phi_skew": skew_phi,
                "pi_mean": np.mean(pi_series),
                "pi_var": var_pi,
                "time_to_collapse": time_to_collapse,
                "near_transition": near_transition,
            }
        )

    return pd.DataFrame(features)


def build_feature_dataset(
    input_dir: str = "simulation_runs",
    output_file: str = "feature_dataset.csv",
    window_size: int = 50,
) -> pd.DataFrame:
    """
    Build ML-ready feature dataset from all simulation runs.

    Parameters:
    -----------
    input_dir : str
        Directory containing run CSV files
    output_file : str
        Output filename for feature dataset
    window_size : int
        Sliding window size

    Returns:
    --------
    pd.DataFrame
        Complete feature dataset
    """
    import glob

    all_features = []

    run_files = glob.glob(f"{input_dir}/run_*.csv")
    print(f"Processing {len(run_files)} runs...")

    for run_file in run_files:
        df = pd.read_csv(run_file)
        features = extract_features(df, window_size=window_size)
        all_features.append(features)

    feature_df = pd.concat(all_features, ignore_index=True)
    feature_df.to_csv(output_file, index=False)

    print(f"Feature dataset saved to {output_file}")
    print(f"  Total samples: {len(feature_df)}")
    print(f"  Positive samples (near transition): {feature_df['near_transition'].sum()}")

    return feature_df


# Example usage
if __name__ == "__main__":
    # ========== OPTION 1: Single simulation ==========
    print("OPTION 1: Running single simulation...")
    sim = FinancialNetworkSimulator(n_firms=100, n_banks=10, topology="barabasi_albert", seed=42)

    sim.build_network(m=3)
    sim.save_network_spec()

    df = sim.run_simulation(T=100.0, dt=0.01, record_interval=10)
    sim.save_timeseries(df, run_id="000", output_dir="simulation_runs")

    print("\nNetwork statistics:")
    print(
        f"Average firm degree: {np.mean([d for n, d in sim.G.degree() if n.startswith('F')]):.2f}"
    )
    print(f"Network density: {nx.density(sim.G):.4f}")

    # ========== OPTION 2: Generate ensemble ==========
    print("\n" + "=" * 60)
    print("OPTION 2: Generating ensemble dataset...")
    print("=" * 60)

    metadata = generate_ensemble(
        n_runs=5,  # Start with small number for testing
        n_firms=50,
        n_banks=5,
        T=50.0,
        dt=0.01,
        output_dir="simulation_runs",
        seed_base=100,
    )

    print("\nEnsemble metadata:")
    print(metadata)

    # ========== OPTION 3: Extract features ==========
    print("\n" + "=" * 60)
    print("OPTION 3: Extracting features for ML...")
    print("=" * 60)

    feature_df = build_feature_dataset(
        input_dir="simulation_runs", output_file="feature_dataset.csv", window_size=30
    )

    print("\nFeature summary:")
    print(feature_df.describe())

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - network_spec.json")
    print("  - simulation_runs/run_*.csv")
    print("  - simulation_runs/metadata.csv")
    print("  - feature_dataset.csv")
