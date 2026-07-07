import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from wc_dataset import create_windows_from_universes, collate_fn_graphs
from config import EPSILON_MIN, EPSILON_MAX, WINDOW_SIZE, DEVICE

def plot_training_loss(train_losses, val_losses):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss (MSE)")
    plt.plot(val_losses, label="Validation Loss (MSE)")
    plt.title("Model Training")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("Training loss plot saved as 'training_loss.png'")

def plot_figure_4b(model, test_universe, window_size, device):
    """Generates the data and plots Figure 4(b) for a single universe."""
    model.eval()
    
    # 1. Get data for this universe
    states = test_universe['full_states']
    eps_c = test_universe['eps_c']
    eps_values = test_universe['eps_values']
    mean_activity = np.mean(states, axis=1)
    
    # 2. Create all windows (which will be normalized inside this function)
    windows, eis, labels, _ = create_windows_from_universes([test_universe], window_size)
    if not windows:
        print("Test universe is too short to plot Fig 4(b).")
        return

    # 3. Get model predictions for all windows
    predictions = []
    with torch.no_grad():
        for i in range(len(windows)):
            win, ei, lbl = windows[i], eis[i], labels[i]
            batch = [(win, ei, lbl)]
            graph_batch, _, batch_info = collate_fn_graphs(batch)
            
            pred = model(graph_batch.to(device), batch_info)
            predictions.append(pred.cpu().item())
            
    predictions = np.array(predictions)
    eps_s_values = eps_values[:len(predictions)]
    
    # 4. Calculate Relative Error
    rel_errors = np.abs(predictions - eps_c) / (EPSILON_MAX - EPSILON_MIN)
    
    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, mean_activity, 'o-', color="tab:blue", label="System", markersize=4)
    ax.axvline(eps_c, color="gray", linestyle="--", label=f"True $\\epsilon_c$ = {eps_c:.3f}")
    ax.scatter(eps_s_values, mean_activity[:len(predictions)], color='red', s=10, zorder=5, label="Prediction Start ($\epsilon_s$)")

    cmap = plt.get_cmap('viridis_r')
    norm = plt.Normalize(vmin=0, vmax=0.1)
    
    for i in range(len(predictions)):
        eps_s = eps_s_values[i]
        eps_pred = predictions[i]
        activity_s = mean_activity[i]
        color = cmap(norm(rel_errors[i]))
        ax.plot([eps_s, eps_pred], [activity_s, activity_s], color=color, linestyle="-", linewidth=0.5)
        ax.plot([eps_pred, eps_pred], [activity_s-0.5, activity_s+0.5], color=color, linewidth=0.5)

    ax.set_xlabel("$\\epsilon$ (Link Degradation)")
    ax.set_ylabel("Mean Activity")
    ax.legend(loc="best")
    ax.set_title("Fig 4(b) Reproduction: Predictions vs. System")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Relative Error")
    
    plt.savefig("figure_4b.png")
    print("Figure 4(b) plot saved as 'figure_4b.png'")

def plot_figure_4c(model, test_loader, device):
    """Generates the data and plots Figure 4(c) using the full test set."""
    model.eval()
    
    all_preds, all_labels_flat, all_lead_dists_flat = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            graph_batch, labels, batch_info, lead_distances = batch
            graph_batch = graph_batch.to(device)
            predictions = model(graph_batch.to(device), batch_info)
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels_flat.extend(labels.numpy().flatten())
            all_lead_dists_flat.extend(lead_distances.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_labels_flat = np.array(all_labels_flat)
    all_lead_dists_flat = np.array(all_lead_dists_flat)
    
    # Calculate Relative Error (Inaccuracy I)
    rel_errors = np.abs(all_preds - all_labels_flat) / (EPSILON_MAX - EPSILON_MIN)
    
    # Bin the results by lead distance
    num_bins = 40
    # --- FIX: Ensure bins have a defined range even if lead_distances is empty ---
    min_dist = min(all_lead_dists_flat) if len(all_lead_dists_flat) > 0 else 0
    max_dist = max(all_lead_dists_flat) if len(all_lead_dists_flat) > 0 else 1
    bins = np.linspace(min_dist, max_dist, num_bins + 1)
    # --- END FIX ---
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    df = pd.DataFrame({'lead_distance': all_lead_dists_flat, 'rel_error': rel_errors})
    df['bin'] = pd.cut(df['lead_distance'], bins=bins, labels=bin_centers, include_lowest=True)
    df['bin'] = pd.to_numeric(df['bin'])
    
    df_grouped = df.groupby('bin')['rel_error'].mean().reset_index()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['bin'], df_grouped['rel_error'], 'o-', markersize=5)
    plt.xlabel("Lead Distance")
    plt.ylabel("Mean Relative Error")
    plt.title("Fig 4(c) Reproduction: Mean Relative Error vs. Lead Distance")
    
    if not df_grouped.empty:
        # --- FIX: Handle potential NaN values from empty bins ---
        valid_errors = df_grouped['rel_error'].dropna()
        if not valid_errors.empty:
            plt.ylim(min(valid_errors) * 0.8, max(valid_errors) * 1.2)
        # --- END FIX ---
        
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.savefig("figure_4c.png")
    print("Figure 4(c) plot saved as 'figure_4c.png'")