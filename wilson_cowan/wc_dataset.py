import random

import numpy as np
import torch

# Import constants from config
from config import WINDOW_SIZE
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


def create_windows_from_universes(universes, window_size):
    """
    Processes all universes to create a flat list of windows, labels,
    and their associated metadata (edge_index, lead_distance).
    """
    all_windows = []
    all_edge_indices = []
    all_labels = []
    all_lead_distances = []

    empty_edge_index = torch.empty(2, 0, dtype=torch.long)

    for universe in universes:
        states = universe["full_states"]
        eps_c = universe["eps_c"]
        idx_c = universe["transition_idx"]
        eps_values = universe["eps_values"]
        # Normalization factor from the pre-transition segment
        mean_high_0 = universe["mean_high_0"]

        max_start_idx = idx_c - window_size
        if max_start_idx < 0:
            continue

        for s in range(max_start_idx + 1):
            i = s
            window_data = states[i : i + window_size]

            # Normalize the window data
            if mean_high_0 > 1e-5:  # Avoid division by zero
                window_data_normalized = window_data / mean_high_0
            else:
                window_data_normalized = window_data

            eps_s = eps_values[i]
            lead_distance = np.abs(eps_s - eps_c)

            # Use the normalized data
            all_windows.append(
                torch.tensor(window_data_normalized, dtype=torch.float).unsqueeze(-1)
            )
            all_edge_indices.append(empty_edge_index)  # Pass the empty index
            all_labels.append(torch.tensor([eps_c], dtype=torch.float))
            all_lead_distances.append(lead_distance)

    return all_windows, all_edge_indices, all_labels, all_lead_distances


class GraphWindowDataset(Dataset):
    """Custom PyTorch Dataset for graph sequences."""

    def __init__(self, windows, edge_indices, labels, lead_distances=None):
        self.windows = windows
        self.edge_indices = edge_indices
        self.labels = labels
        self.lead_distances = lead_distances

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = (self.windows[idx], self.edge_indices[idx], self.labels[idx])
        if self.lead_distances:
            return (*item, self.lead_distances[idx])
        return item


def collate_fn_graphs(batch):
    """
    Custom collate function to handle batches of (window, edge_index, label).
    It creates a single giant batched graph for all graphs in all sequences.
    """
    windows, edge_indices, labels = [], [], []
    lead_distances = []
    has_lead_dist = len(batch[0]) == 4

    for item in batch:
        windows.append(item[0])
        edge_indices.append(item[1])
        labels.append(item[2])
        if has_lead_dist:
            lead_distances.append(item[3])

    labels_batch = torch.stack(labels)

    data_list = []
    for i in range(len(windows)):  # Loop over batch B
        window = windows[i]  # Shape (w, N_i, 1)
        ei = edge_indices[i]  # Shape (2, E_i)
        for t in range(window.shape[0]):  # Loop over sequence w
            x_t = window[t]
            data_list.append(Data(x=x_t, edge_index=ei))

    graph_batch = Batch.from_data_list(data_list)

    batch_size = len(windows)
    # Handle edge case where batch might be empty
    seq_len = windows[0].shape[0] if batch_size > 0 else 0

    if has_lead_dist:
        return graph_batch, labels_batch, (batch_size, seq_len), torch.tensor(lead_distances)

    return graph_batch, labels_batch, (batch_size, seq_len)


def get_data_loaders(universes, batch_size, seed):
    """
    Splits universes and creates train, val, and test DataLoaders.
    """
    num_universes = len(universes)
    num_train = int(num_universes * 0.7)
    num_val = int(num_universes * 0.1)

    # Shuffle universes for splitting
    random.Random(seed).shuffle(universes)

    train_universes = universes[:num_train]
    val_universes = universes[num_train : num_train + num_val]
    test_universes = universes[num_train + num_val :]

    print(
        f"Splitting universes: {len(train_universes)} train, {len(val_universes)} val, {len(test_universes)} test"
    )

    # Create windowed datasets for each split
    train_windows, train_ei, train_labels, _ = create_windows_from_universes(
        train_universes, WINDOW_SIZE
    )
    val_windows, val_ei, val_labels, _ = create_windows_from_universes(val_universes, WINDOW_SIZE)
    test_windows, test_ei, test_labels, test_lead_dist = create_windows_from_universes(
        test_universes, WINDOW_SIZE
    )

    print(
        f"Total windows: {len(train_windows)} train, {len(val_windows)} val, {len(test_windows)} test"
    )

    # Create Datasets
    train_dataset = GraphWindowDataset(train_windows, train_ei, train_labels)
    val_dataset = GraphWindowDataset(val_windows, val_ei, val_labels)
    test_dataset = GraphWindowDataset(test_windows, test_ei, test_labels, test_lead_dist)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_graphs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_graphs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_graphs
    )

    print("DataLoaders created.")
    return train_loader, val_loader, test_loader, test_universes
