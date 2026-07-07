"""Smoke tests for the Wilson-Cowan GIN-GRU model and data pipeline.

The full simulation uses networks of several hundred nodes with long
relaxation loops, so the model is exercised on a small synthetic batch built
through the real dataset/collate path rather than a live simulation.
"""

import torch

from wc_dataset import GraphWindowDataset, collate_fn_graphs
from wc_model import GIN_GRU_Predictor


def _synthetic_batch(batch_size=4, window=5, n_nodes=12):
    """Build a batch the same way create_windows_from_universes does."""
    empty_edge_index = torch.empty(2, 0, dtype=torch.long)
    windows, edges, labels = [], [], []
    for _ in range(batch_size):
        windows.append(torch.randn(window, n_nodes, 1))
        edges.append(empty_edge_index)
        labels.append(torch.tensor([0.5], dtype=torch.float))
    dataset = GraphWindowDataset(windows, edges, labels)
    return collate_fn_graphs([dataset[i] for i in range(len(dataset))])


def test_model_forward_returns_finite_prediction_per_sample():
    torch.manual_seed(0)
    batch_size = 4
    graph_batch, labels, batch_info = _synthetic_batch(batch_size=batch_size)

    model = GIN_GRU_Predictor(
        node_feat_dim=1, gin_dim=16, gru_dim=16, num_gin_layers=2, num_gru_layers=1
    )
    model.eval()
    with torch.no_grad():
        preds = model(graph_batch, batch_info)

    assert preds.shape == (batch_size, 1)
    assert torch.isfinite(preds).all()
    assert labels.shape == (batch_size, 1)


def test_model_is_trainable_one_step():
    torch.manual_seed(0)
    graph_batch, labels, batch_info = _synthetic_batch()
    model = GIN_GRU_Predictor(
        node_feat_dim=1, gin_dim=16, gru_dim=16, num_gin_layers=2, num_gru_layers=1
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    preds = model(graph_batch, batch_info)
    loss = torch.nn.functional.mse_loss(preds, labels)
    loss.backward()
    opt.step()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one parameter to receive a gradient"
