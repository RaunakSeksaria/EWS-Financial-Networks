import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

# This is a custom GIN-GRU Cell
class GIN_GRU_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIN_GRU_Cell, self).__init__()
        self.hidden_dim = hidden_dim

        # GINConv for processing graph structure
        # We will feed it (x_t, h_{t-1})
        self.gin_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin = pyg_nn.GINConv(self.gin_mlp, train_eps=True)
        
        # GRUCell to update hidden state
        # It will take the output of the GIN as its "input"
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x_t, edge_index, h_prev):
        # x_t shape: [N, input_dim]
        # h_prev shape: [N, hidden_dim]
        
        # 1. Combine current input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # 2. Apply GIN layer (with empty edge_index, it's just an MLP per node)
        # gin_out shape: [N, hidden_dim]
        gin_out = torch.relu(self.gin(combined, edge_index))
        
        # 3. Apply GRUCell
        # h_next shape: [N, hidden_dim]
        h_next = self.gru_cell(gin_out, h_prev)
        
        return h_next

class GIN_GRU_Predictor(nn.Module):
    def __init__(self, node_feat_dim, gin_dim, gru_dim, num_gin_layers, num_gru_layers):
        # num_gin_layers and num_gru_layers are now unused, but kept for API
        super(GIN_GRU_Predictor, self).__init__()
        
        self.hidden_dim = gru_dim # Use GRU_DIM as the main hidden dim
        
        # --- Input Normalization ---
        # This is crucial. It normalizes node features (N, C)
        self.input_bn = nn.BatchNorm1d(node_feat_dim)

        # --- GIN-GRU Cell ---
        # We'll use one GIN-GRU cell and run it in a loop
        self.cell = GIN_GRU_Cell(node_feat_dim, self.hidden_dim)
        
        # --- Pooling ---
        self.pool = pyg_nn.global_max_pool

        # --- Output MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, graph_batch, batch_info):
        x_all_steps, edge_index, batch_idx = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        batch_size, seq_len = batch_info
        
        # --- 1. Normalize all inputs at once ---
        x_all_steps = self.input_bn(x_all_steps)
        
        # --- 2. Un-batch the temporal data ---
        # x_all_steps shape: [B*w*N, node_feat_dim]
        # We need to get x_t for each step t
        # We find the number of nodes per graph (N_i)
        # num_nodes_per_graph will be [N_1, N_2, ..., N_{B*w}]
        _, counts = torch.unique(batch_idx, return_counts=True)
        
        # x_steps shape: [B*w, N_max, node_feat_dim]
        # This is a bit complex, but pyg_nn.to_dense_batch does this
        # It pads graphs in the batch to have the same size N_max
        x_steps_padded, mask = pyg_nn.to_dense_batch(x_all_steps, batch_idx)
        
        # Reshape to [B, w, N_max, node_feat_dim]
        x_steps_padded = x_steps_padded.view(batch_size, seq_len, -1, node_feat_dim)
        
        # Get edge_index per batch item (it's the same empty one, but we need to batch it)
        # This is tricky because edge_index is for B*w graphs
        # We need to get the edge_index for just one time step, i.e., B graphs
        # We can just rebuild a "Data" object for each *batch item* (not time step)
        
        # Simpler approach: Process one batch item at a time in a loop
        # This avoids all the complex batching/unbatching code
        
        # --- RE-IMPLEMENTING FORWARD (Simpler, No PyG Batching) ---
        # This function will now be called from a *new* collate_fn
        # x_list: List[Tensor[w, N, 1]]
        # ei_list: List[Tensor[2, E]]
        
        # Oh, wait. The collate_fn is already written. Let's stick with the complex way.
        
        # We need to get the edge_index for each graph in the batch (size B)
        # Since they are all identical (empty), we can just take the first one
        # ... but N varies.
        
        # This is the problem. The PyG Batch object is for B*w graphs.
        # The GRU needs to run B times (in parallel) over w steps.
        
        # Let's abandon the collate_fn and do the batching in the model
        # This requires a new collate_fn and dataset.
        
        # --- OK, let's try a different architecture that FITS the current batching ---
        # This is closer to the paper's description anyway.
        
        # x_all_steps shape: [TotalNodes, 1]
        x = self.input_bn(x_all_steps)
        
        # --- GIN Block (as before) ---
        for i in range(self.num_gin_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
        
        # --- Pool (as before) ---
        # x_pooled shape: [B * w, gin_dim]
        x_pooled = self.pool(x, batch_idx)
        
        # --- Reshape for GRU (as before) ---
        # x_seq shape: [B, w, gin_dim]
        x_seq = x_pooled.view(batch_size, seq_len, self.gin_dim)
        
        # --- GRU (as before) ---
        # gru_output shape: [B, w, gru_dim]
        gru_output, hn = self.gru(x_seq)
        
        # --- MODIFICATION: Use *all* GRU outputs, not just the last one ---
        # The final hidden state (hn) might be too simple.
        # Let's average the GRU's output over the entire window.
        # This allows the model to see the "whole picture" of the sequence.
        
        # avg_gru_output shape: [B, gru_dim]
        avg_gru_output = torch.mean(gru_output, dim=1) 
        
        # --- Output MLP (as before) ---
        # prediction shape: [B, 1]
        prediction = self.mlp(avg_gru_output)
        
        return prediction