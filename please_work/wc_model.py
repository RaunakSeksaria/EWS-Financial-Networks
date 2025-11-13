import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GIN_GRU_Predictor(nn.Module):
    def __init__(self, node_feat_dim, gin_dim, gru_dim, num_gin_layers=6, num_gru_layers=4):
        super(GIN_GRU_Predictor, self).__init__()
        
        self.gin_dim = gin_dim
        self.gru_dim = gru_dim
        self.num_gin_layers = num_gin_layers
        self.num_gru_layers = num_gru_layers
        
        # --- GIN Block (6 layers) ---
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.gin_layers.append(pyg_nn.GINConv(
            nn.Sequential(
                nn.Linear(node_feat_dim, gin_dim), nn.ReLU(),
                nn.Linear(gin_dim, gin_dim), nn.ReLU()
            ), train_eps=True
        ))
        self.batch_norms.append(nn.BatchNorm1d(gin_dim))
        
        for _ in range(num_gin_layers - 1):
            self.gin_layers.append(pyg_nn.GINConv(
                nn.Sequential(
                    nn.Linear(gin_dim, gin_dim), nn.ReLU(),
                    nn.Linear(gin_dim, gin_dim), nn.ReLU()
                ), train_eps=True
            ))
            self.batch_norms.append(nn.BatchNorm1d(gin_dim))
            
        self.pool = pyg_nn.global_max_pool

        # --- GRU Block (4 layers) ---
        self.gru = nn.GRU(
            input_size=gin_dim,
            hidden_size=gru_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.1 if num_gru_layers > 1 else 0
        )
        
        # --- Output MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(gru_dim, gru_dim // 2),
            nn.ReLU(),
            nn.Linear(gru_dim // 2, 1)
        )

    def forward(self, graph_batch, batch_info):
        x, edge_index, batch_idx = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        batch_size, seq_len = batch_info

        # --- 1. GIN Processing ---
        for i in range(self.num_gin_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            
        # --- 2. Global Max Pooling ---
        x = self.pool(x, batch_idx)
        
        # --- 3. Reshape for GRU ---
        x = x.view(batch_size, seq_len, self.gin_dim)
        
        # --- 4. GRU Processing ---
        gru_output, hn = self.gru(x)
        last_step_output = gru_output[:, -1, :]
        
        # --- 5. Output MLP ---
        prediction = self.mlp(last_step_output)
        
        return prediction