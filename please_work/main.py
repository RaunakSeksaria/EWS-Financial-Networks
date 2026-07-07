import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random

# Import from our custom modules
from config import * # This will now import EARLY_STOPPING_PATIENCE
from wc_simulation import generate_universes
from wc_dataset import get_data_loaders
from wc_model import GIN_GRU_Predictor
from wc_utils import plot_training_loss, plot_figure_4b, plot_figure_4c

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seeds(RNG_SEED)
    print(f"Using device: {DEVICE}")

    # --- 1. Data Generation ---
    universes = generate_universes(G, RNG_SEED)

    # --- 2. Data Preprocessing ---
    train_loader, val_loader, test_loader, test_universes = get_data_loaders(
        universes, BATCH_SIZE, RNG_SEED
    )
    
    # --- MODIFICATION: Exit if no data was generated ---
    if len(train_loader.dataset) == 0:
        print("No training data was generated. Exiting.")
        return
    # --- END MODIFICATION ---

    # --- 3. Model Initialization ---
    model = GIN_GRU_Predictor(
        node_feat_dim=NODE_FEATURE_DIM,
        gin_dim=GIN_DIM,
        gru_dim=GRU_DIM,
        num_gin_layers=NUM_GIN_LAYERS,
        num_gru_layers=NUM_GRU_LAYERS
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

    # --- 4. Training Loop ---
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training for up to {NUM_EPOCHS} epochs (with patience of {EARLY_STOPPING_PATIENCE})...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        
        # Use tqdm for train_loader
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False):
            graph_batch, labels, batch_info = batch
            graph_batch = graph_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(graph_batch, batch_info)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_info[0]
        
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            # --- MODIFICATION: Add tqdm to val_loader ---
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False):
            # --- END MODIFICATION ---
                graph_batch, labels, batch_info = batch
                graph_batch = graph_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                
                predictions = model(graph_batch, batch_info)
                loss = criterion(predictions, labels)
                epoch_val_loss += loss.item() * batch_info[0]
                
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_wc.pth")
            print(f"  -> New best model saved with Val Loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
        # --- END NEW ---

    print("Training complete.")
    
    # --- 5. Evaluation & Plotting ---
    print("Generating training loss plot...")
    plot_training_loss(train_losses, val_losses)
    
    print("Loading best model for evaluation...")
    try:
        model.load_state_dict(torch.load("best_model_wc.pth"))
    except FileNotFoundError:
        print("Error: 'best_model_wc.pth' not found. Evaluation will use the last epoch's model.")
        
    print("Generating Figure 4(b) reproduction...")
    if test_universes:
        plot_figure_4b(model, test_universes[0], WINDOW_SIZE, DEVICE)
    else:
        print("No test universes available to plot Figure 4(b).")
        
    print("Generating Figure 4(c) reproduction...")
    if test_loader.dataset:
        plot_figure_4c(model, test_loader, DEVICE)
    else:
        print("No test data available to plot Figure 4(c).")

if __name__ == "__main__":
    main()