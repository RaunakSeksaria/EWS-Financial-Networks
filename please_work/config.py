import torch

# --- Global Setup ---
RNG_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulation Parameters (Wilson-Cowan) ---
WEIGHT_MIN = 10.0
WEIGHT_MAX = 20.0
TAU = 8.0  # Sigmoid steepness
MU = 4.0   # Firing-rate threshold

# --- Epsilon Sweep ---
EPSILON_MIN = 0.0
EPSILON_MAX = 1.0
EPSILON_STEP = 0.01

# --- Simulation Dynamics ---
MAX_ITER = 50000  # Max iterations for steady state
DT = 0.01         # Time step for Euler integration
ABS_TOL = 1e-8    # Convergence tolerance

# --- Data Generation & Preprocessing ---
G = 100           # Number of "universes" to simulate (use 1000 for full reproduction)
WINDOW_SIZE = 20  # Sliding window size (w=20)

# --- Model Hyperparameters ---
NODE_FEATURE_DIM = 1 # Each node has 1 feature (its activity)
GIN_DIM = 64
GRU_DIM = 64
NUM_GIN_LAYERS = 6
NUM_GRU_LAYERS = 4

# --- Training Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100   # Max epochs (was 75, increased as recommended)
EARLY_STOPPING_PATIENCE = 10 # Stop after 10 epochs of no improvement