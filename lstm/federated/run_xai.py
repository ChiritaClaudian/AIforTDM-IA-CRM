import torch
import numpy as np
import matplotlib.pyplot as plt
from client import LSTMModel, load_client_data
from explainability import visualize_har_explanation

# 1. Configuration (Must match your training)
TIMESTEPS = 100
FEATURES = 3
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "global_model_final_dp.pth" 

# 2. Load the Model
model = LSTMModel(TIMESTEPS, FEATURES, NUM_CLASSES).to(DEVICE)
try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Successfully loaded model from {model_path}")
except FileNotFoundError:
    print(f"Error: {model_path} not found. Ensure you saved the weights in server.py!")
    exit()

# 3. Load actual data to explain
# We use cid=0 to grab some test samples
_, X_test, _, y_test = load_client_data(cid=0)

# 4. Select a sample to explain
# Let's pick the first sample from the test set
sample_idx = 0
input_tensor = X_test[sample_idx:sample_idx+1].to(DEVICE) # Shape (1, 100, 3)
original_label = y_test[sample_idx].item()

# 5. Run Explanation
print(f"Generating explanation for sample {sample_idx}...")

# Call the function
visualize_har_explanation(model, input_tensor, target_label=original_label)

# INSTEAD OF plt.show(), use plt.savefig()
plt.savefig("har_explanation_dp.png")
print("Explanation saved as har_explanation.png")