import numpy as np
import torch

from client import LSTMModel, TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES

# ---------- config ----------
NPZ_PATH = "fed_round_10.npz"           # final global parameters
SAVE_PATH = "base_lstm_member2_federated.pt"
# -----------------------------


def main():
    # Load npz arrays in stored order
    npz = np.load(NPZ_PATH)
    arrays = [npz[key] for key in npz.files]  # order preserved

    # Build model with same architecture as clients
    model = LSTMModel(TIMESTEPS, FEATURES, GLOBAL_NUM_CLASSES)
    state_dict = model.state_dict()

    # Copy npz arrays into state_dict in the same order
    for (name, _), arr in zip(state_dict.items(), arrays):
        state_dict[name] = torch.tensor(arr)

    model.load_state_dict(state_dict)

    # Save as a normal PyTorch checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timesteps": TIMESTEPS,
            "features": FEATURES,
            "num_classes": GLOBAL_NUM_CLASSES,
        },
        SAVE_PATH,
    )

    print(f"Saved FEDERATED baseline model to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
