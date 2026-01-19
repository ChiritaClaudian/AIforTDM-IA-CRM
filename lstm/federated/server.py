import flwr as fl
import numpy as np

import torch
from client import LSTMModel # Import your model class

# 1. Define a function that saves the model weights
def get_evaluate_fn():
    # This runs on the server after every round
    def evaluate(server_round, parameters, config):
        if server_round == 1: # Save only on the final round
            model = LSTMModel(timesteps=100, features=3, num_classes=6)
            
            # Convert Flower parameters back to PyTorch state_dict
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Save the file
            torch.save(model.state_dict(), "global_model_final_dp.pth")
            print("--- Global Model Saved to global_model_final.pth ---")
        return None # We don't need actual evaluation here, just the save
    return evaluate

# -------------------------------
# Metric aggregation functions
# -------------------------------
def fit_metrics_aggregation(metrics_list):
    """
    Aggregate training metrics from all clients.
    Computes weighted average of loss and accuracy.
    """
    total_examples = 0
    weighted_loss = 0.0
    weighted_acc = 0.0
    for m in metrics_list:
        if m is None:
            continue
        num_examples, metrics = m
        if metrics:
            if "loss" in metrics:
                weighted_loss += metrics["loss"] * num_examples
            if "accuracy" in metrics:
                weighted_acc += metrics["accuracy"] * num_examples
            total_examples += num_examples
    return {
        "loss": float(weighted_loss / total_examples) if total_examples > 0 else 0.0,
        "accuracy": float(weighted_acc / total_examples) if total_examples > 0 else 0.0
    }

def evaluate_metrics_aggregation(metrics_list):
    """
    Aggregate evaluation metrics from all clients.
    Computes weighted average of accuracy only.
    """
    total_examples = 0
    weighted_acc = 0.0
    for m in metrics_list:
        if m is None:
            continue
        num_examples, metrics = m
        if metrics and "accuracy" in metrics:
            weighted_acc += metrics["accuracy"] * num_examples
            total_examples += num_examples
    return {"accuracy": float(weighted_acc / total_examples) if total_examples > 0 else 0.0}

# -------------------------------
# Main server code
# -------------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 2
    ROUNDS = 1

    # Flower FedAvg strategy with custom metric aggregation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        on_fit_config_fn=lambda rnd: {"local_epochs": 3},  # Number of local epochs per round
        evaluate_fn=get_evaluate_fn(),
    )

    print("=== Starting Flower server ===")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
