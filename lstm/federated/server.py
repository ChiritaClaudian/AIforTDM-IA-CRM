import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays

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


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg strategy that saves aggregated global parameters each round
    as fed_round_{round}.npz
    """

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            np.savez(f"fed_round_{server_round}.npz", *ndarrays)
            print(
                f"[Server] Saved global parameters for round {server_round} to fed_round_{server_round}.npz")

        return aggregated_parameters, metrics


# -------------------------------
# Main server code
# -------------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 3
    ROUNDS = 10

    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": 3},  # local epochs/client
    )

    print("=== Starting Flower server ===")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
