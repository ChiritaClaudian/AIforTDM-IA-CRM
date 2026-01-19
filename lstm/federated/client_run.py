import argparse
import flwr as fl
from client import FlowerClient

def main():
    parser = argparse.ArgumentParser(description="Flower Client for Stage 2")
    
    # Allows you to specify which data partition to load
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    
    # Flag to enable/disable the Stage 2 Trust Mechanism (Differential Privacy)
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy")

    args = parser.parse_args()

    # Create the client with the CID and the DP flag
    # This matches your new __init__(self, cid, enable_dp=True)
    client = FlowerClient(cid=args.cid, enable_dp=args.dp)

    print(f"Starting Client {args.cid} | Differential Privacy: {args.dp}")

    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=client,
    )

if __name__ == "__main__":
    main()