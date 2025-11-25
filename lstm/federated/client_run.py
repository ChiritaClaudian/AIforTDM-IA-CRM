import sys
import flwr as fl
from client import FlowerClient

cid = int(sys.argv[1])     # client ID passed from command line

client = FlowerClient(cid)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client,
)
