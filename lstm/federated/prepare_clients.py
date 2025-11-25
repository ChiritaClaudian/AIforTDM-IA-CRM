import pandas as pd
import numpy as np
import os

def split_member2_into_clients(
    member_name="Member2",
    input_dir="../../splits",
    output_dir="clients",
    n_clients=3
):

    # -------------------------------
    # 1. Load Member2 CSV files
    # -------------------------------
    print("Loading Member2 data...")

    X_path = os.path.join(input_dir, f"{member_name}_X.csv")
    y_path = os.path.join(input_dir, f"{member_name}_Y.csv")
    sub_path = os.path.join(input_dir, f"{member_name}_subject_id.csv")

    df_X = pd.read_csv(X_path, header=None)
    df_y = pd.read_csv(y_path, header=None)
    df_sub = pd.read_csv(sub_path, header=None)
    df_sub.columns = ["user_id"]

    print(f"Loaded {len(df_X)} samples.")

    # -------------------------------
    # 2. Ensure output folder exists
    # -------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 3. Get unique users and shuffle
    # -------------------------------
    unique_users = df_sub["user_id"].unique()
    print(f"Found {len(unique_users)} unique users in {member_name}.")

    np.random.seed(42)   # <-- IMPORTANT: reproducible splits
    np.random.shuffle(unique_users)

    print("User order (shuffled, reproducible):", unique_users)

    # -------------------------------
    # 4. Split users into N clients
    # -------------------------------
    user_groups = np.array_split(unique_users, n_clients)

    print("\nUser groups for FL clients:")
    for cid, users in enumerate(user_groups):
        print(f"  Client {cid} -> users: {list(users)}")

    # -------------------------------
    # 5. Create per-client datasets
    # -------------------------------
    for cid, users in enumerate(user_groups):

        # mask rows belonging to these users
        mask = df_sub["user_id"].isin(users)

        X_client = df_X[mask]
        y_client = df_y[mask]
        sub_client = df_sub[mask]

        # output file paths
        X_out = os.path.join(output_dir, f"X_client_{cid}.csv")
        y_out = os.path.join(output_dir, f"y_client_{cid}.csv")
        sub_out = os.path.join(output_dir, f"sub_client_{cid}.csv")

        # save each component
        X_client.to_csv(X_out, index=False, header=False)
        y_client.to_csv(y_out, index=False, header=False)
        sub_client.to_csv(sub_out, index=False, header=False)

        print(f"\nClient {cid}:")
        print(f"  Users: {list(users)}")
        print(f"  Samples saved: {len(X_client)}")
        print(f"  Saved -> {X_out}, {y_out}, {sub_out}")

    print("\n✓ Done! All client files saved in:", output_dir)


if __name__ == "__main__":
    split_member2_into_clients()
