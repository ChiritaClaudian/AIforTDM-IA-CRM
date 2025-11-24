# import pandas as pd
# import numpy as np
# import os

# # Load data
# X = pd.read_csv("../../splits/Member2_X.csv")
# y = pd.read_csv("../../splits/Member2_Y.csv")
# subjects = pd.read_csv("../../splits/Member2_subject_id.csv")

# num_clients = 3
# os.makedirs("clients", exist_ok=True)

# # Convert to array
# subjects = subjects.values.ravel()

# # Unique subjects
# unique_subj = np.unique(subjects)

# # Split subjects into client groups
# subj_splits = np.array_split(unique_subj, num_clients)

# for cid, subj_group in enumerate(subj_splits):
#     # Find all rows belonging to these subjects
#     idx = [i for i, sbj in enumerate(subjects) if sbj in subj_group]

#     X_client = X.iloc[idx]
#     y_client = y.iloc[idx]

#     X_client.to_csv(f"clients/X_client_{cid}.csv", index=False)
#     y_client.to_csv(f"clients/y_client_{cid}.csv", index=False)

#     print(f"Client {cid}: {len(idx)} samples, subjects={list(subj_group)}")

# print("Done. Federated subject-based client datasets saved.")


import pandas as pd
import numpy as np
import os

X = pd.read_csv("../../splits/Member2_X.csv", header=None)
y = pd.read_csv("../../splits/Member2_Y.csv", header=None)

assert len(X) == len(y), "X and Y sizes do NOT match!"

num_clients = 3
indices = np.array_split(np.arange(len(X)), num_clients)

os.makedirs("./clients", exist_ok=True)

for cid, idx in enumerate(indices):
    X.iloc[idx].to_csv(f"./clients/X_client_{cid}.csv",
                       index=False, header=False)
    y.iloc[idx].to_csv(f"./clients/y_client_{cid}.csv",
                       index=False, header=False)

print("Split completed with equal sizes.")
