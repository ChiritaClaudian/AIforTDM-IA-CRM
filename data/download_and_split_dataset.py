import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download

def split_wisdm2_dataset(data_dir='.', output_dir='splits'):
    """
    Splits the WISDM2 dataset into 3 subsets based on Subject IDs.
    """
    
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("1. Loading dataset files (this may take a moment)...")
    
    try:
        df_X = pd.read_csv(hf_hub_download(repo_id="monster-monash/WISDM2", filename="WISDM2_X.csv",repo_type="dataset"), header=None)
        df_y = pd.read_csv(hf_hub_download(repo_id="monster-monash/WISDM2", filename="WISDM2_y.csv",repo_type="dataset"), header=None)
        df_sub = pd.read_csv(hf_hub_download(repo_id="monster-monash/WISDM2", filename="WISDM2_subject_id.csv",repo_type="dataset"), header=None)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # Rename the subject column for clarity
    df_sub.columns = ['user_id']
    
    # Get list of unique users
    unique_users = df_sub['user_id'].unique()
    print(f"   Found {len(unique_users)} unique users.")
    
    # Shuffle users for random distribution
    np.random.seed(42) # Fixed seed for reproducibility
    np.random.shuffle(unique_users)
    
    # Split users into 3 roughly equal groups
    user_groups = np.array_split(unique_users, 3)
    
    members = ['Member1', 'Member2', 'Member3']
    
    for i, member in enumerate(members):
        assigned_users = user_groups[i]
        
        # Find indices where the user_id is in the assigned group
        # We use df_sub to find the indices, then select those rows from X and Y
        mask = df_sub['user_id'].isin(assigned_users)
        
        member_X = df_X[mask]
        member_y = df_y[mask]
        member_sub = df_sub[mask]
        
        # Save to separate CSVs
        m_x_path = os.path.join(output_dir, f'{member}_X.csv')
        m_y_path = os.path.join(output_dir, f'{member}_Y.csv')
        m_sub_path = os.path.join(output_dir, f'{member}_subject_id.csv')
        
        member_X.to_csv(m_x_path, index=False, header=False)
        member_y.to_csv(m_y_path, index=False, header=False)
        member_sub.to_csv(m_sub_path, index=False, header=False)

if __name__ == "__main__":
    # Run the function in the current directory
    split_wisdm2_dataset()