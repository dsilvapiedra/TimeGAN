"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data 
"""
# Local packages
import os
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_preprocess(
    file_name: str, 
    max_seq_len: int, 
    padding_value: float = -1.0, 
    impute_method: str = "interpolation", 
    scaling_method: str = "minmax"
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load and preprocess the dataset for time-series analysis.
    Preprocessing includes:
    1. Remove outliers: Detect and remove outliers in the "level" column using z-scores.
    2. Impute missing data: Fill missing values in the "level" column using the specified imputation method ("median" or "mode").
    3. Normalize data: Apply normalization to the "level" column using the specified scaling method ("minmax" or "standard").
    4. Pad sequences: Pad the time series data to a fixed length (max_seq_len) using a specified padding value.
    5. Extract sequence length: Calculate and store the actual length of each time series before padding.

    Args:
    - file_name (str): Path to the CSV file containing the data.
    - max_seq_len (int): Maximum sequence length for time-series.
    - padding_value (float): Value used for padding sequences.
    - impute_method (str): Imputation method ("median" or "mode").
    - scaling_method (str): Scaling method ("standard" or "minmax").

    Returns:
    - processed_data (np.ndarray): Preprocessed 3D array of shape [num_samples, max_seq_len, 1].
    - time (np.ndarray): Array of sequence lengths for each sample.
    - params (List): Parameters used for scaling.
    """
    # Load data
    print("Loading data...")
    ori_data = pd.read_csv(file_name, parse_dates=["datetime"])

    # Ensure data is sorted by datetime
    ori_data = ori_data.sort_values(by="datetime").reset_index(drop=True)

    # Remove outliers (z-score > 3)
    print("Removing outliers...")
    no = len(ori_data)
    z_scores = stats.zscore(ori_data["level"], nan_policy="omit")
    ori_data = ori_data[np.abs(z_scores) < 3]
    print(f"Dropped {no - len(ori_data)} rows (outliers)")

    # Impute missing data
    print("Imputing missing data...")
    impute_vals = None
    if impute_method == "median":
        impute_vals = [ori_data["level"].median()]
    elif impute_method == "mode":
        impute_vals = [stats.mode(ori_data["level"]).mode[0]]
    elif impute_method == "interpolation":
        impute_vals = [np.nan]  # No specific value, interpolation is used
    else:
        raise ValueError("Imputation method should be 'interpolation', 'median', or 'mode'")

    # Use imputer function for imputations
    ori_data["level"] = imputer(
        curr_data=ori_data["level"].to_numpy().reshape(-1, 1),
        impute_vals=impute_vals,
        zero_fill=False
    ).ravel()

    # Check for remaining missing values
    if ori_data["level"].isnull().any():
        raise ValueError("NaN values remain after imputation")
    
    # Normalize data
    print("Scaling data...")
    scaler = MinMaxScaler() if scaling_method == "minmax" else StandardScaler()
    ori_data["level"] = scaler.fit_transform(ori_data[["level"]])
    params = {
        "scaler": "minmax" if scaling_method == "minmax" else "standard",
        "min": scaler.data_min_.item(),
        "max": scaler.data_max_.item(),
    } if scaling_method == "minmax" else {
        "mean": scaler.mean_.item(),
        "var": scaler.var_.item(),
    }

    # Initialize output arrays
    print("Preprocessing sequences...")
    num_samples = len(ori_data) // max_seq_len
    processed_data = np.full((num_samples, max_seq_len, 1), padding_value)
    time = []

    for i in tqdm(range(num_samples)):
        # Extract sequences
        start_idx = i * max_seq_len
        end_idx = start_idx + max_seq_len
        seq = ori_data.iloc[start_idx:end_idx]["level"].to_numpy()

        # Assign sequence to output
        processed_data[i, :len(seq), 0] = seq
        time.append(len(seq))

    return processed_data, np.array(time), params, max_seq_len,  padding_value


def imputer(
    curr_data: np.ndarray, 
    impute_vals: List, 
    zero_fill: bool = False
) -> np.ndarray:
    """Improved imputer for time-series water level data."""
    
    curr_data = pd.DataFrame(data=curr_data)
    
    # Imputation with provided values for each column
    impute_vals = pd.Series(impute_vals)
    imputed_data = curr_data.fillna(impute_vals)

    # Interpolation for missing values
    imputed_data = imputed_data.interpolate(method='linear', limit_direction='both')

    # Zero-fill only if explicitly required
    if zero_fill:
        imputed_data = imputed_data.fillna(0.0)

    # Check for any remaining NaN values
    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()

