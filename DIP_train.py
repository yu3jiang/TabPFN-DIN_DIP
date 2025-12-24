import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles import AutoTabPFNRegressor
import joblib
from tabpfn.model.loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)

# Set the GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# -------------------
# Load the preprocessed dataset
# -------------------
data_path = "DIP.csv"
df = pd.read_csv(data_path)

# Define the names of feature columns and target column
feature_columns = [
    'MLD', 'SSH', 'SSS^2', 'U', 'V',
    'Rrs_412', 'Rrs_443', 'Rrs_488', 'Rrs_555', 'Rrs_667', 'SST',
    'POC', 'KD490', 'ln_CHL',
    'SST_x_cos_Longitude', 'SST_x_cos_Latitude'
]
target_column = 'lg_DIP'

# -------------------
# Stratified split of training and validation sets by site ID
# -------------------
# Initialize lists for training and validation sets
X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []
train_indices_list, test_indices_list = [], []  # Save indices of training and validation sets

# Record sites with special processing
sites_with_insufficient_data = []

# Get all unique site IDs
sites = df['MonitoringLocationIdentifier'].unique()
print(f"Total number of sites: {len(sites)}")

# Split data for each site individually and collect results
for site in sites:
    # Filter data for the current site
    site_data = df[df['MonitoringLocationIdentifier'] == site]
    site_indices = site_data.index.tolist()

    # Extract features and target variables (by specified column names)
    site_X = site_data[feature_columns].values  # Extract features by feature column names
    site_y = site_data[target_column].values    # Extract target variable by target column name

    # Check the number of samples
    n_samples = len(site_data)

    if n_samples < 2:
        # Fewer than 2 samples, all added to the training set
        X_train_list.append(site_X)
        y_train_list.append(site_y)
        train_indices_list.extend(site_indices)
        sites_with_insufficient_data.append(site)
        print(f"Site {site} has insufficient data ({n_samples} samples), all have been added to the training set")
    elif n_samples == 2:
        # 2 samples, 1 for training and 1 for testing
        X_train_list.append(site_X[0:1])
        X_test_list.append(site_X[1:2])
        y_train_list.append(site_y[0:1])
        y_test_list.append(site_y[1:2])
        train_indices_list.append(site_indices[0])
        test_indices_list.append(site_indices[1])
        sites_with_insufficient_data.append(site)
        print(f"Site {site} has 2 samples, split into 1 training and 1 testing sample")
    elif n_samples == 3:
        # 3 samples, 2 for training and 1 for testing
        X_train_list.append(site_X[0:2])
        X_test_list.append(site_X[2:3])
        y_train_list.append(site_y[0:2])
        y_test_list.append(site_y[2:3])
        train_indices_list.extend(site_indices[0:2])
        test_indices_list.append(site_indices[2])
        sites_with_insufficient_data.append(site)
        print(f"Site {site} has 3 samples, split into 2 training and 1 testing samples")
    elif n_samples == 4:
        # 4 samples, 2 for training and 2 for testing
        X_train_list.append(site_X[0:2])
        X_test_list.append(site_X[2:4])
        y_train_list.append(site_y[0:2])
        y_test_list.append(site_y[2:4])
        train_indices_list.extend(site_indices[0:2])
        test_indices_list.extend(site_indices[2:4])
        sites_with_insufficient_data.append(site)
        print(f"Site {site} has 4 samples, split into 2 training and 2 testing samples")
    else:
        # Sufficient samples, split into 8:2 ratio
        X_train_site, X_test_site, y_train_site, y_test_site, train_indices_site, test_indices_site = train_test_split(
            site_X, site_y, site_indices, test_size=0.2, random_state=42
        )

        # Collect split results
        X_train_list.append(X_train_site)
        X_test_list.append(X_test_site)
        y_train_list.append(y_train_site)
        y_test_list.append(y_test_site)
        train_indices_list.extend(train_indices_site)
        test_indices_list.extend(test_indices_site)

# Merge training and validation sets from all sites
X_train = np.vstack(X_train_list) if X_train_list else np.array([])
X_test = np.vstack(X_test_list) if X_test_list else np.array([])
y_train = np.hstack(y_train_list) if y_train_list else np.array([])
y_test = np.hstack(y_test_list) if y_test_list else np.array([])

# Output information about sites with special processing
if sites_with_insufficient_data:
    print(f"A total of {len(sites_with_insufficient_data)} sites have insufficient data and have been evenly distributed to training and validation sets as much as possible")

# Output basic information of training and validation sets
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_test.shape}")

# -------------------
# Initialize and train the model
# -------------------
print("\nStarting to train the AutoTabPFNRegressor model...")
regressor = TabPFNRegressor(
#    max_time=300,
    device=device
)

# Train the model
regressor.fit(X_train, y_train)

# -------------------
# Model prediction and evaluation
# -------------------
print("\nPerforming prediction on the validation set...")
y_pred = regressor.predict(X_test)

# Calculate validation set evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Safely calculate MAPE to avoid division by zero
non_zero_mask = y_test != 0
if np.sum(non_zero_mask) > 0:
    mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100  # Calculate MAPE in percentage
else:
    mape = float('inf')  # MAPE is meaningless when all target values are zero

# Output validation set evaluation results
print("\nValidation set regression model evaluation results:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Coefficient of Determination (R²): {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# -------------------
# Evaluate on the training set
# -------------------
print("\nPerforming prediction on the training set...")
y_train_pred = regressor.predict(X_train)

# Calculate training set evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Safely calculate training set MAPE
train_non_zero_mask = y_train != 0
if np.sum(train_non_zero_mask) > 0:
    train_mape = np.mean(np.abs((y_train[train_non_zero_mask] - y_train_pred[train_non_zero_mask]) / y_train[
        train_non_zero_mask])) * 100  # Calculate MAPE in percentage
else:
    train_mape = float('inf')

# Output training set evaluation results
print("\nTraining set regression model evaluation results:")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"Coefficient of Determination (R²): {train_r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")

# -------------------
# Save the model
# -------------------
save_fitted_tabpfn_model(regressor, "lgDIP.tabpfn_fit")