import os
import glob
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import torch
from tqdm import tqdm
from datetime import datetime
import re
import pickle
from collections import defaultdict
import joblib

# Import TabPFN model loading utilities
from tabpfn import TabPFNRegressor
from tabpfn.model.loading import load_fitted_tabpfn_model

# Set the GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# Optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model and data paths
MODEL_PATH = 'lgDIP.tabpfn_fit'
NC_DIR = r'./nCS_DINEOF'
OUTPUT_DIR = r'./nCS_DINEOF_OutPut_DIP'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the original parameters to be extracted
ORIGINAL_PARAMS = [
    'MLD', 'SSH', 'SSS', 'U', 'V',
    'Rrs_412', 'Rrs_443', 'Rrs_488', 'Rrs_555', 'Rrs_667',
    'SST', 'POC', 'KD490', 'CHL',
    'lon', 'lat'
]

# Construct the final feature name list
FEATURE_NAMES = [
    'MLD', 'SSH', 'SSS^2', 'U', 'V',
    'Rrs_412', 'Rrs_443', 'Rrs_488', 'Rrs_555', 'Rrs_667', 'SST',
    'POC', 'KD490', 'ln_CHL',
    'SST_x_cos_Longitude', 'SST_x_cos_Latitude'
]

# Batch size (adjust based on GPU memory capacity)
BATCH_SIZE = 32768


def load_model(model_path):
    """Load the trained TabPFN model and ensure GPU utilization"""
    try:
        # Detect available GPU devices
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if device == "cuda":
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GiB")
            # Load model with TabPFN's dedicated method and specify GPU device
            model = load_fitted_tabpfn_model(model_path, device=device)
        else:
            # Load model to CPU if no GPU is available
            model = load_fitted_tabpfn_model(model_path, device="cpu")

        print(f"Model loaded successfully and using {device}: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_year_month_from_nc(nc_file):
    """Extract year and month information from filename or metadata"""
    filename = os.path.basename(nc_file)

    patterns = [
        r'20(0[3-9]|1[0-9]|2[0-2])(0[1-9]|1[0-2])',
        r'20(0[3-9]|1[0-9]|2[0-2])-(0[1-9]|1[0-2])',
        r'20(0[3-9]|1[0-9]|2[0-2])_(0[1-9]|1[0-2])'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return 2000 + year if year < 100 else year, month

    try:
        with xr.open_dataset(nc_file) as ds:
            if 'time' in ds:
                time = pd.to_datetime(ds['time'].values[0])
                return time.year, time.month
    except:
        pass

    try:
        timestamp = os.path.getctime(nc_file)
        dt = datetime.fromtimestamp(timestamp)
        return dt.year, dt.month
    except:
        print(f"Warning: Failed to extract year and month information for {filename}, using default values")
        return 2003, 1


def process_monthly_data(monthly_files, model):
    """Process monthly NC files, predict DIP in batches, and save results"""
    if not monthly_files:
        return False

    # Extract longitude and latitude and generate 2D grid
    first_file = monthly_files[0]
    with xr.open_dataset(first_file) as ds:
        lon_1d = ds['lon'].values if 'lon' in ds else ds['longitude'].values
        lat_1d = ds['lat'].values if 'lat' in ds else ds['latitude'].values
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        rows, cols = lon_grid.shape  # (n_lat, n_lon)

    # Extract year and month information
    year, month = extract_year_month_from_nc(first_file)

    # Create data matrix for all files in the current month
    all_data = []
    for nc_file in monthly_files:
        with xr.open_dataset(nc_file) as ds:
            param_data = []
            for param in ORIGINAL_PARAMS:
                if param == 'lon':
                    data = lon_grid.flatten()
                elif param == 'lat':
                    data = lat_grid.flatten()
                elif param in ds:
                    data = ds[param].values
                    if len(data.shape) == 3:
                        data = data[0]  # Take the first time step
                    data = data.flatten()
                else:
                    print(f"Warning: {param} not found in {os.path.basename(nc_file)}")
                    data = np.full(rows * cols, np.nan)
                param_data.append(data)
            sample = np.column_stack(param_data)
            all_data.append(sample)

    # Merge all samples
    all_samples = np.vstack(all_data)

    # Create DataFrame and remove rows with all NaN values
    df = pd.DataFrame(all_samples, columns=ORIGINAL_PARAMS)
    valid_rows = df.dropna(how='all').copy()

    if valid_rows.empty:
        print(f"{year}-{month:02d} has no valid data, skipping processing")
        return False

    # Feature transformation
    transformed_features = []
    epsilon = 1e-8

    transformed_features.append(valid_rows['MLD'])
    transformed_features.append(valid_rows['SSH'])
    transformed_features.append(-(valid_rows['SSS'] ** 2))
    transformed_features.append(valid_rows['U'])
    transformed_features.append(valid_rows['V'])
    transformed_features.append(valid_rows['Rrs_412'])
    transformed_features.append(valid_rows['Rrs_443'])
    transformed_features.append(valid_rows['Rrs_488'])
    transformed_features.append(valid_rows['Rrs_555'])
    transformed_features.append(valid_rows['Rrs_667'])
    transformed_features.append(valid_rows['SST'])
    transformed_features.append(valid_rows['POC'])
    transformed_features.append(valid_rows['KD490'])

    # Handle logarithmic transformation of CHL
    valid_chl = valid_rows['CHL'].where(valid_rows['CHL'] > 0, epsilon)
    transformed_features.append(np.log(valid_chl + epsilon))

    # Handle interaction features between latitude/longitude and SST
    lon_rad = np.radians(valid_rows['lon'])
    lat_rad = np.radians(valid_rows['lat'])
    transformed_features.append(valid_rows['SST'] * np.cos(lon_rad))
    transformed_features.append(valid_rows['SST'] * np.cos(lat_rad))

    # Construct final feature DataFrame
    final_features = pd.DataFrame(np.column_stack(transformed_features), columns=FEATURE_NAMES)
    total_samples = len(final_features)
    print(f"Number of valid samples: {total_samples}, predicting in batches (each batch with {min(BATCH_SIZE, total_samples)} samples)")

    # Batch prediction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_predictions = []

    # Calculate total number of batches
    total_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(total_batches), desc=f"Batch prediction {year}-{month:02d}"):
        # Calculate start and end indices for current batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)
        batch_features = final_features.iloc[start_idx:end_idx]

        # Convert to numpy array (TabPFN accepts numpy arrays directly)
        batch_np = batch_features.values.astype(np.float32)

        # Predict current batch
        with torch.no_grad():  # Disable gradient calculation to save memory
            batch_log_pred = model.predict(batch_np)

        # Collect prediction results
        log_predictions.extend(batch_log_pred)

        # Clear GPU memory for current batch
        if device == "cuda":
            torch.cuda.empty_cache()

    # Convert to actual DIP values
    predictions = 10 ** np.array(log_predictions)

    # Create complete prediction array
    full_predictions = np.full(rows * cols, np.nan)
    full_predictions[valid_rows.index] = predictions

    # Reshape back to grid format
    din_grid = full_predictions.reshape(rows, cols)

    # Save results
    output_filename = f"DIP_prediction_{year}_{month:02d}.h5"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        with h5py.File(output_path, 'w') as hf:
            group = hf.create_group('DIP_Predictions')
            group.create_dataset('longitude', data=lon_1d, compression='gzip')
            group.create_dataset('latitude', data=lat_1d, compression='gzip')
            group.create_dataset('DIP', data=din_grid, compression='gzip')

            group.attrs['year'] = year
            group.attrs['month'] = month
            group.attrs['creation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            group.attrs['model_type'] = 'TabPFN'
            group.attrs['processing_note'] = 'Feature transformation: -SSS², ln(CHL), SST*cos(longitude/latitude); Results converted from log10(DIP) to actual DIP values'
            group.attrs['device_used'] = device
            group.attrs['batch_size'] = BATCH_SIZE

        print(f"Successfully saved {output_filename}")
        return True
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")
        return False


def main():
    """Main function: Process NC files by month and generate DIP products"""
    model = load_model(MODEL_PATH)
    if model is None:
        print("Failed to load model, program exiting")
        return

    # Get all NC files
    nc_files = glob.glob(os.path.join(NC_DIR, '*.nc'))
    if not nc_files:
        print(f"No NC files found in {NC_DIR}")
        return

    print(f"Found {len(nc_files)} NC files, starting processing...")

    # Group NC files by year and month
    monthly_groups = defaultdict(list)
    for nc_file in nc_files:
        year, month = extract_year_month_from_nc(nc_file)
        monthly_groups[(year, month)].append(nc_file)

    # Process data for each month
    success_count = 0
    for (year, month), files in tqdm(monthly_groups.items(), desc="Processing months"):
        print(f"\nProcessing data for {year}-{month:02d} ({len(files)} files)...")
        if process_monthly_data(files, model):
            success_count += 1

    print("\nAll months processed!")
    print(f"Successfully generated {success_count} monthly product files")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    print(f"Processing start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print(f"Processing end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")