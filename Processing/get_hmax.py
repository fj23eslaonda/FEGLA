import matplotlib.pyplot as plt
import xarray as xr
import argparse
import sys
import os
import pandas as pd
from pathlib import Path
import numpy as np

# Add the parent directory of Functions to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Simulation_processing import (
    concatenate_transects,
    interpolate_simulations_to_xarray,
    compute_mean_hmax_at_shoreline,
    select_scenarios_weighted_kmeans,
    plot_scenario_selection, 
    extract_selected_scenarios_to_h5,
    plot_random_transects
)

def main(city, n_selected_sim):
    
    # Define main data path
    mainpath = Path(f'../Data/{city}')

    # Load flood maps
    print('\nLoading data from simulations...')
    hmax_files = sorted(mainpath.glob("hmax*.nc"))
    flood_maps = {filepath.name: xr.open_dataset(filepath) for filepath in hmax_files}

    # Load transect data
    print('\nLoading transects...')
    with pd.HDFStore(mainpath / 'Transects_data.h5', mode="r") as h5_store:
        transectData = {key.lstrip('/'): h5_store[key] for key in h5_store.keys()}
    print(f'N° of transects: {len(transectData.keys())}')

    transectData = {key: df.reset_index(drop=True) for key, df in transectData.items()}
    all_transect_points = concatenate_transects(transectData)

    # Interpolate simulations and create the xarray.Dataset
    print('\nInterpolating transects over simulation...')
    ds = interpolate_simulations_to_xarray(flood_maps, all_transect_points)

    # Compute mean hmax at shoreline
    print('\nComputing mean hmax for all scenarios...')
    scenario_mean_hmax = compute_mean_hmax_at_shoreline(ds)

    # Select representative scenarios
    selected_scenarios = select_scenarios_weighted_kmeans(scenario_mean_hmax, n_clusters=n_selected_sim)

    # Plot scenario selection
    plot_scenario_selection(scenario_mean_hmax, selected_scenarios)

    # Extract selected scenarios to HDF5 file
    print('\nSaving flooded transects as h5 file ...')
    results = extract_selected_scenarios_to_h5(ds, selected_scenarios, output_h5=mainpath / "Selected_Flooded_transects.h5")

    # Plot randomly selected scenarios
    plot_random_transects(results)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process flood simulation data.")
    parser.add_argument("--city", type=str, required=True, help="City name for data processing.")
    parser.add_argument("--n_selected_sim", type=int, required=True, help="Number of selected simulations.")
    args = parser.parse_args()

    # Assign parsed arguments
    city = args.city
    n_selected_sim = args.n_selected_sim

    main(city, n_selected_sim)