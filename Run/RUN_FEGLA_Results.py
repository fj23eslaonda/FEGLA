#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Transect_processing import find_contour_coordinates

from Functions.Results_processing import (
    load_height_iterations,
    extract_cte_squared_name,
    extract_linear_name,
    compute_error_for_extension,
    plot_1Derror_distribution,
    plot_2Derror_distribution,
    plot_best_fit_models,
    integrate_hmax_predictions,
    plot_flood_extent_contours,
    plot_froude_parameterization,
    compute_flood_extent_areas,
    initial_heights_by_scenario,
    plot_flooding_curve
)


#--------------------------------------------------------
#
# Main function
#
#--------------------------------------------------------
def main(city):

    outputPath = Path(f'../Results/{city}')
    outfigPath = outputPath / 'Figs'
    outfigPath.mkdir(parents=True, exist_ok=True)

    ## Load flooded transect
    flooded_hmax = {}
    # Open the HDF5 file and extract only the 'hmax' values
    with pd.HDFStore(f"../Data/{city}/Selected_Flooded_transects.h5", "r") as store:
        keys = store.keys()  # Get all keys
        for key in tqdm(keys, desc="Loading hmax from flooded transects"):  # Use tqdm for progress tracking
            df = store[key]  # Load the DataFrame
            if 'hmax' in df.columns:  # Ensure 'hmax' exists in the DataFrame
                flooded_hmax[key.strip('/')] = df['hmax'].dropna().values  # Store only hmax values
            # Explicitly delete the DataFrame to free memory
            del df
    # Print the structure of the new dictionary
    print(f"Loaded {len(flooded_hmax)} keys in flooded_hmax dictionary\n")

    # Load all datasets
    resultsCte_dict = load_height_iterations(outputPath, '*cte*', extract_cte_squared_name)
    resultsSquared_dict = load_height_iterations(outputPath, '*squared*', extract_cte_squared_name)
    resultsLinear_dict = load_height_iterations(outputPath, '*linear*', extract_linear_name)

    # Compute error
    error_resultsCTE     = compute_error_for_extension(resultsCte_dict, flooded_hmax)
    print("Error computation completed for Constant Froude\n")
    error_resultsSquared = compute_error_for_extension(resultsSquared_dict, flooded_hmax)
    print("Error computation completed for Squared Froude\n")
    error_resultslinear  = compute_error_for_extension(resultsLinear_dict, flooded_hmax)
    print("Error computation completed for Linear Froude\n")

    # Plot error distribution
    min_F0_cte, min_error_cte                       = plot_1Derror_distribution(error_resultsCTE, outfigPath, froude='constante')
    min_F0_squared, min_error_squared               = plot_1Derror_distribution(error_resultsSquared, outfigPath, froude='squared')
    min_F0_linear, min_FR_linear, min_error_linear  = plot_2Derror_distribution(error_resultslinear, outfigPath)

    # Plot froude parameterization
    plot_froude_parameterization(min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear, outfigPath)

    # Plot best-fit models along scenarios
    selected_scenarios = plot_best_fit_models(error_resultsCTE, error_resultsSquared, error_resultslinear, 
                            resultsCte_dict, min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear, outfigPath,
                            n_clusters=9)
    
    print("Plotting area for 9 representative scenarios")
    # Plot area for 9 scenarios
    with pd.HDFStore(f"../Data/{city}/Selected_Flooded_transects.h5", mode="r") as h5_store:
        transectData = {key.lstrip('/'): h5_store[key] for key in tqdm(h5_store.keys(), desc="Loading hmax from flooded transects")}

    updated_transectData = integrate_hmax_predictions(transectData, 
                                                        resultsCte_dict, resultsSquared_dict, resultsLinear_dict,
                                                        min_F0_cte, min_error_cte, 
                                                        min_F0_squared, min_error_squared, 
                                                        min_F0_linear, min_FR_linear, min_error_linear)

    flood_extent_dict = compute_flood_extent_areas(updated_transectData)

    # Load bathy
    bathy_nc = xr.open_dataset(f'../Data/{city}/Bathymetry.nc')
    grid_lat = bathy_nc["lat"].values
    grid_lon = bathy_nc["lon"].values
    bathy = bathy_nc["bathy"].values

    shoreline = find_contour_coordinates(grid_lon, grid_lat, bathy, level=0)
    ## Remove values from the ends of the line
    shoreline = shoreline[10:-10]

    # Create a figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex='col', sharey='row')

    # Get all available scenario keys from flood_extent_dict
    scenario_keys = list(flood_extent_dict.keys())

    # Loop through each subplot and plot the corresponding scenario by index
    for i, (ax, scenario_index) in enumerate(zip(axes.flat, selected_scenarios)):
        plot_flood_extent_contours(
            flood_extent_dict, scenario=scenario_index, 
            grid_lon=grid_lon, grid_lat=grid_lat, 
            bathy=bathy, shoreline=shoreline, 
            ax=ax, show_legend=(i == 0), show_colorbar=(i % 3 == 2)
        )

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(outfigPath / f"Area_extension_SWEvsFEGLA.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.1)

    # compute hmax at shoreline
    initialH = initial_heights_by_scenario(updated_transectData)

    Hmean = [stats[2] for s_key, stats in initialH.items() if s_key.startswith('S')]

    plot_flooding_curve(flood_extent_dict, Hmean, outfigPath)

    input("Press Enter to close all figures...")  # Wait for user input
    plt.close('all')  # Close all figures after pressing Enter


#--------------------------------------------------------
#
# Execute code
#
#--------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process flood simulation data.")
    parser.add_argument("--city", type=str, required=True, help="City name for data processing.")
    args = parser.parse_args()

    # Assign parsed arguments
    city = args.city
    # Run main function
    main(city)