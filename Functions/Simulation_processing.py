#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------

import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import h5py
import random

#--------------------------------------------------------
#
# FUNCTIONS
#
#--------------------------------------------------------

def concatenate_transects(transect_data):
    """
    Concatenates multiple transect dataframes into a single dataframe.

    Parameters:
    - transect_data (dict): A dictionary where keys are transect IDs and values are pandas DataFrames 
                            containing columns ['Lon', 'Lat', 'Elevation'] for each transect.

    Returns:
    - pd.DataFrame: A single concatenated dataframe containing all transects, with an additional 
                    'transect_id' column to retain the original transect information.
    """
    transect_points = []
    for transect_id, transect_df in transect_data.items():
        transect_df['transect_id'] = transect_id  # Keep track of transect IDs
        transect_points.append(transect_df[['Lon', 'Lat', 'Elevation', 'transect_id']])

    return pd.concat(transect_points, ignore_index=True)

#--------------------------------------------------------
#--------------------------------------------------------

def interpolate_simulations_to_xarray(flood_maps, all_transect_points):
    """
    Interpolates transect points over all simulations from multiple flood maps and stores the results
    in a structured xarray.Dataset. Each simulation is traced back to its source file for efficient analysis.
    
    Additionally, replaces trailing zeros in hmax values with NaN.

    Parameters:
    - flood_maps (dict): A dictionary where keys are file names and values are xarray datasets
                         containing flood maps with 'hmax' data (dimensions: simulation, lon, lat).
    - all_transect_points (pd.DataFrame): A DataFrame containing all transect points with the following columns:
                                          ['Lon', 'Lat', 'Elevation', 'transect_id'].

    Returns:
    - xarray.Dataset: An xarray dataset with interpolated values and trailing zeros replaced with NaN.
    """
    # Initialize storage for interpolated results
    total_simulations = sum(fm.sizes['simulation'] for fm in flood_maps.values())
    result_hmax = np.full((total_simulations, len(all_transect_points)), np.nan)  # (simulation, points)
    source_files = []  # List to store the source of each simulation

    # Interpolate over all simulations in each flood map
    sim_offset = 0  # Track where the simulations from each file start
    for file_name, flood_map in tqdm(flood_maps.items(), desc="Processing Flood Maps"):
        n_simulations = flood_map.sizes['simulation']
        
        # Interpolate all points for this flood map
        interpolated = flood_map['hmax'].interp(
            lon=xr.DataArray(all_transect_points['Lon'], dims='points'),
            lat=xr.DataArray(all_transect_points['Lat'], dims='points')
        )  # Shape: (simulation, points)
        
        # Convert trailing zeros to NaN
        interpolated_values = interpolated.values
        interpolated_values[interpolated_values == 0] = np.nan

        # Store the results in the result array
        result_hmax[sim_offset:sim_offset + n_simulations, :] = interpolated_values
        
        # Append simulation traceability information
        source_files.extend([file_name] * n_simulations)
        
        sim_offset += n_simulations  # Update the offset

    # Convert source_files list into an array for storage in NetCDF
    source_files = np.array(source_files, dtype="object")

    # Create an xarray.Dataset to store results with traceability
    ds = xr.Dataset(
        {
            'hmax': (['simulation', 'point'], result_hmax),
            'lat': (['point'], all_transect_points['Lat']),
            'lon': (['point'], all_transect_points['Lon']),
            'elevation': (['point'], all_transect_points['Elevation']),
            'transect_id': (['point'], all_transect_points['transect_id']),
            'source_file': (['simulation'], source_files),  # Store the source file for each simulation
        },
        coords={
            'simulation': np.arange(total_simulations),
            'point': np.arange(len(all_transect_points)),
        }
    )

    return ds

#--------------------------------------------------------
#--------------------------------------------------------

def compute_mean_hmax_at_shoreline(results_dict):
    """
    Compute the mean hmax at the shoreline (first value of each transect) for each scenario across all transects
    for each source file.

    Parameters:
    - results_dict (dict): Dictionary where keys are source file names, and values are dictionaries of simulation-transect data.

    Returns:
    - dict: A dictionary where keys are source file names, and values are dictionaries with scenario IDs as keys
            and their mean hmax at the shoreline as values.
    """
    # Initialize a dictionary to store mean hmax for each scenario
    scenario_hmax = {}

    # Process each source file
    for _, result_dict in tqdm(results_dict.items(), desc="Processing source files"):
        
        # Group keys by scenario (e.g., S001, S002, ...)
        for key, df in result_dict.items():
            scenario_id = key.split('_')[0]  # Extract scenario (e.g., S001)
            if not df.empty: 
                # Add hmax at the shoreline to the scenario's list
                scenario_hmax.setdefault(scenario_id, []).append(df['hmax'].iloc[0])

        # Compute the mean hmax for each scenario
        scenario_hmax = {
            scenario: np.nanmean(hmax_values) for scenario, hmax_values in scenario_hmax.items()
        }

    return scenario_hmax

#--------------------------------------------------------
#--------------------------------------------------------

def select_scenarios_weighted_kmeans(scenario_hmax, n_clusters=100):
    """
    Use weighted KMeans clustering to select representative scenarios with MinMax scaling.

    Parameters:
    - scenario_hmax (dict): Dictionary where keys are scenario IDs and values are hmax means.
    - n_clusters (int): Number of clusters.

    Returns:
    - list: Selected scenario IDs.
    """
    df = pd.DataFrame.from_dict(scenario_hmax, orient='index', columns=['hmax'])
    df['scenario'] = df.index

    # Normalize hmax values using MinMaxScaler
    scaler = MinMaxScaler()
    df['hmax_scaled'] = scaler.fit_transform(df[['hmax']])

    # Compute weights: Inverse frequency of hmax values (handle duplicates properly)
    df['weight'] = 1 / df.groupby('hmax_scaled')['hmax_scaled'].transform('count')

    # Apply KMeans with sample weights
    kmeans = KMeans(n_clusters=n_clusters, random_state=44, n_init=20)
    df['cluster'] = kmeans.fit_predict(df[['hmax_scaled']], sample_weight=df['weight'])

    # Select closest to cluster center
    selected_scenarios = []
    for cluster_id in tqdm(range(n_clusters), desc="Selecting representatives"):
        cluster_data = df[df['cluster'] == cluster_id].copy()
        cluster_centroid = kmeans.cluster_centers_[cluster_id][0]

        # Select scenario closest to centroid
        representative_scenario = cluster_data.loc[(cluster_data['hmax_scaled'] - cluster_centroid).abs().idxmin(), 'scenario']
        selected_scenarios.append(representative_scenario)

    return selected_scenarios

#--------------------------------------------------------
#--------------------------------------------------------

def extract_selected_scenarios_to_h5(ds, selected_scenarios, output_h5, n_jobs=-1):
    """
    Extract selected simulations and transects from an xarray dataset into an HDF5 file in parallel,
    and return the selected scenarios dictionary.

    Parameters:
    - ds (xarray.Dataset): The dataset containing all scenarios.
    - selected_scenarios (list): List of selected scenario IDs (e.g., ['S2570', 'S1183']).
    - output_h5 (str): Output HDF5 filename.
    - n_jobs (int): Number of parallel jobs (-1 uses all available CPUs).

    Returns:
    - dict: A dictionary where keys are scenario-transect IDs (e.g., 'S2570_T001') and values are DataFrames.
    """

    def process_scenario(ds, scenario_id):
        """Extracts transect data for a given scenario."""
        scenario_results = {}
        scenario_index = int(scenario_id[1:])  # Extract numeric ID from 'Sxxxx'
        
        # Extract data for the given scenario
        scenario_data = ds.sel(simulation=scenario_index)

        # Get unique transect IDs
        unique_transects = np.unique(scenario_data['transect_id'].values)

        for transect_id in unique_transects:
            transect_mask = scenario_data['transect_id'].values == transect_id
            if not np.any(transect_mask):
                continue

            # Extract hmax properly based on its dimensionality
            hmax_values = scenario_data['hmax'].values
            if hmax_values.ndim == 2:
                hmax_values = hmax_values[:, transect_mask].flatten()
            elif hmax_values.ndim == 1:
                hmax_values = hmax_values[transect_mask]  # Keep as 1D

            df = pd.DataFrame({
                'lat': scenario_data['lat'].values[transect_mask],
                'lon': scenario_data['lon'].values[transect_mask],
                'elevation': scenario_data['elevation'].values[transect_mask],
                'hmax': hmax_values,
            })
            # Wrap df in a dictionary, apply the function, and extract the modified DataFrame
            df = fill_initial_nans({'temp': df}, 'hmax')['temp']
            
            key = f"{scenario_id}_{transect_id}"
            scenario_results[key] = df

        return scenario_results

    # Run parallel processing for all selected scenarios
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_scenario)(ds, scenario_id) for scenario_id in selected_scenarios
    )

    # Merge results from parallel processing
    results_dict = {k: v for res in results_list for k, v in res.items()}

    # Save to HDF5
    with pd.HDFStore(output_h5, mode="w", complevel=9, complib="blosc") as store:
        for key, df in tqdm(results_dict.items(), desc="Saving to HDF5"):
            # Convert only float columns to float32
            float_cols = df.select_dtypes(include=["float"]).columns
            df[float_cols] = df[float_cols].astype("float32")

            store.put(key, df, format="table", data_columns=True)
            
    print(f"\nSelected scenarios saved to {output_h5}")

    return results_dict  # Return the extracted dictionary

#--------------------------------------------------------
#--------------------------------------------------------

def compute_mean_hmax_at_shoreline(ds):
    """
    Compute the mean hmax at the shoreline (first value of each transect) for each scenario 
    across all transects directly from the xarray dataset.

    Parameters:
    - ds (xarray.Dataset): Dataset containing 'hmax', 'transect_id', and 'simulation'.

    Returns:
    - dict: A dictionary with scenario IDs as keys and their mean hmax at the shoreline as values.
    """

    # Ensure required variables exist
    if 'hmax' not in ds or 'transect_id' not in ds:
        raise KeyError("Dataset must contain 'hmax' and 'transect_id' variables.")

    # Convert transect_id to a categorical index for grouping
    unique_transects, transect_indices = np.unique(ds['transect_id'], return_inverse=True)

    # Select first occurrence of each unique transect per simulation
    first_transect_indices = np.array([np.where(transect_indices == i)[0][0] for i in range(len(unique_transects))])

    # Extract first hmax values for each transect
    shoreline_hmax = ds['hmax'].isel(point=first_transect_indices)

    # Compute mean hmax at shoreline across all transects for each simulation
    mean_hmax_per_scenario = shoreline_hmax.mean(dim='point', skipna=True)

    # Convert to dictionary format {scenario_id: mean_hmax}
    scenario_hmax_dict = {f"S{int(sim_id):04d}": float(mean_hmax)
                          for sim_id, mean_hmax in zip(ds['simulation'].values, mean_hmax_per_scenario.values)}

    return scenario_hmax_dict

#--------------------------------------------------------
#--------------------------------------------------------

def plot_scenario_selection(scenario_hmax, selected_scenarios):
    """
    Generates two subplots to visualize the distribution of hmax values in all scenarios
    and the selected representative scenarios.

    Parameters:
    - scenario_hmax (dict): Dictionary where keys are scenario IDs and values are hmax means.
    - selected_scenarios (list): List of selected scenario IDs for representation.

    Returns:
    - None (Displays the plots)
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(scenario_hmax, orient='index', columns=['hmax'])
    df['scenario'] = df.index
    df['selected'] = df['scenario'].isin(selected_scenarios)

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of hmax (all vs selected)
    sns.histplot(df['hmax'], bins=30, kde=True, stat='density', label="All Scenarios", color='blue', alpha=0.6, ax=axes[0])
    sns.histplot(df[df['selected']]['hmax'], bins=30, kde=True, stat='density', label="Selected", color='orange', alpha=0.8, ax=axes[0])
    axes[0].set_title("Histogram of hmax: All vs Selected scenarios")
    axes[0].set_xlabel("hmax [m]")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Scatter plot of all scenarios with selected ones highlighted
    axes[1].scatter(range(len(df)), df['hmax'], label="All Scenarios", alpha=0.4, color='blue')
    axes[1].scatter(
        [df.index.get_loc(s) for s in df[df['selected']].index], 
        df[df['selected']]['hmax'], 
        label="Selected", color='orange', edgecolor='black', s=50
    )
    axes[1].set_title("Scatter plot of scenarios with selected representations")
    axes[1].set_xlabel("N° of Scenario")  # Numeric index reference
    axes[1].set_ylabel("hmax [m]")

    # Dynamically adjust x-ticks based on max_index
    max_index = len(df) - 1
    
    # Determine tick interval dynamically (rounding to nearest multiple of 50 or 100)
    tick_interval = max(50, round(max_index / 10, -2))  # Ensures reasonable spacing
    
    axes[1].set_xticks(np.arange(0, max_index + 1, tick_interval))
    axes[1].set_xticklabels([str(int(tick)) for tick in np.arange(0, max_index + 1, tick_interval)]) 
    
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

#--------------------------------------------------------
#--------------------------------------------------------

def plot_random_transects(scenario_dict, num_plots=20):
    """
    Selects 20 equidistant transects from the scenario dictionary and plots their elevation and flood statistics.

    Parameters:
    - scenario_dict (dict): Dictionary where keys are scenario-transect IDs (e.g., 'S1561_T075'),
      and values are DataFrames with columns ['lat', 'lon', 'elevation', 'hmax'].
    - num_plots (int, optional): Number of equidistant transects to plot (default: 20).
    """

    # Ensure at least 20 transects are available
    num_available = len(scenario_dict)
    num_plots = min(num_plots, num_available)

    # Extract unique transects (T001, T002, etc.) ignoring scenario IDs (SXXXX)
    unique_transects = sorted(set([key.split('_')[1] for key in scenario_dict.keys()]))

    # Select equidistant transects
    indices = np.linspace(0, len(unique_transects) - 1, num_plots, dtype=int)
    selected_transects = [unique_transects[i] for i in indices]

    # Generate formatted names based on actual transect IDs (instead of hardcoding)
    transect_labels = [f"T{int(transect_id[1:]):03d}" for transect_id in selected_transects]

    # Create a 5x4 subplot layout
    fig, axes = plt.subplots(4, 5, figsize=(12, 7))
    axes = axes.flatten()  # Flatten to easily iterate over 20 plots

    for i, (transect_id, transect_label) in enumerate(zip(selected_transects, transect_labels)):
        ax = axes[i]

        # Extract all scenarios corresponding to the selected transect
        transect_scenarios = [df for key, df in scenario_dict.items() if key.endswith(transect_id)]

        # Merge data into a single DataFrame for all scenarios
        merged_df = pd.concat(transect_scenarios, axis=0)

        # Group by distance index (assuming each transect follows the same sampling points)
        grouped = merged_df.groupby(merged_df.index)

        # Compute mean & std of hmax over all scenarios at each distance
        hmax_mean = grouped['hmax'].apply(lambda x: x.dropna().mean())  # Only use valid values
        hmax_std = grouped['hmax'].apply(lambda x: x.dropna().std())  # Ensure std is computed correctly
        elevation_mean = grouped['elevation'].mean()  # Compute mean elevation if elevations vary slightly

        # Define total water level based on mean hmax
        mean_flood_level = elevation_mean + hmax_mean

        # Clamp lower flood bound so it never goes below elevation
        lower_bound = np.maximum(mean_flood_level - hmax_std, elevation_mean)
        upper_bound = mean_flood_level + hmax_std

        # Plot elevation (mean across simulations)
        ax.plot(elevation_mean, color='black', linewidth=1.5)

        # Plot mean flood level
        ax.plot(mean_flood_level, color='blue', linewidth=1.5)

        # Fill between (clamped mean - std) and (mean + std)
        ax.fill_between(range(len(elevation_mean)), 
                lower_bound.fillna(mean_flood_level),  # Avoid gaps in shading
                upper_bound.fillna(mean_flood_level), 
                color='blue', alpha=0.3)

        # Place the transect name inside each plot
        ax.text(0.05, 0.85, transect_label, transform=ax.transAxes, fontsize=12, 
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
        
        valid_length = len(mean_flood_level.dropna())  # Get length without NaN values
        ax.set_xlim(0, valid_length * 1.1)  # Extend by 10%

        # Set y-label only for the first column
        if i % 5 == 0:
            ax.set_ylabel("Elevation [m]")
        # Set x-label only for the last row
        if i >= 15:
            ax.set_xlabel("Cumulative Distance [m]")

    # Create an external legend at (x=0.5, y=1.05)
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Elevation'),
        plt.Line2D([0], [0], color='blue', linewidth=1.5, label='Mean Flood Level'),
        plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.3, label='Flood Std')
    ]
    plt.tight_layout()  # Ensure plots are correctly arranged **before adding the legend**
    fig.subplots_adjust(top=0.93)  # Ensures enough space for the legend
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fontsize=12, frameon=True)
    plt.show()
    
#-------------------------------------------------------
#-------------------------------------------------------

def fill_initial_nans(datadict, column):
    """
    Fills NaN values in the specified column from the beginning of the DataFrame until the first non-NaN value 
    with the first valid (non-NaN) value.

    Args:
    df (pd.DataFrame): The DataFrame where NaN values need to be filled.
    column (str): The column name where the NaN values will be filled.
    
    Returns:
    pd.DataFrame: The DataFrame with the NaN values filled in the specified column.
    """
    for key, df in datadict.items():
        # Find the first valid (non-NaN) value index
        first_valid_index = df[column].first_valid_index()

        if first_valid_index is not None:  # If there is a non-NaN value in the DataFrame
            # Get the first valid value
            first_valid_value = df[column].iloc[first_valid_index]
            
            # Fill NaN values from the start to the first non-NaN value using df.loc to avoid chained assignment
            df.loc[:first_valid_index, column] = first_valid_value
        
        datadict[key] = df
    return datadict
