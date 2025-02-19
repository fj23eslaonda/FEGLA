#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon as MplPolygon
from sklearn.cluster import KMeans

#--------------------------------------------------------
#--------------------------------------------------------
def load_height_iterations(main_path, file_pattern, name_extraction_fn):
    """
    Loads only `height_iteration[-1]` from multiple PKL files into a structured dictionary.

    Parameters:
    - city (str): Name of the city (used to construct file paths).
    - file_pattern (str): Pattern to match files (e.g., '*cte*', '*squared*', '*linear*').
    - name_extraction_fn (function): Function to extract keys for dictionary storage.

    Returns:
    - dict: A dictionary structured as {F0_X: {Transect: height_iteration[-1]}}.
    """
    results_list = list(main_path.glob(file_pattern))  # Get matching files
    results_dict = {}

    for results in tqdm(results_list, desc=f"Loading {file_pattern} files"):
        name_key = name_extraction_fn(str(results))  # Extract the correct key format

        with open(results, 'rb') as f:
            data = pickle.load(f)  # Load PKL data

        # Store only height_iteration[-1] for each transect
        results_dict[name_key] = {
            key: value['height_iteration'][-1] if 'height_iteration' in value else None
            for key, value in data.items()
        }

        # Free memory
        del data  

    print(f"Loaded {len(results_dict)} simulations for pattern {file_pattern}\n")
    return results_dict

#--------------------------------------------------------
#--------------------------------------------------------
# Function to extract names for CTE and SQUARED
def extract_cte_squared_name(filepath):
    return f'F0_{filepath.split("_")[-2]}'

# Function to extract names for LINEAR (handles F0 and FR separately)
def extract_linear_name(filepath):
    F0name = filepath.split("_")[-4]
    FRname = filepath.split("_")[-2]
    return f'F0_{F0name}_FR_{FRname}'

#--------------------------------------------------------
#--------------------------------------------------------

def compute_error_for_extension(results_dict, scen_tran_all):
    """
    Computes the mean error in flood extension for different scenarios based on the results and simulated data.

    Parameters:
    - results_dict (dict): Dictionary structured as {F0_X: {Scenario: {Transect: height_iteration[-1]}}}.
    - scen_tran_all (dict): Dictionary structured as {Scenario: {Transect: hmax values}}.
    - max_scen (int): The maximum number of scenarios to compute errors for. Default is 50.

    Returns:
    - error_extension (dict): Dictionary structured as {F0_X: {"extent_error": [mean_scenario_errors]}}.
    """

    error_extension = {}

    for f0_key, dictionary in tqdm(results_dict.items(), desc="Computing errors for extensions"):
        scen_list = sorted(set(key.split('_')[0].strip('/') for key in dictionary.keys()))
        mean_scenario_errors = []

        for scen in scen_list:
            # Extract data for this scenario
            results_scen = {k: v for k, v in dictionary.items() if k.startswith(scen)}
            data_scen = {k: v for k, v in scen_tran_all.items() if k.startswith(scen)}

            error_tran = []

            for transect_key in results_scen.keys():
                if transect_key not in data_scen:
                    continue  # Skip missing transects

                h_pred = results_scen[transect_key] # Predicted height
                h_sim = data_scen[transect_key]  # Simulated height

                if len(h_pred) == 0 or len(h_sim) == 0:
                    continue  # Skip empty values

                try:
                    # Compute percentage flood extent error
                    extent_error = np.abs(len(h_pred) - len(h_sim)) / len(h_sim) * 100
                    
                    error_tran.append(extent_error)
                except Exception:
                    pass  # Ignore errors

            if error_tran:
                mean_scenario_errors.append(np.mean(error_tran))  # Compute mean error for the scenario

        if mean_scenario_errors:
            error_extension[f0_key] =  mean_scenario_errors  # Store all mean errors per F0
        
    return error_extension

#--------------------------------------------------------
#--------------------------------------------------------

def plot_1Derror_distribution(error_data, outfigPath, froude, figsize=(10, 4), palette="Spectral", line_color="m"):
    """
    Plots the error distribution as a violin plot with a line connecting the mean values.
    
    Parameters:
    - error_data (dict): Dictionary containing error data, where keys are F0 values (e.g., "F0_0.1") 
                         and values are dictionaries with error lists (e.g., {"extent_error": [...]})
    - error_type (str): The type of error to plot ("extent_error", "rmse", "mae").
    - figsize (tuple): Figure size in inches (width, height). Default is (10, 4).
    - palette (str): Color palette for the violin plot. Default is "Spectral".
    - line_color (str): Color of the line connecting the mean values. Default is "m" (magenta).
    
    Returns:
    - None (displays the plot).
    """

    # Sort the F0 keys numerically
    sorted_keys = sorted(error_data.keys(), key=lambda x: float(x.split("_")[1]))

    # Extract sorted error lists
    sorted_error_lists = [error_data[key] for key in sorted_keys]
    means_sorted = [np.mean(errors) for errors in sorted_error_lists]

    # Modify the x-axis labels
    numeric_labels = [key.split("_")[1] for key in sorted_keys]

    # Identify the minimum error and corresponding F0
    min_error_index = np.argmin(means_sorted)
    min_error_value = means_sorted[min_error_index]
    min_error_f0 = numeric_labels[min_error_index]

    # Plot the sorted data
    plt.figure(figsize=figsize)

    # Create violin plot with sorted data
    sns.violinplot(data=sorted_error_lists, palette=palette)

    # Add line connecting the sorted means
    plt.plot(range(len(sorted_keys)), means_sorted, marker='o', linestyle='-', color=line_color, label="Mean")

    # Customize the plot
    plt.title(f"F0 = {min_error_f0} with min error = {min_error_value:.2f}%", fontsize=16)
    plt.ylabel("Error Value %", fontsize=14)
    plt.xticks(ticks=range(len(sorted_keys)), labels=numeric_labels, ha="right", rotation=45)
    plt.xlabel("F0", fontsize=14)
    plt.legend()
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfigPath / f"Error_distribution_{froude}.png", dpi=300, bbox_inches='tight', format='png')
    # Display the plot
    plt.show(block=False)
    plt.pause(0.1)

    return min_error_f0, min_error_value

#--------------------------------------------------------
#--------------------------------------------------------

def plot_2Derror_distribution(error_data, outfigPath, cmap="Blues"):
    """
    Plots a contour map of errors as a function of Froude numbers (F0 and FR), including level curves at 10, 20, 30, etc.
    
    Parameters:
    - error_data (dict): Dictionary where keys are strings in the format "F0_x_FR_y" 
                         and values are dictionaries containing "extent_error" lists.
    - cmap (str): Colormap for the contour plot. Default is "Blues".
    
    Returns:
    - None
    """

    # Extract F0, FR, and mean extent error values from the dictionary
    data = []
    for key, value in error_data.items():
        parts = key.split('_')
        F0 = float(parts[1])  # Extract F0 from the key
        FR = float(parts[3])  # Extract FR from the key
        
        mean_value = np.mean(value)  # Compute the mean error
        data.append((F0, FR, mean_value))
    
    # Convert to a NumPy array for easier manipulation
    data = np.array(data)
    if data.size == 0:
        print("No valid error data to plot.")
        return
    
    F0_values = data[:, 0]
    FR_values = data[:, 1]
    error_values = data[:, 2]

    # Create unique arrays for F0 and FR
    F0_unique = np.unique(F0_values)
    FR_unique = np.unique(FR_values)

    # Create a grid for F0 and FR
    F0_grid, FR_grid = np.meshgrid(F0_unique, FR_unique)

    # Create a grid for the error values, initializing with NaN
    error_grid = np.full_like(F0_grid, np.nan, dtype=float)

    # Fill the error grid based on the provided data
    for F0, FR, error in zip(F0_values, FR_values, error_values):
        i = np.where(FR_unique == FR)[0][0]
        j = np.where(F0_unique == F0)[0][0]
        error_grid[i, j] = error

    # Automate the levels based on fixed intervals (10, 20, 30, ...)
    min_error = np.nanmin(error_grid)
    max_error = np.nanmax(error_grid)
    levels = np.arange(10, max_error + 10, 10)  # Custom contour levels at fixed intervals

    # Find the minimum error and its corresponding F0 and FR
    min_index = np.unravel_index(np.nanargmin(error_grid), error_grid.shape)
    min_F0 = F0_grid[min_index]
    min_FR = FR_grid[min_index]

    # Plot the filled contour map
    plt.figure(figsize=(8, 5))
    contourf = plt.contourf(F0_grid, FR_grid, error_grid, levels=levels, cmap=cmap)
    cbar = plt.colorbar(contourf, ticks=levels)  # Show only selected levels
    cbar.set_label('Error % (EA)', fontsize=12)

    # Add contour lines at 10, 20, 30, etc.
    contour = plt.contour(F0_grid, FR_grid, error_grid, levels=levels, colors='black', linewidths=0.5)
    plt.clabel(contour, fmt='%1.1f', fontsize=8)

    # Highlight the minimum error point
    plt.scatter(min_F0, min_FR, color='red', marker='x', s=100, label=f"Min Error (EA = {min_error:.2f}%)")

    # Customize the plot
    plt.xlabel("Froude number on the Coast (F0)", fontsize=12)
    plt.ylabel("Froude number on the Runup (FR)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(F0_unique.min(), F0_unique.max() + 0.1, 0.1))
    plt.yticks(np.arange(FR_unique.min(), FR_unique.max() + 0.1, 0.1))
    plt.title(f'F0 = {min_F0} and FR = {min_FR} with mean error {min_error:.2f}%')
    plt.tight_layout()
    plt.savefig(outfigPath / "Error_distribution_linear.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.1)

    return min_F0, min_FR, min_error

#--------------------------------------------------------
#--------------------------------------------------------

def plot_best_fit_models(error_resultsCTE, error_resultsSquared, error_resultslinear, 
                         resultsCte_dict, min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear):
    """
    Plots the error analysis for the best-fit models using:
    1. A line plot comparing error evolution across simulations.
    2. A boxplot summarizing error distributions for each model.

    Parameters:
    - error_resultsCTE (dict): Dictionary containing errors for constant Froude models.
    - error_resultsSquared (dict): Dictionary containing errors for squared Froude models.
    - error_resultslinear (dict): Dictionary containing errors for linear Froude models.
    - resultsCte_dict (dict): Dictionary containing simulation results for constant Froude models.
    - min_F0_cte (str): Best-fit Froude number for the constant model.
    - min_F0_squared (str): Best-fit Froude number for the squared model.
    - min_F0_linear (str): Best-fit Froude number for the linear model.
    - min_FR_linear (str): Best-fit FR value for the linear model.

    Returns:
    - None (displays the plot).
    """

    # Create figure with two subplots (Line plot + Boxplot)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]})

    # Generate keys for the best-fit models
    min_CteKey = f'F0_{min_F0_cte}'
    min_SquaredKey = f'F0_{min_F0_squared}'
    min_LinearKey = f'F0_{min_F0_linear}_FR_{min_FR_linear}'

    # Extract all unique scenarios from dictionary keys
    Allscenarios = sorted(set(key.split('_')[0].strip('/') for key in resultsCte_dict[min_CteKey].keys()))

    # Extract the best-fit model errors
    errorCte_opti     = error_resultsCTE[min_CteKey]
    errorSquared_opti = error_resultsSquared[min_SquaredKey]
    errorLinear_opti  = error_resultslinear[min_LinearKey]

    # Extract best-fit Froude numbers
    cteKey     = min_CteKey.split('_')[1]
    SquaredKey = min_SquaredKey.split('_')[1]
    LinearKey  = min_LinearKey.split('_')[1], min_LinearKey.split('_')[3]

    # Format best-fit model names for legends
    legend_cte     = f'Cte F0 = {cteKey}'
    legend_squared = f'Squared F0 = {SquaredKey}'
    legend_linear  = f'Linear F0={LinearKey[0]} and FR={LinearKey[1]}'

    # Compute mean and standard deviation for squared and linear models
    mean_squared = np.mean(np.abs(errorSquared_opti))
    std_squared = np.std(np.abs(errorSquared_opti))
    mean_linear = np.mean(np.abs(errorLinear_opti))
    std_linear = np.std(np.abs(errorLinear_opti))

    # Define consistent colors for the plots
    colors = ['#1f77b4', '#ff7f0e', 'm']

    # === Line Plot (Evolution of EA % across simulations) ===
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorCte_opti, marker='x', ls='--', lw=0.5, 
               label=legend_cte, color=colors[0])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorSquared_opti, marker='o', ls='--', lw=0.5, 
               label=legend_squared, color=colors[1])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorLinear_opti, marker='v', ls='--', lw=0.5, 
               label=legend_linear, color=colors[2])
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_xlabel("Simulations", fontsize=12)
    ax[0].set_ylabel("EA %", fontsize=12)
    ax[0].set_xticks(np.arange(2, len(Allscenarios) + 1, 4))  # Set ticks every 4 simulations
    ax[0].set_xlim(0, len(Allscenarios) + 0.5)

    # === Boxplot (Distribution of Errors) ===
    # Create a DataFrame for plotting
    data = pd.DataFrame({
        'Error': list(np.abs(errorCte_opti)) + list(np.abs(errorSquared_opti)) + list(np.abs(errorLinear_opti)),
        'Category': ['Constant'] * len(errorCte_opti) + 
                    ['Squared'] * len(errorSquared_opti) + 
                    ['Linear'] * len(errorLinear_opti)
    })

    # Plot boxplot with category colors
    sns.boxplot(
        x='Category', 
        y='Error', 
        data=data, 
        ax=ax[1], 
        hue='Category',  # Assign hue to match categories
        palette=colors,
        legend=False
    )
    ax[1].set_ylabel("EA(%)", fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].set_xlabel('Froude Number')

    # Add statistics text to the boxplot
    stats_text = (
        f"Squared:  Mean: {mean_squared:.2f}  Std: {std_squared:.2f}\n"
        f"Linear:  Mean: {mean_linear:.2f}  Std: {std_linear:.2f}"
    )
    ax[1].text(0.25, 0.9, stats_text, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    
#--------------------------------------------------------
#--------------------------------------------------------

def integrate_hmax_predictions(transectData, resultsCte_dict, resultsSquared_dict, resultsLinear_dict,
                               min_F0_cte, min_error_cte, 
                               min_F0_squared, min_error_squared, 
                               min_F0_linear, min_FR_linear, min_error_linear):
    """
    Integrates only the hmax predictions corresponding to the model with the minimum error into the SWE simulation dataset.

    Parameters:
    - transectData (dict): Dictionary containing DataFrames of SWE simulations, with keys as transect IDs.
    - resultsCte_dict (dict): Dictionary containing hmax predictions from the constant Froude model.
    - resultsSquared_dict (dict): Dictionary containing hmax predictions from the squared Froude model.
    - resultsLinear_dict (dict): Dictionary containing hmax predictions from the linear Froude model.
    - min_F0_cte (str): Best-fit Froude number for the constant model.
    - min_error_cte (float): Minimum error for the constant model.
    - min_F0_squared (str): Best-fit Froude number for the squared model.
    - min_error_squared (float): Minimum error for the squared model.
    - min_F0_linear (str): Best-fit Froude number for the linear model.
    - min_FR_linear (str): Best-fit FR value for the linear model.
    - min_error_linear (float): Minimum error for the linear model.

    Returns:
    - updated_transectData (dict): A new dictionary with an integrated `hmax_best` column containing the best hmax predictions.
    """

    # Determine which model has the lowest error
    error_mapping = {
        min_error_cte: ('cte', resultsCte_dict.get(f'F0_{min_F0_cte}', {})),
        min_error_squared: ('squared', resultsSquared_dict.get(f'F0_{min_F0_squared}', {})),
        min_error_linear: ('linear', resultsLinear_dict.get(f'F0_{min_F0_linear}_FR_{min_FR_linear}', {}))
    }

    # Identify the best-fit model
    min_error_value = min(error_mapping.keys())
    best_model_name, best_model_data = error_mapping[min_error_value]

    # Print which model was chosen
    print(f"\nIntegrating hmax from the best-fit model: {best_model_name} (Error: {min_error_value:.2f})")

    # Create a copy of transectData to store the updates
    updated_transectData = {}

    # Iterate over each transect in transectData
    for transect_id, df in transectData.items():
        # Copy the DataFrame to avoid modifying the original
        df = df.copy()

        # Extract the best-fit hmax prediction, if available
        hmax_best = best_model_data.get(transect_id, [])

        # Ensure the prediction array matches the length of SWE simulation data
        df['hmax_best'] = np.pad(hmax_best, (0, max(0, len(df) - len(hmax_best))), constant_values=np.nan)[:len(df)]

        # Store the updated DataFrame
        updated_transectData[transect_id] = df

    return updated_transectData

#--------------------------------------------------------
#--------------------------------------------------------

def calculate_polygon_and_area(first_points, last_points):
    """
    This function takes first_points and last_points arrays, creates a polygon by connecting
    the points, calculates the area, and stores both in a dictionary.

    Args:
    first_points (np.array): Array of first points (lon, lat) for each transect.
    last_points (np.array): Array of last points (lon, lat) for each transect.

    Returns:
    polygon (dict)
    area (dict)
    """
    # Combine the first and last points to create the polygon
    points = np.vstack([first_points, last_points[::-1]])
    # Create the polygon
    polygon = Polygon(points)

    # Create a GeoDataFrame from the polygon
    gdf = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[polygon])
    # Convert to a metric projection, for example UTM zone 19S
    gdf_utm = gdf.to_crs(epsg=32719)
    # Calculate the area in square meters
    area_m2 = gdf_utm.area[0]

    return polygon, area_m2

#--------------------------------------------------------
#--------------------------------------------------------
def get_boundary_points(data, column):
    """
    Extracts the boundary points (first and last) for each transect in the provided data,
    calculates a polygon that connects these boundary points, and computes the polygon's area.

    Parameters:
    - data (dict): A dictionary where keys are transect IDs, and values are DataFrames containing
      'lon', 'lat', and a specified column (e.g., 'hmax', 'hmax_cte', etc.).
    - column (str): The column name representing water height (e.g., 'hmax', 'hmax_cte').

    Returns:
    - polygonSim (shapely.geometry.Polygon): The polygon created by connecting the boundary points.
    - aream2Sim (float): The area of the polygon in square meters.
    """

    firstptoList = []
    lastptoList = []

    for key, df in data.items():
        # Drop NaN values for the given column
        df = df.dropna(subset=[column])
        
        # Ensure there's data left
        if df.empty or len(df) < 2:
            #print(f"Skipping {key}: Not enough valid points in {column}.")
            continue  # Skip this transect

        try:
            # Extract first and last valid points
            firstptoList.append((df['lon'].values[0], df['lat'].values[0]))  # First point
            lastptoList.append((df['lon'].values[-1], df['lat'].values[-1]))  # Last point
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue  # Skip if there's an unexpected issue

    if len(firstptoList) == 0 or len(lastptoList) == 0:
        print(f"Warning: No valid points found for column {column}.")
        return None, None  # Return None if no valid data is found

    # Convert lists to NumPy arrays
    firstptoArray = np.array(firstptoList)
    lastptoArray = np.array(lastptoList)

    # Create a polygon and compute the area
    polygonSim, aream2Sim = calculate_polygon_and_area(firstptoArray, lastptoArray)

    return polygonSim, aream2Sim

#--------------------------------------------------------
#--------------------------------------------------------
def plot_topobathymetry_and_contours(x, y, z, elev_min=-90, elev_max=240, elev_delta=30, z0_contour=None, cmap='viridis', ax=None, alpha=1, show_colorbar=False):
    """
    Plot a predefined elevation map with contours and a color bar. A contour for z=0 can also be plotted if provided.

    Parameters:
    - x, y (np.array): 2D arrays of longitude and latitude values for each grid point.
    - z (np.array): 2D array of elevation data at the grid points defined by x and y.
    - elev_min (int): Minimum elevation value for contour levels.
    - elev_max (int): Maximum elevation value for contour levels.
    - elev_delta (int): Interval between contour levels.
    - z0_contour (np.array): Optional. 2D array of x, y coordinates for the z=0 contour line.
    - cmap (str): Colormap for the plot.
    - show_colorbar (bool): If True, display the color bar.

    Returns:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib subplot axis for further customization.
    """
    # Initialize the plot and set its size
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    # Define the contour levels
    contour_levels = np.arange(elev_min, elev_max + elev_delta, elev_delta)

    # Fill the contours with color
    cp = ax.contourf(x, y, z, levels=contour_levels, cmap=cmap, alpha=alpha)

    # Draw contour lines
    cs = ax.contour(x, y, z, levels=contour_levels, colors='black', linestyles='solid', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f m')

    # Show color bar only in the last column
    if show_colorbar:
        cbar = plt.colorbar(cp, ax=ax, label='Elevation (m)', ticks=contour_levels)
        cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(elev_delta))

    # Plot shoreline
    if z0_contour is not None:
        ax.plot(z0_contour[:, 0], z0_contour[:, 1], c='r', label='Shoreline [0 m]')
        #ax.legend()

    return ax

#--------------------------------------------------------
#--------------------------------------------------------
def get_elevation_parameters(bathy):
    """
    Computes the elevation parameters for plotting topobathymetry contours.

    Parameters:
    - bathy (numpy array): Bathymetry data.

    Returns:
    - elev_min (int): Rounded minimum elevation.
    - elev_max (int): Rounded maximum elevation.
    - elev_delta (int): Recommended contour interval.
    """
    elev_min = int(np.floor(np.nanmin(bathy) / 10) * 10)  # Round down to nearest 10
    elev_max = int(np.ceil(np.nanmax(bathy) / 10) * 10)   # Round up to nearest 10

    range_elev = elev_max - elev_min

    # Determine the elevation interval
    if range_elev < 100:
        elev_delta = 10
    elif range_elev < 300:
        elev_delta = 20
    else:
        elev_delta = 30

    return elev_min, elev_max, elev_delta

#--------------------------------------------------------
#--------------------------------------------------------
def compute_flood_extent_areas(updated_transectData):
    """
    Computes the flood extent polygons and areas for all scenarios based on hmax (SWE Simulations) 
    and hmax_best (Best-Fit Model: cte, linear, or squared).
    
    Parameters:
    - updated_transectData (dict): Dictionary containing DataFrames with hmax (SWE) and hmax_best.

    Returns:
    - flood_extent_dict (dict): Dictionary structured as:
      {
          scenario: {
              "swe": (polygon, area),
              "best_fit": (polygon, area)
          }
      }
    """
    
    flood_extent_dict = {}

    # Get all unique scenarios from keys
    unique_scenarios = sorted(set(key.split('_')[0] for key in updated_transectData.keys()))

    for scenario in tqdm(unique_scenarios, desc="Computing flood extent areas"):
        # Extract transects for this scenario
        scenario_transects = {key: value for key, value in updated_transectData.items() if key.startswith(scenario)}

        if not scenario_transects:
            continue

        # Compute boundary polygons and areas
        polygon_swe, area_swe = get_boundary_points(
            {k: v[['lon', 'lat', 'hmax']].dropna() for k, v in scenario_transects.items()},
            column='hmax'
        )
        
        polygon_best, area_best = get_boundary_points(
            {k: v[['lon', 'lat', 'hmax_best']].dropna() for k, v in scenario_transects.items()},
            column='hmax_best'
        )

        # Store results in dictionary
        flood_extent_dict[scenario] = {
            "swe": (polygon_swe, area_swe / 1e6),   # Convert to km²
            "best_fit": (polygon_best, area_best / 1e6)  # Convert to km²
        }

    return flood_extent_dict

#--------------------------------------------------------
#--------------------------------------------------------
def plot_flood_extent_contours(flood_extent_dict, scenario, grid_lon, grid_lat, bathy, shoreline, ax=None, show_legend=False, show_colorbar=False):
    """
    Plots the flood extent boundaries for a given scenario using precomputed flood extent areas.

    Parameters:
    - flood_extent_dict (dict): Dictionary containing polygons and areas for SWE and best-fit models.
    - scenario_index (int): Index of the scenario to plot (used to select from scenario_keys).
    - scenario_keys (list): List of available scenario keys (e.g., ['S0023', 'S0304', ...]).
    - grid_lon, grid_lat (numpy arrays): Longitude and latitude grid for topography.
    - bathy (numpy array): Bathymetry data.
    - shoreline (float): Contour level for shoreline representation.
    - ax (matplotlib axis, optional): If provided, the plot will be drawn on this axis (for subplots).
    - show_legend (bool): If True, display the legend (only for the first subplot).
    - show_colorbar (bool): If True, display the color bar (only for the last column).

    Returns:
    - None (displays a plot).
    """

    
    # Retrieve the precomputed polygons and areas
    polygon_swe, area_swe = flood_extent_dict[scenario]["swe"]
    polygon_best, area_best = flood_extent_dict[scenario]["best_fit"]

    # If no axis is provided, create a new figure
    is_standalone = ax is None
    if is_standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute elevation parameters automatically
    elev_min, elev_max, elev_delta = get_elevation_parameters(bathy)

    # Plot topobathymetry and shoreline
    plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy,
                                     elev_min=elev_min,
                                     elev_max=elev_max,
                                     elev_delta=elev_delta,
                                     z0_contour=shoreline,
                                     cmap='gray',
                                     ax=ax,
                                     alpha=0.8,
                                     show_colorbar=show_colorbar)

    # Plot flood polygons
    ax.add_patch(MplPolygon(np.c_[polygon_swe.exterior.xy[0], polygon_swe.exterior.xy[1]], closed=True, 
                            edgecolor='m', fill=False, linewidth=2, label='SWE Simulation'))
    
    ax.add_patch(MplPolygon(np.c_[polygon_best.exterior.xy[0], polygon_best.exterior.xy[1]], closed=True, 
                            edgecolor='cyan', fill=False, linewidth=2, label='Best-Fit Model'))

    # Set subplot title with computed areas
    ax.set_title(f"Scenario {scenario}\nSWE: {area_swe:.2f} km² | Best-Fit: {area_best:.2f} km²",
                 fontsize=10, pad=12)

    # Add legend only in the first subplot
    if show_legend:
        ax.legend(loc="best", fontsize=8, facecolor='white', framealpha=0.8, edgecolor='black')

    # If it's a standalone figure, show the full plot
    if is_standalone:
        fig.suptitle(f"Flood Extent for Scenario {scenario}", fontsize=14, ha='center', x=0.55)
        plt.tight_layout()
        plt.show()

#--------------------------------------------------------
#--------------------------------------------------------
def plot_best_fit_models(error_resultsCTE, error_resultsSquared, error_resultslinear, 
                         resultsCte_dict, min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear,
                         outfigPath, n_clusters=9):
    """
    Plots the error analysis for the best-fit models using:
    1. A line plot comparing error evolution across simulations.
    2. A boxplot summarizing error distributions for each model.
    3. Selects the most interesting scenarios using K-means clustering.

    Parameters:
    - error_resultsCTE (dict): Dictionary containing errors for constant Froude models.
    - error_resultsSquared (dict): Dictionary containing errors for squared Froude models.
    - error_resultslinear (dict): Dictionary containing errors for linear Froude models.
    - resultsCte_dict (dict): Dictionary containing simulation results for constant Froude models.
    - min_F0_cte (str): Best-fit Froude number for the constant model.
    - min_F0_squared (str): Best-fit Froude number for the squared model.
    - min_F0_linear (str): Best-fit Froude number for the linear model.
    - min_FR_linear (str): Best-fit FR value for the linear model.
    - n_clusters (int): Number of clusters to select the most representative scenarios.

    Returns:
    - selected_scenarios (list): Indices of the most representative scenarios.
    """

    # Generate keys for the best-fit models
    min_CteKey = f'F0_{min_F0_cte}'
    min_SquaredKey = f'F0_{min_F0_squared}'
    min_LinearKey = f'F0_{min_F0_linear}_FR_{min_FR_linear}'

    # Extract all unique scenarios from dictionary keys
    Allscenarios = sorted(set(key.split('_')[0].strip('/') for key in resultsCte_dict[min_CteKey].keys()))

    # Extract the best-fit model errors
    errorCte_opti     = error_resultsCTE[min_CteKey]
    errorSquared_opti = error_resultsSquared[min_SquaredKey]
    errorLinear_opti  = error_resultslinear[min_LinearKey]

    # Stack errors for clustering
    error_matrix = np.vstack([errorCte_opti, errorSquared_opti, errorLinear_opti]).T

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(error_matrix)
    cluster_labels = kmeans.labels_
    selected_scenarios = []
    
    # Select one scenario per cluster (closest to cluster center)
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(error_matrix[cluster_indices] - cluster_center, axis=1)
        best_index = cluster_indices[np.argmin(distances)]
        selected_scenarios.append(Allscenarios[best_index])

    # Create figure with two subplots (Line plot + Boxplot)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]})

    # Extract best-fit Froude numbers
    cteKey     = min_CteKey.split('_')[1]
    SquaredKey = min_SquaredKey.split('_')[1]
    LinearKey  = min_LinearKey.split('_')[1], min_LinearKey.split('_')[3]

    # Format best-fit model names for legends
    legend_cte     = f'Cte F0 = {cteKey}'
    legend_squared = f'Squared F0 = {SquaredKey}'
    legend_linear  = f'Linear F0={LinearKey[0]} and FR={LinearKey[1]}'

    # Compute mean and standard deviation for squared and linear models
    mean_squared = np.mean(np.abs(errorSquared_opti))
    std_squared = np.std(np.abs(errorSquared_opti))
    mean_linear = np.mean(np.abs(errorLinear_opti))
    std_linear = np.std(np.abs(errorLinear_opti))

    # Define consistent colors for the plots
    colors = ['#1f77b4', '#ff7f0e', 'm']

    # === Line Plot ===
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorCte_opti, marker='x', ls='--', lw=0.5, 
               label=legend_cte, color=colors[0])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorSquared_opti, marker='o', ls='--', lw=0.5, 
               label=legend_squared, color=colors[1])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorLinear_opti, marker='v', ls='--', lw=0.5, 
               label=legend_linear, color=colors[2])

    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_xlabel("Simulations", fontsize=12)
    ax[0].set_ylabel("EA %", fontsize=12)
    ax[0].set_xticks(np.arange(2, len(Allscenarios) + 1, 4))  # Set ticks every 4 simulations
    ax[0].set_xlim(0, len(Allscenarios) + 0.5)

    # === Boxplot ===
    data = pd.DataFrame({
        'Error': list(np.abs(errorCte_opti)) + list(np.abs(errorSquared_opti)) + list(np.abs(errorLinear_opti)),
        'Category': ['Constant'] * len(errorCte_opti) + 
                    ['Squared'] * len(errorSquared_opti) + 
                    ['Linear'] * len(errorLinear_opti)
    })

    sns.boxplot(x='Category', y='Error', data=data, ax=ax[1], hue='Category', palette=colors, legend=False)
    ax[1].set_ylabel("EA(%)", fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].set_xlabel('Froude Number')

    stats_text = (
        f"Squared:  Mean: {mean_squared:.2f}  Std: {std_squared:.2f}\n"
        f"Linear:  Mean: {mean_linear:.2f}  Std: {std_linear:.2f}"
    )
    ax[1].text(0.25, 0.9, stats_text, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(outfigPath / f"Best-fit_models_along_scenarios.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.1)

    return selected_scenarios

#--------------------------------------------------------
#--------------------------------------------------------

def plot_froude_parameterization(min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear, outfigPath):
    """
    Plots Froude parameterization for constant, linear, and squared models.
    
    Parameters:
    - min_F0_cte (float): Constant Froude number.
    - min_F0_squared (float): Initial Froude number for squared model.
    - min_F0_linear (float): Initial Froude number for linear model.
    - min_FR_linear (float): Final Froude number for linear model.
    
    Returns:
    - None (displays the plot).
    """

    # Ensure values are floats
    min_F0_cte = float(min_F0_cte)
    min_F0_squared = float(min_F0_squared)
    min_F0_linear = float(min_F0_linear)
    min_FR_linear = float(min_FR_linear)

    # Generate distance vector
    distance = np.linspace(0, 1, 100)  # Normalized distance from 0 to 1
    XR = distance[-1]  # Last point of the distance vector

    # Compute Froude parameterizations
    froude_cte = np.full_like(distance, min_F0_cte)  # Constant value
    froude_linear = min_F0_linear + (min_FR_linear - min_F0_linear) * (distance / XR)
    froude_squared = min_F0_squared * (1 - distance / XR) ** 0.5

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(distance, froude_cte, label=f"Constant F0 = {min_F0_cte}", linestyle="--")
    plt.plot(distance, froude_linear, label=f"Linear: F0={min_F0_linear}, FR={min_FR_linear}", linestyle="-.")
    plt.plot(distance, froude_squared, label=f"Squared: F0={min_F0_squared}", linestyle=":")

    # Labels and legend
    plt.xlabel("Normalized Distance")
    plt.ylabel("Froude Number")
    plt.title("Froude Parameterization")
    plt.legend()
    plt.grid(True)
    plt.savefig(outfigPath / f"Best-fit_froude_parameterization.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.01)
#--------------------------------------------------------
#--------------------------------------------------------

def initial_heights_by_scenario(data_dict):
    """
    This function calculates the min, max, mean, and std of the initial height (df['height'].iloc[0]) 
    for all transects 'T00X' within each 'S00X'.

    Parameters:
    data_dict (dict): Dictionary where the keys are 'S00X_T00X' and the values are DataFrames with a 'height' column.

    Returns:
    dict: A dictionary where each key is 'S00X' and the value is a list with [min, max, mean, std] 
          of the initial heights for all 'T00X' transects within that 'S00X'.
    """
    result_dict = {}

    # Iterate through the dictionary and collect initial heights for each 'S00X'
    for key, df in data_dict.items():
        s_key = key.split('_')[0]  # Extract the 'S00X' part of the key
        initial_height = df['hmax'].iloc[0]  # Get the first value of the 'height' column
        
        # If 'S00X' is not already in the result_dict, initialize an empty list
        if s_key not in result_dict:
            result_dict[s_key] = []

        # Append the initial height to the list for the corresponding 'S00X'
        result_dict[s_key].append(initial_height)

    # Now calculate min, max, mean, and std for each 'S00X'
    summary_dict = {}
    for s_key, heights in result_dict.items():
        # Convert the list of heights to a NumPy array for easier calculation
        heights_array = np.array(heights)
        summary_dict[s_key] = [np.nanmin(heights_array), np.nanmax(heights_array), np.nanmean(heights_array), np.nanstd(heights_array)]

    return summary_dict

#--------------------------------------------------------
#--------------------------------------------------------
def plot_flooding_curve(flood_extent_dict, Hmean, outfigPath):
    """
    Plots the relationship between average shoreline flood depth (Hmean) and flood extent area 
    for the best-fit model (Squared) and SWE simulation.

    Parameters:
    - flood_extent_dict (dict): Dictionary containing flood extent polygons and area values 
                                for different scenarios. Structured as:
                                {scenario: {"swe": [polygon, area], "best_fit": [polygon, area]}}
    - Hmean (list or np.array): List or array of average shoreline flood depths.

    Returns:
    - None (Displays a figure with a scatter plot, top histogram, and side histogram).
    """
    
    # Extract SWE and best-fit (Squared) areas
    AreaSim = np.array([flood_extent_dict[scenario]["swe"][1] for scenario in flood_extent_dict.keys()]) 
    
    AreaFEGLA = np.array([flood_extent_dict[scenario]["best_fit"][1] for scenario in flood_extent_dict.keys()])

    # Create the figure and grid layout
    fig = plt.figure(figsize=(9, 6))
    grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)

    # Central plot (Scatterplot)
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    sns.scatterplot(x=Hmean, y=AreaFEGLA, ax=main_ax, s=20, edgecolor=None, label='FEGLA')
    sns.scatterplot(x=Hmean, y=AreaSim, ax=main_ax, s=20, edgecolor=None, label='SWE Simulations')
    main_ax.set_xlabel("Average shoreline flood depth (m)", fontsize=12)
    main_ax.set_ylabel("Area ($km^2$)", fontsize=12)
    main_ax.grid(True, linestyle='--', alpha=0.7)
    main_ax.set_xticks(np.arange(0, max(Hmean) + 1, 2))  # Adjust `max_x_value` as needed
    main_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Ensure only integers appear
    main_ax.legend()

    # Top histogram (Distribution of Hmean)
    top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    sns.histplot(Hmean, bins=20, kde=True, stat='density', color="green", alpha=0.3, ax=top_ax)
    top_ax.set_ylabel("Density", fontsize=12)
    top_ax.set_xlabel("")
    top_ax.tick_params(labelbottom=False)

    # Side histogram (Distribution of Areas)
    side_ax = fig.add_subplot(grid[1:4, 3], sharey=main_ax)

    # Define histogram bins
    max_area = np.max([np.max(AreaSim), np.max(AreaFEGLA)])
    bins = np.linspace(0, max_area, 20)
    hist_Squared, _ = np.histogram(AreaFEGLA, bins=bins, density=True)
    hist_Sim, _ = np.histogram(AreaSim, bins=bins, density=True)

    bar_width = 0.3  # Bar width

    # Positions for bars
    bar_positions = (bins[:-1] + bins[1:]) / 2  # Center of bins

    # Plot side-by-side bars
    side_ax.barh(bar_positions - bar_width / 2, hist_Squared, height=bar_width, color='#1f77b4', label='FEGLA')
    side_ax.barh(bar_positions + bar_width / 2, hist_Sim, height=bar_width, color='#ff7f0e', label='SWE Simulations')

    # Labels and legend
    side_ax.set_xlabel("Density", fontsize=12)
    side_ax.set_ylabel("", fontsize=12)
    #side_ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1, 1.05), ncol=1)
    plt.savefig(outfigPath / f"Flooding_Curve.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.1)