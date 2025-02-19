#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
from scipy.interpolate import griddata, CubicSpline, RegularGridInterpolator 
from tqdm import tqdm
from collections import defaultdict
from shapely.geometry import Polygon
from skimage import measure

 
#--------------------------------------------------------
#
# FUNCTIONS
#
#--------------------------------------------------------
def plot_flood_for_transect(scen_tran, params):
    """
    Plots the flood elevation for a given transect, highlighting the area between the elevation and flood levels,
    and indicating the runup and XR values with dashed lines.

    Parameters:
    - scen_tran (DataFrame): DataFrame containing the transect data with columns 'Cum_distance', 'Elevation', and 'flood'.
    - params (dict): Dictionary containing 'R' (runup) and 'XR' (cumulative distance for runup).
    """
    # Directly work with the original DataFrame and params dictionary
    fig, ax = plt.subplots(figsize=(16, 3))

    # Plot the elevation and flood data
    ax.plot(scen_tran['cum_distance'], scen_tran['elevation'], label='Elevation')
    ax.plot(scen_tran['cum_distance'], scen_tran['flood'], label='Flood Level')

    # Prepare data for fill_between
    cum_dist = scen_tran['cum_distance']
    elevation = scen_tran['elevation']
    flood = scen_tran['flood']

    # Fill the area between the elevation and flood level where flood level is above elevation
    ax.fill_between(cum_dist, elevation, flood, where=(flood >= elevation), facecolor='blue', interpolate=True, alpha=0.6)

    # Fill the area between the zero level and elevation where elevation is above zero level
    ax.fill_between(cum_dist, 0, elevation, where=(elevation >= 0), facecolor='brown', interpolate=True, alpha=0.3)

    # Add horizontal and vertical dashed lines for Runup and XR
    ax.axhline(y=params['R'], color='black', linestyle='dashed', label='Runup')
    ax.axvline(x=params['XR'], color='brown', linestyle='dashed', label='$X_R$')

    # Set labels and legend
    ax.set_xlabel('Horizontal distance [m]')
    ax.set_ylabel('Elevation [m NMM]')
    ax.legend()

    # Display the plot
    plt.show()

#-------------------------------------------------------
#-------------------------------------------------------
def create_batches_from_combinations(combinations, batch_size):
    """
    Creates a list of batches where each batch contains up to `batch_size` combinations.
    Each combination corresponds to a specific scenario and transect pair.

    Parameters:
    - combinations (List[str]): The list of valid scenario-transect combinations.
    - batch_size (int): The number of combinations in each batch.

    Returns:
    - List[List[str]]: A list of batches, where each batch is a list of combinations.
    """
    # Create batches by slicing the combinations list into chunks of size `batch_size`
    batches = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
    
    return batches

#-------------------------------------------------------
#-------------------------------------------------------
def load_batch_from_hdf5(filepath, batch_keys):
    """
    Loads a batch of DataFrames from an HDF5 file based on a list of keys.

    Parameters:
    - filepath (str): The path to the HDF5 file.
    - batch_keys (List[str]): A list of keys (scenario_transect combinations) to load from the HDF5 file.

    Returns:
    - dict: A dictionary where keys are the scenario_transect identifiers and values are the corresponding DataFrames.
    """
    batch_data = {}

    # Open the HDF5 file
    with pd.HDFStore(filepath, mode='r') as store:
        # Load each DataFrame corresponding to the keys in batch_keys
        for key in batch_keys:
            if key in store:
                batch_data[key] = store[key]
            else:
                print(f"Warning: Key '{key}' not found in the HDF5 file.")

    return batch_data

#-------------------------------------------------------
#-------------------------------------------------------
def load_results_for_iteration(file_path):
    """
    Load the last iteration data from a pickle file.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    tuple: A tuple containing the last height iteration, XRmax iteration, XRmin iteration, and error iteration.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    height_iteration = data['height_iteration'][-1]
    XRmax_iteration = data['XRmax_iteration']
    XRmin_iteration = data['XRmin_iteration']
    error_iteration = data['Error_h0']

    return height_iteration, XRmax_iteration, XRmin_iteration, error_iteration

#-------------------------------------------------------
#-------------------------------------------------------
def remove_nan_values(list1, list2):
    """
    Remove pairs where either of the values in the two lists is NaN.

    Parameters:
    - list1 (array-like): First list of values.
    - list2 (array-like): Second list of values.

    Returns:
    - tuple: Two lists with NaN pairs removed.
    """
    mask = ~np.isnan(list1) & ~np.isnan(list2)
    return list1[mask], list2[mask]

#-------------------------------------------------------
#-------------------------------------------------------
def load_height_from_simulations(path):
    '''
    Loads a dictionary of dictionaries containing DataFrames from an HDF5 file.

    Parameters:
    path (str): The file path from where the HDF5 file will be loaded.

    Returns:
    dict: A dictionary where keys are scenarios, and values are dictionaries of transects. Each transect dictionary contains DataFrames.
    '''
    data = {}

    # Open the HDF5 file at the given path in read mode
    with pd.HDFStore(path, mode='r') as store:
        keys = store.keys()
        for key in tqdm(keys, desc="Loading data from HDF5"):
            # Get scenario and transect identifiers
            scenario_transect = key.strip('/')
            
            # Load the DataFrame for the current transect
            data[scenario_transect] = store.get(key)
    
    return data


#-------------------------------------------------------
#-------------------------------------------------------
def plot_topobathymetry_and_contours(x, y, z, elev_min=-90, elev_max=240, elev_delta=30, z0_contour=None, cmap='viridis', ax=None, alpha=1):
    """
    Plot a predefined elevation map with contours and a color bar. A contour for z=0 can also be plotted if provided.

    Parameters:
    x (np.array): 2D array of longitude values for each grid point.
    y (np.array): 2D array of latitude values for each grid point.
    z (np.array): 2D array of elevation data at the grid points defined by x and y.
    elev_min (int): Minimum elevation value for contour levels.
    elev_max (int): Maximum elevation value for contour levels.
    elev_delta (int): Interval between contour levels.
    z0_contour (np.array): Optional. 2D array of x, y coordinates for the z=0 contour line.
    cmap: to define colormap

    Returns:
    fig (matplotlib.figure.Figure): Matplotlib Figure object for further customization.
    ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib subplot axes for further customization.
    """
    # Initialize the plot and set its size
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    # Define the contour levels based on input parameters
    contour_levels = np.arange(elev_min, elev_max + elev_delta, elev_delta)

    # Fill the contours with color based on the elevation levels
    cp = ax.contourf(x, y, z, levels=contour_levels, cmap=cmap)

    # Draw the contour lines and label them
    cs = ax.contour(x, y, z, levels=contour_levels, colors='black', linestyles='solid', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f m')

    # Add a color bar with specific scale intervals
    cbar = plt.colorbar(cp, ax=ax, label='Elevation (m)', ticks=contour_levels)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(elev_delta))

    # Plot the z=0 contour line if provided
    if z0_contour is not None:
        ax.plot(z0_contour[:, 0], z0_contour[:, 1], c='r', label='Shoreline [0 m]')
        ax.legend()

    # Set the title and labels for the axes
    ax.set_title('Elevation Map')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')

    # Set x and y axis ticks to increment by a specific interval
    #ax.set_xticks(np.arange(np.min(x)+0.01, np.max(x), 0.01))
    #ax.set_yticks(np.arange(np.min(y)+0.01, np.max(y), 0.01))

    # Format the tick labels to show with two decimal places
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if create_new_fig:
        return fig, ax
    return ax
#-------------------------------------------------------
#-------------------------------------------------------
def plot_surface_and_area_for_scenario(x_UTM, y_UTM, z_elev, shoreline, boundary_x, boundary_y, grid_x, grid_y, grid_z, area, city, surface=False, cmap_scale=[0,1.5, 0.1]):
    '''
    Plots a flood map with topobathymetry, shoreline, and optionally a flood surface.

    Parameters:
    - x_UTM: numpy.ndarray, 1D array of UTM x coordinates.
    - y_UTM: numpy.ndarray, 1D array of UTM y coordinates.
    - z_elev: numpy.ndarray, 2D array of elevation data corresponding to x_UTM and y_UTM.
    - shoreline: numpy.ndarray, 2D array of shoreline coordinates [lon, lat].
    - boundary_x: numpy.ndarray, 1D array of x coordinates for the flood map boundary.
    - boundary_y: numpy.ndarray, 1D array of y coordinates for the flood map boundary.
    - grid_x: numpy.ndarray, 2D array of x coordinates for the flood surface grid.
    - grid_y: numpy.ndarray, 2D array of y coordinates for the flood surface grid.
    - grid_z: numpy.ndarray, 2D array of flood height values.
    - area: float, area of the flood map in square meters.
    - city: str, name of the city for the title.
    - surface: bool, if True, plots the flood surface.
    - cmap_scale: list, color scale for the flood surface [min, max, interval].

    Returns:
    - None
    '''

    # Create the plot with a reasonable figure size
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot topobathymetry and shoreline
    plot_topobathymetry_and_contours(x_UTM, y_UTM, z_elev,
                                     elev_min=-90,
                                     elev_max=240,
                                     elev_delta=30,
                                     z0_contour=shoreline,
                                     cmap='gray', ax=ax, alpha=1)

    # Plot flood map boundary
    ax.plot(boundary_x, boundary_y, 'm-', label='Flood map edges')

    # Optionally plot the flood surface
    if surface:
        levels = np.arange(cmap_scale[0], cmap_scale[1], cmap_scale[2])
        contourf_plot = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='viridis', vmin=cmap_scale[0], vmax=cmap_scale[1])
        # Add colorbar for flood surface
        cbar = plt.colorbar(contourf_plot, ax=ax, label='Height [m]')
    
    # Set title
    plt.title(f'{city} Flood Map')
    
    # Add area text to the plot
    ax.text(0.95, 0.05, f'Area: {area:.2f} m²\nArea: {area/1e6:.2f} km$^2$', 
            transform=ax.transAxes, fontsize=12, color='white',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='black', alpha=0.5))

    # Ensure equal scaling of axes
    ax.axis('equal')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

#-------------------------------------------------------
#-------------------------------------------------------
def find_contour_coordinates(x, y, z, level=0):
    """
    Finds the geographic coordinates of a contour at a specified elevation level 
    within a dataset where x, y represent longitude and latitude respectively, 
    and z represents elevation.
    
    Parameters:
        x (np.array): 2D array of longitude values for each grid point.
        y (np.array): 2D array of latitude values for each grid point.
        z (np.array): 2D array representing elevation data at the grid points defined by x and y.
        level (float): The elevation level at which to find the contour.
    
    Returns:
        np.array: A 2D array where each row contains the geographic coordinates 
        [longitude, latitude] of points along the contour of the specified elevation level.
    """
    
    # Find the contours at the specified elevation level
    contours = measure.find_contours(z, level)
    
    # Initialize an empty list to store the transformed coordinates
    coordinates = []

    # Iterate over each contour
    for contour in contours:
        for point in contour:
            # Los contornos devuelven los índices, no las coordenadas en lon/lat
            i, j = int(point[0]), int(point[1])
            lon = x[j]
            lat = y[i]
            coordinates.append((lon, lat))

    # Convertir a numpy array para su manipulación
    coordinates = np.array(coordinates)
    
    return coordinates

#-------------------------------------------------------
#-------------------------------------------------------
def moving_average_extended(data, window_size):
    """ Apply a simple moving average filter to the data, preserving the original length. """
    padded_data = np.pad(data, (window_size//2, 
                                window_size-1-window_size//2), mode='edge')
    smoothed_data = np.convolve(padded_data, 
                                np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

#-------------------------------------------------------
#-------------------------------------------------------
def create_spline_function(shoreline, n_point=1000, smooth_window=20):
    """
    Creates a cubic spline interpolation function for a given shoreline curve using arc length parameterization.
    """
    # Separate the x and y coordinates
    x = shoreline[:, 0]
    y = shoreline[:, 1]

    # Calculate the arc length of each segment
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    arc_lengths = np.concatenate(([0], np.cumsum(distances)))

    # Filter out duplicate arc lengths
    unique_arc_lengths, unique_indices = np.unique(arc_lengths, return_index=True)
    x_unique = x[unique_indices]
    y_unique = y[unique_indices]
    print(f"Unique arc lengths: {len(unique_arc_lengths)}")

    if len(unique_arc_lengths) < 2:
        raise ValueError("Not enough unique points for spline interpolation. Check input data or reduction factor.")

    # Create the cubic spline interpolation functions
    spline_function_x = CubicSpline(unique_arc_lengths, x_unique)
    spline_function_y = CubicSpline(unique_arc_lengths, y_unique)

    # Generate a dense set of arc length points for a smooth curve
    arc_lengths_dense = np.linspace(unique_arc_lengths[0], unique_arc_lengths[-1], num=n_point)

    # Evaluate the spline functions at the dense arc length points
    x_smooth = spline_function_x(arc_lengths_dense)
    y_smooth = spline_function_y(arc_lengths_dense)

    # Apply moving average to smooth the points further
    x_smooth = moving_average_extended(x_smooth, smooth_window)
    y_smooth = moving_average_extended(y_smooth, smooth_window)

    # Combine the x and y dense points into an array
    smoothed_points = np.column_stack((x_smooth, y_smooth))

    return spline_function_x, spline_function_y, smoothed_points

#-------------------------------------------------------
#-------------------------------------------------------
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



#-------------------------------------------------------
#-------------------------------------------------------
def get_boundary_points(data):
    """
    Extracts the boundary points (first and last) for each transect in the provided data,
    calculates a polygon that connects these boundary points, and computes the polygon's area.

    Parameters:
    - data (dict): A dictionary where keys are identifiers and values are DataFrames containing
      'lon', 'lat', and 'height' columns, representing the transect data.

    Returns:
    - polygonSim (shapely.geometry.Polygon): The polygon created by connecting the boundary points.
    - aream2Sim (float): The area of the polygon in square meters.
    """

    # Initialize lists to store the first and last points of each transect
    firstptoList = list()
    lastptoList = list()

    # Loop through each item in the data dictionary
    for _, dfdummy in data.items():
        # Create a copy of the current DataFrame
        df = dfdummy.copy()

        # Find the last valid index for the 'height' column and slice the DataFrame up to that point
        index = df['height'].last_valid_index()
        dfSim = df.iloc[:index]

        try:
            # Attempt to extract the first and last points from the sliced DataFrame
            firstptoList.append((dfSim['lon'].values[0], dfSim['lat'].values[0]))  # First point
            lastptoList.append((dfSim['lon'].values[-1], dfSim['lat'].values[-1]))  # Last point
        except:
            # If slicing fails, use the original DataFrame
            firstptoList.append((dfdummy['lon'].values[0], dfdummy['lat'].values[0]))  # First point
            lastptoList.append((dfdummy['lon'].values[1], dfdummy['lat'].values[1]))  # Last point

    # Convert the lists of points into numpy arrays
    firstptoArray = np.array(firstptoList)
    lastptoArray = np.array(lastptoList)

    # Create a polygon and calculate its area using the boundary points
    polygonSim, aream2Sim = calculate_polygon_and_area(firstptoArray, lastptoArray)

    return polygonSim, aream2Sim

#--------------------------------------------
#--------------------------------------------
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
        initial_height = df['height'].iloc[0]  # Get the first value of the 'height' column
        
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

#--------------------------------------------
#--------------------------------------------
def load_transect_from_hdf5(path):
    '''
    Loads all DataFrames from an HDF5 file into a dictionary, converting float64 to float32 
    to optimize memory usage.

    Parameters:
    path (str): The file path where the HDF5 file is stored.

    Returns:
    dict: A dictionary where keys are the original keys used when saving 
          and values are the corresponding optimized DataFrames.
    '''
    df_dict = {}
    
    # Open the HDF5 file at the given path in read mode
    with pd.HDFStore(path, mode='r') as store:
        # Iterate over all keys in the HDF5 store
        for key in store.keys():
            # Load the DataFrame
            df = store[key]
            # Convert float64 columns to float32 to save memory
            float_columns = df.select_dtypes(include=['float64']).columns
            df[float_columns] = df[float_columns].astype('float32')
            # Store the optimized DataFrame in the dictionary
            df_dict[key] = df
    
    return df_dict

#--------------------------------------------
#--------------------------------------------

def equidistant_points_on_curve(lon_curve, lat_curve, distance=20):
    '''
    Calculate points along a curve that are approximately equidistant based on a specified distance.

    Parameters:
    lon_curve (np.array): The longitude coordinates of the curve.
    lat_curve (np.array): The latitude coordinates of the curve.
    distance (float): Target distance in meters for equidistant points.

    Returns:
    tuple: Two arrays containing the longitude and latitude coordinates of the equidistant points along the curve.
    '''
    # Convert differences in longitude and latitude to distances in meters
    avg_lat = np.mean(lat_curve)
    lat_to_meters = 111000
    lon_to_meters = 111000 * np.cos(np.radians(avg_lat))

    # Calculate differential lengths along the curve in meters
    dx_meters = np.diff(lon_curve) * lon_to_meters
    dy_meters = np.diff(lat_curve) * lat_to_meters
    distances_meters = np.sqrt(dx_meters**2 + dy_meters**2)
    cumulative_distance = np.cumsum(distances_meters)
    cumulative_distance = np.insert(cumulative_distance, 0, 0)  # Insert 0 at the start
    
    # Initialize output arrays
    lon_equidistant = []
    lat_equidistant = []

    # Iterate through multiples of the distance
    target = distance
    for i, d in enumerate(cumulative_distance):
        if d >= target:
            lon_equidistant.append(lon_curve[i])
            lat_equidistant.append(lat_curve[i])
            target += distance  # Move to the next target distance

    return np.array(lon_equidistant), np.array(lat_equidistant)

#--------------------------------------------
#--------------------------------------------
def elevation_function(bathy, lon, lat):
    """
    Interpolates the elevation for the specified coordinates (lon, lat).

    Parameters:
    - bathy: xarray.DataArray, bathymetric dataset.
    - lon: float, longitude.
    - lat: float, latitude.

    Returns:
    - float, interpolated elevation value.
    """
    elevation = bathy.interp(lon=lon, lat=lat)
    return elevation.values.item()

#--------------------------------------------
#--------------------------------------------
def perpendicular_line(x_equidistant, y_equidistant, smoothed_points, extension_length, bathy, dummy_length = 100):
    """
    Generates perpendicular lines to the shoreline using equidistant points from the original shoreline
    and the slopes of the smoothed shoreline. Uses a dummy distance to determine the elevation direction.

    Parameters:
    - x_equidistant: numpy.ndarray, x-coordinates (longitude) of the equidistant points.
    - y_equidistant: numpy.ndarray, y-coordinates (latitude) of the equidistant points.
    - smoothed_points: numpy.ndarray, smoothed shoreline points (lon, lat).
    - extension_length: float, length of the perpendicular lines (in meters).
    - bathy: xarray.DataArray, bathymetric dataset with dimensions (lon, lat).

    Returns:
    - lines_latlon: numpy.ndarray, starting points of the perpendicular lines (lon, lat).
    - lines_UTM: numpy.ndarray, ending points of the perpendicular lines (x, y).
    """
    # Conversion factors: degrees to meters
    avg_lat = np.mean(y_equidistant)
    lat_to_meters = 111000  # Approx. 111 km per degree of latitude
    lon_to_meters = 111000 * np.cos(np.radians(avg_lat))  # Adjusted by average latitude

    # Convert smoothed_points to meters
    smoothed_points_meters = np.array([
        [lon * lon_to_meters, lat * lat_to_meters] for lon, lat in smoothed_points
    ])

    # Convert equidistant points to meters
    x_meters = x_equidistant * lon_to_meters
    y_meters = y_equidistant * lat_to_meters

    lines_latlon = []
    lines_UTM    = []

    for x_meter, y_meter in zip(x_meters, y_meters):
        # Find the closest point on the smoothed shoreline
        distances = np.sqrt((smoothed_points_meters[:, 0] - x_meter) ** 2 +
                            (smoothed_points_meters[:, 1] - y_meter) ** 2)
        nearest_index = np.argmin(distances)

        # Calculate the slope at the nearest point using its neighbors
        if nearest_index == 0:  # Start of the smoothed shoreline
            x_next, y_next = smoothed_points_meters[nearest_index + 1]
            x_prev, y_prev = x_meter, y_meter
        elif nearest_index == len(smoothed_points_meters) - 1:  # End of the smoothed shoreline
            x_prev, y_prev = smoothed_points_meters[nearest_index - 1]
            x_next, y_next = x_meter, y_meter
        else:  # Intermediate point
            x_prev, y_prev = smoothed_points_meters[nearest_index - 1]
            x_next, y_next = smoothed_points_meters[nearest_index + 1]

        slope = (y_next - y_prev) / (x_next - x_prev) if x_next != x_prev else np.inf
        perpendicular_slope = -1 / slope if slope != 0 else np.inf

        # Calculate the dummy line endpoints
        if perpendicular_slope == np.inf:  # Vertical line
            line_end1 = (x_meter, y_meter + dummy_length / 2)
            line_end2 = (x_meter, y_meter - dummy_length/ 2)
        else:
            delta_x_dummy = dummy_length / np.sqrt(1 + perpendicular_slope**2)
            delta_y_dummy = perpendicular_slope * delta_x_dummy
            line_end1 = (x_meter + delta_x_dummy, y_meter + delta_y_dummy)
            line_end2 = (x_meter - delta_x_dummy, y_meter - delta_y_dummy)

        # Convert dummy line endpoints to lon/lat
        line_end1_lonlat = (line_end1[0] / lon_to_meters, line_end1[1] / lat_to_meters)
        line_end2_lonlat = (line_end2[0] / lon_to_meters, line_end2[1] / lat_to_meters)

        # Pass bathy to the elevation_function
        elevation1 = elevation_function(bathy, *line_end1_lonlat)
        elevation2 = elevation_function(bathy, *line_end2_lonlat)
        
        if np.isnan(elevation1) or np.isnan(elevation2):
            continue  # Skip this line if elevation data is invalid

        # Choose the correct direction based on elevation
        if elevation1 > elevation2:
            direction = 1  # Use line_end1 as the direction
        else:
            direction = -1  # Use line_end2 as the direction

        # Calculate the final line endpoints with extension_length
        if perpendicular_slope == np.inf:  # Vertical line
            line_start = (x_meter, y_meter)
            line_end = (x_meter, y_meter + direction * extension_length / 2)
        else:
            delta_x = direction * extension_length / np.sqrt(1 + perpendicular_slope**2)
            delta_y = perpendicular_slope * delta_x
            line_start = (x_meter, y_meter)
            line_end = (x_meter + delta_x, y_meter + delta_y)

        line = np.linspace(line_start, line_end, extension_length)
        lines_latlon.append((line[:,0] /lon_to_meters, line[:,1]/lat_to_meters ))
        lines_UTM.append((line[:,0], line[:,1]))

    return lines_latlon, lines_UTM

#--------------------------------------------
#--------------------------------------------
def create_interpolation_function(maps):
    """
    Creates an interpolation function based on the provided maps.

    Parameters:
    maps (list of np.ndarray): List containing arrays of x-coordinates, y-coordinates, and values (e.g., topographical or bathymetrical data).

    Returns:
    callable: Function that takes x and y coordinates and returns interpolated values.
    """
    coords = np.column_stack((maps[0].ravel(), maps[1].ravel()))
    topobathy = maps[2].ravel()

    def interpolation_function(x, y):
        points = np.column_stack((x, y))
        return griddata(coords, topobathy, points, method='linear')
    
    return interpolation_function


#--------------------------------------------
#--------------------------------------------
def save_transects_to_netcdf(transects, filename):
    """
    Save a dictionary of dataframes to a NetCDF file using xarray.

    Parameters:
    transects (dict): Dictionary where keys are transect identifiers and values are pandas DataFrames.
    filename (str): Path to the output NetCDF file.
    """
    # Find the maximum length of the transects
    max_length = max(len(df) for df in transects.values())
    
    # Prepare data arrays with NaN padding for variable lengths
    transect_ids = list(transects.keys())
    lon_data = np.full((len(transect_ids), max_length), np.nan)
    lat_data = np.full((len(transect_ids), max_length), np.nan)
    x_data   = np.full((len(transect_ids), max_length), np.nan)
    y_data   = np.full((len(transect_ids), max_length), np.nan)
    elevation_data = np.full((len(transect_ids), max_length), np.nan)
    
    for i, (transect_id, df) in enumerate(transects.items()):
        lon_data[i, :len(df)] = df['lon'].values
        lat_data[i, :len(df)] = df['lat'].values
        x_data[i, :len(df)] = df['x'].values
        y_data[i, :len(df)] = df['y'].values
        elevation_data[i, :len(df)] = df['Elevation'].values
    
    # Create an xarray Dataset
    ds = xr.Dataset(
        {   'lon': (['transect', 'point'], lon_data),
            'lat': (['transect', 'point'], lat_data),
            'x': (['transect', 'point'], x_data),
            'y': (['transect', 'point'], y_data),
            'elevation': (['transect', 'point'], elevation_data)
        },
        coords={
            'transect': transect_ids,
            'point': np.arange(max_length)
        }
    )
    
    # Add metadata
    ds.attrs['description'] = 'Transect data saved from dataframes'
    ds.attrs['history'] = 'Created ' + pd.Timestamp.now().isoformat()
    
    # Save to NetCDF file
    ds.to_netcdf('./Processed/'+filename)

#--------------------------------------------
# Elimating????
#--------------------------------------------

def save_to_netcdf(scenario_dict, filename):
    """
    Save the nested dictionary of DataFrames to a NetCDF file.

    Parameters:
    - scenario_dict: dict, nested dictionary where keys are simulation identifiers and values are dictionaries of transects with interpolated heights.
    - filename: str, the path to save the NetCDF file.
    """
    # Create a defaultdict for storing data arrays
    data_vars = defaultdict(list)

    # Extracting data and preparing for xarray Dataset creation
    for sim_key, transects in scenario_dict.items():
        sim_index = int(sim_key[1:]) - 1  # Convert 'S001' to index 0
        for transect_key, df in transects.items():
            transect_index = int(transect_key[1:]) - 1  # Convert 'T001' to index 0
            for col in df.columns:
                if col not in data_vars:
                    data_vars[col] = []
                data_vars[col].append(((sim_index, transect_index), df[col].values))

    # Create lists for storing variables and coordinates
    variables = {}
    coords = {'simulation': [], 'transect': [], 'point': []}

    # Convert collected data to xarray DataArray
    for var_name, values in data_vars.items():
        if var_name not in ['lon', 'lat', 'x', 'y', 'Elevation', 'height']:
            continue  # Skip if not a relevant variable
        max_len = max(len(val[1]) for val in values)
        data = np.full((len(scenario_dict), len(transects), max_len), np.nan)

        for (sim_idx, trans_idx), val in values:
            data[sim_idx, trans_idx, :len(val)] = val
        
        variables[var_name] = (('simulation', 'transect', 'point'), data)

    # Creating the Dataset
    ds = xr.Dataset(
        data_vars=variables,
        coords={
            'simulation': [f'S{str(i+1).zfill(3)}' for i in range(len(scenario_dict))],
            'transect': [f'T{str(i+1).zfill(3)}' for i in range(len(transects))],
            'point': np.arange(max_len)
        }
    )

    # Adding attributes (optional)
    ds.attrs['description'] = 'Interpolated transect data'
    ds.attrs['units'] = 'Degrees and meters'

    # Save to NetCDF
    ds.to_netcdf(filename)


#--------------------------------------------
#--------------------------------------------

def load_transects_from_netcdf(filename):
    """
    Load a NetCDF file containing transect data into a dictionary of pandas DataFrames.

    Parameters:
    filename (str): Path to the NetCDF file.

    Returns:
    dict: Dictionary where keys are transect identifiers and values are pandas DataFrames.
    """
    # Load the NetCDF file using xarray
    ds = xr.open_dataset(filename)
    
    # Convert to a dictionary of dataframes
    transects = {}
    for transect_id in ds.transect.values:
        df = pd.DataFrame({
            'lon': ds['lon'].sel(transect=transect_id).values,
            'lat': ds['lat'].sel(transect=transect_id).values,
            'x': ds['x'].sel(transect=transect_id).values,
            'y': ds['y'].sel(transect=transect_id).values,
            'Elevation': ds['elevation'].sel(transect=transect_id).values
        })
        df = df.dropna().reset_index(drop=True)  # Remove rows with NaN and reset index
        transects[transect_id] = df
    
    # Close the dataset
    ds.close()

    return transects

#--------------------------------------------
# Elimating????
#--------------------------------------------
def load_from_netcdf(filename):
    """
    Load a NetCDF file and reconstruct the nested dictionary of DataFrames.

    Parameters:
    - filename: str, the path to the NetCDF file.

    Returns:
    - dict, nested dictionary where keys are simulation identifiers and values are dictionaries of transects with interpolated heights.
    """
    # Load the Dataset from the NetCDF file
    ds = xr.open_dataset(filename)

    # Prepare the output dictionary
    scenario_dict = {}

    # Iterate over each simulation
    for sim in ds.simulation.values:
        transects_for_sim = {}
        
        # Iterate over each transect within the simulation
        for trans in ds.transect.values:
            # Extract DataFrame for the current transect and simulation
            df = ds.sel(simulation=sim, transect=trans).to_dataframe().reset_index()
            
            # Drop unwanted columns and set transect as index
            df = df.drop(columns=['simulation', 'transect', 'point'])
            
            # Add to the transect dictionary
            transects_for_sim[trans] = df.dropna()
        
        # Add the transect dictionary to the simulation dictionary
        scenario_dict[sim] = transects_for_sim

    return scenario_dict


#--------------------------------------------
#--------------------------------------------

def load_asc_file(filename):
    """
    Load and parse a .asc file to extract the geospatial grid data and the corresponding elevation values.

    .asc files typically contain header information such as the number of columns and rows,
    the lower left corner coordinates, the cell size, and a no-data value, followed by the elevation data.
    
    Parameters:
    filename (str): The path to the .asc file.
    
    Returns:
    x_grid (np.array): 2D array of longitude values for each grid point.
    y_grid (np.array): 2D array of latitude values for each grid point.
    data (np.array): 2D array of elevation data at the grid points defined by x_grid and y_grid.
    """
    # Define the keys that are expected in the .asc file header
    header_keys = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']
    header = {}

    # Open the file and read the header values
    with open(filename, 'r') as file:
        for key in header_keys:
            header[key] = float(file.readline().split()[1])
        
        # Load the elevation data from the file after the header
        data = np.loadtxt(file)  # Loads the z-values into a 2D numpy array

    # Extract the header information to calculate the coordinate grids
    ncols, nrows = int(header['ncols']), int(header['nrows'])
    xll, yll, cellsize = header['xllcorner'], header['yllcorner'], header['cellsize']
    
    # Generate the longitude (x) coordinates
    x_coords = xll + cellsize * np.arange(ncols)
    # Generate the latitude (y) coordinates, with a reversal to align with the ASC grid
    y_coords = yll + cellsize * np.arange(nrows)[::-1]
    
    # Create the 2D grids of longitude and latitude coordinates
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    return x_grid, y_grid, data

#--------------------------------------------
#--------------------------------------------

def get_Hmax_for_transects(initial_points, maps):
    '''
    Computes interpolated values along a transect defined by two points, extended by a specified length, at a given number of points.

    Parameters:
    initial_point (tuple): Tuple containing the x and y coordinates of the initial point.
    end_point (tuple): Tuple containing the x and y coordinates of the end point.
    extension_length (float): Length by which the transect is extended beyond the end point.
    n_points (int): Number of points at which interpolation is evaluated along the transect.
    maps (list of np.ndarray): List containing three elements: arrays of x-coordinates, y-coordinates, and values (e.g., topographical or bathymetrical data).

    Returns:
    tuple: Three numpy arrays containing the x-coordinates along the transect, the corresponding y-coordinates, and interpolated values along these coordinates.
    '''
    
    # Prepare coordinates for interpolation: flatten the map coordinates and values
    coords = np.column_stack((maps[0].ravel(), maps[1].ravel()))
    topobathy = maps[2].ravel()
    

    # Perform interpolation using griddata to find nearest values from the map data
    Z2 = griddata(coords, topobathy, initial_points, method='cubic')

    # Return the x-coordinates, y-coordinates along the transect, and interpolated values
    return Z2

#--------------------------------------------
#--------------------------------------------

def interpolate_and_organize_height(transects, hmax_local_nc, lon, lat):
    """
    Interpolates transect points on a flood height map for multiple simulations and organizes the results.

    Parameters:
    - transects: dict, dictionary where keys are transect identifiers and values are pandas DataFrames with transect data.
    - hmax_local_nc: numpy.ndarray, 3D array (simulations, lat, lon) of flood heights.
    - lon: numpy.ndarray, vector of longitude values.
    - lat: numpy.ndarray, vector of latitude values.

    Returns:
    - dict, nested dictionary where keys are simulation identifiers and values are dictionaries of transects with interpolated heights.
    """
    # Extract simulation count
    num_simulations = hmax_local_nc.shape[0]

    # Prepare the output dictionary
    scenario_dict = {}

    # Flatten transect points into a single array for efficient interpolation
    all_points_lon_lat = np.vstack([
        df[['lat', 'lon']].values for df in transects.values()
    ])

    # Interpolate heights for each simulation
    for sim_index in tqdm(range(num_simulations), desc="Interpolating simulations"):
        # Prepare the interpolator for this simulation
        height_map = hmax_local_nc[sim_index, :, :]
        interpolator = RegularGridInterpolator((lat, lon), height_map, bounds_error=False, fill_value=np.nan)

        # Perform interpolation for all points
        interpolated_heights = interpolator(all_points_lon_lat)

        # Split the interpolated heights back into the respective transect dataframes
        offset = 0
        for key, df in transects.items():
            n_points = len(df)
            df_with_height = df.copy()
            df_with_height[f'height'] = interpolated_heights[offset:offset+n_points]
            df_with_height.rename(columns={'Elevation': 'elevation'}, inplace=True)

            offset += n_points

            # Add the transects for this simulation to the main dictionary
            scenario_dict[f'S{str(sim_index+1).zfill(3)}_{key}'] = df_with_height.astype('float32')

    return scenario_dict

#--------------------------------------------
#--------------------------------------------
def optimize_dataframe(df):
    """
    Convert DataFrame columns to more memory-efficient types.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

#--------------------------------------------
#--------------------------------------------
def save_height_to_simulations(data, path):
    ''' 
    Saves a dictionary of dictionaries containing DataFrames to an HDF5 file.
    
    Parameters:
    data (dict): A dictionary where keys are scenarios, and values are dictionaries 
                 of transects. Each transect dictionary contains DataFrames.
    path (str): The file path where the HDF5 file will be saved.
    '''
    
    # Open an HDF5 file at the given path in write mode
    with pd.HDFStore(path, mode='w') as store:
        # Iterate over each scenario and its corresponding transects in the data dictionary
        for scenario, transects in data.items():
            # Iterate over each transect and its corresponding DataFrame
            for transect, df in transects.items():
                # Save the DataFrame to the HDF5 file with a key based on the scenario and transect
                store.put(f'{scenario}_{transect}', df)


#--------------------------------------------
#--------------------------------------------
def elevation_filter(df, elev):
    # Filter for elevations less than or equal to 30 meters
    df_filtered = df[df['Elevation'] <= elev]
    
    # If there are points that exceed 30 meters at any point
    if len(df_filtered) < len(df):
        # Take the first occurrence of a value greater than 30 and cut there
        first_exceedance = df[df['Elevation'] > elev].index[0]
        df_filtered = df.loc[:first_exceedance - 1]  # Keep the values up to before that point
    
    return df_filtered

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

#-------------------------------------------------------
#-------------------------------------------------------

def transect_processing(lines_latlon, lines_UTM, bathy, elevation_threshold=50):
    """
    Processes transect lines and stores their data in a dictionary of DataFrames.

    Parameters:
    - lines_latlon: list of tuples, each containing arrays of longitude and latitude points for the transects.
    - lines_UTM: list of tuples, each containing arrays of UTM X and Y points for the transects.
    - bathy: xarray.DataArray, the bathymetry data used for elevation interpolation.
    - elevation_threshold: float, elevation value above which the transect is truncated.

    Returns:
    - dict: Dictionary containing DataFrames of transect data, indexed by keys like 'T001', 'T002', etc.
    """
    # Prepare all lon and lat points from the list of lines
    all_lon = np.concatenate([line[0] for line in lines_latlon])
    all_lat = np.concatenate([line[1] for line in lines_latlon])

    # Perform interpolation for all points at once
    bathy_values = bathy.interp(lon=("points", all_lon), lat=("points", all_lat))

    # Split the bathy values back into the structure of the original list of lines
    split_indices = np.cumsum([len(line[0]) for line in lines_latlon])[:-1]
    bathy_lines = np.split(bathy_values.values, split_indices)

    # Initialize an empty dictionary to store dataframes
    lines_dataframes = {}

    # Iterate through each line and its corresponding UTM and bathy values
    for i, ((lon, lat), (x_utm, y_utm), elevation) in enumerate(zip(lines_latlon, lines_UTM, bathy_lines)):
        # Create a dataframe for the current line
        df = pd.DataFrame({
            'Lon': lon,
            'Lat': lat,
            'UTM_X': x_utm,
            'UTM_Y': y_utm,
            'Elevation': elevation
        })

        # Find the first index where elevation is equal to or greater than 50
        index_threshold = df[df['Elevation'] >= elevation_threshold].index.min()

        # Filter rows where elevation is higher than 0
        if pd.notna(index_threshold):
            # Select rows up to the index_threshold, where elevation > 0
            df_filtered = df[(df['Elevation'] >= 0) & (df.index <= index_threshold)]
        else:
            # If no value >= 50 exists, filter only based on elevation > 0
            df_filtered = df[df['Elevation'] >= 0]
        
        # Add the filtered DataFrame to the dictionary
        lines_dataframes[f'T{str(i + 1).zfill(3)}'] = df_filtered

    return lines_dataframes

#-------------------------------------------------------
#-------------------------------------------------------
def manual_shoreline_definition(grid_lon, grid_lat, bathy, current_shoreline, n_point=1000, smooth_window=20):
    """
    Allows the user to manually define a shoreline by clicking on points and creates a smoothed shoreline
    using cubic spline interpolation with additional options for smoothing.

    Parameters:
    - grid_lon: 1D array of longitude values.
    - grid_lat: 1D array of latitude values.
    - bathy: 2D array of bathymetric data.
    - kind_spline: Type of spline ('linear', 'quadratic', 'cubic').
    - n_point: Number of points to generate for the smoothed spline.
    - reduction_factor: Factor by which to reduce the points for spline interpolation.
    - smooth_window: Window size for moving average smoothing.

    Returns:
    - smoothed_points: numpy.ndarray, array of smoothed shoreline points (n_point, 2).
    """
    # Plot bathymetry for reference
    plt.figure(figsize=(10, 8))
    # Filled contour plot
    contour_filled = plt.contourf(grid_lon, grid_lat, bathy, levels=50, cmap="viridis")
    plt.colorbar(label="Elevation (m)")

    # Contour lines with labels
    contour_lines = plt.contour(grid_lon, grid_lat, bathy, levels=10, colors='black', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.0f")  # Add labels to contours

    # Titles and labels
    plt.title("Click to define the shoreline (Right-click to Finish)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Add current shoreline for reference
    plt.plot(current_shoreline[:, 0], current_shoreline[:, 1], 'm-', lw=2, label='Current Shoreline')
    plt.legend()

    # Get user input for points
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close()

    if len(points) < 2:
        raise ValueError("At least two points are required to define a shoreline.")

    # Convert user-defined points into a numpy array
    user_defined_shoreline = np.array(points)

    # Apply the spline function to smooth the manually defined shoreline
    _, _, smoothed_points = create_spline_function(
        shoreline=user_defined_shoreline,
        n_point=n_point,
        smooth_window=smooth_window
    )

    return smoothed_points, user_defined_shoreline

#-------------------------------------------------------
#-------------------------------------------------------
def plot_transect_elevations(transect_dict):
    """
    Plots the elevation profiles for each transect.

    Parameters:
    - transect_dict: dict, Dictionary of transects where each value is a DataFrame with 'Elevation'.

    Returns:
    - None (displays the plot).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate unique colors for each transect
    cmap = plt.get_cmap("tab20", len(transect_dict))

    for i, (transect_name, df) in enumerate(transect_dict.items()):
        color = cmap(i)  # Assign a unique color
        distance = np.linspace(0, len(df), len(df))  # Distance along the transect
        ax.plot(distance, df["Elevation"], linestyle="-", linewidth=1.5, color=color, label=transect_name)

    ax.set_xlabel("Distance Along Transect (Index)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Elevation Profiles for Transects")
    ax.grid(True, linestyle="--", alpha=0.6)

    # # Optional: Show only a limited number of legend labels for clarity
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc="best", fontsize=8, title="Transects (sample)")

    plt.show(block=False)  # Keep the plot open