#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import CubicSpline
from skimage import measure

 
#--------------------------------------------------------
#
# FUNCTIONS
#
#--------------------------------------------------------

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

        # Find the first index where elevation is equal to or greater than elevation threshold
        index_threshold = df[df['Elevation'] >= elevation_threshold].index.min()

        # Filter rows where elevation is higher than 0 and lower than elevation threshold
        if pd.notna(index_threshold):
            df_filtered = df[(df['Elevation'] >= 0) & (df['Elevation'] <= elevation_threshold) & (df.index <= index_threshold)]
        else:
            df_filtered = df[(df['Elevation'] >= 0) & (df['Elevation'] <= elevation_threshold)]
        
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
    contour_filled = plt.contourf(grid_lon, grid_lat, bathy, levels=20, cmap="viridis")
    plt.colorbar(label="Topobathymetry [m]")

    # Contour lines with labels
    contour_lines = plt.contour(grid_lon, grid_lat, bathy, levels=10, colors='black', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.0f")  # Add labels to contours

    # Titles and labels
    plt.title("Click to define the shoreline (Right-click to Finish)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.gca().set_aspect('equal')

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

    ax.set_xlabel("Distance Along Transect [m]")
    ax.set_ylabel("Elevation [m]")
    ax.set_title("Elevation Profiles for Transects")
    ax.grid(True, linestyle="--", alpha=0.6)

    # # Optional: Show only a limited number of legend labels for clarity
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc="best", fontsize=8, title="Transects (sample)")

    plt.show(block=False)  # Keep the plot open