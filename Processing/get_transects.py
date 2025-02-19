#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import sys
import os
import pandas as pd

# Add the parent directory of `Functions` to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import functions
from Functions.Transect_processing import (
    manual_shoreline_definition,
    equidistant_points_on_curve,
    perpendicular_line,
    transect_processing,
    plot_topobathymetry_and_contours,
    find_contour_coordinates,
    plot_transect_elevations,
)

#--------------------------------------------------------
#
# Main function
#
#--------------------------------------------------------

def main(city, extension_length, distance, elevation_threshold):
    while True:
        # Load bathymetry dataset
        bathy_nc = xr.open_dataset(f'../Data/{city}/Bathymetry.nc')
        grid_lat = bathy_nc["lat"].values
        grid_lon = bathy_nc["lon"].values
        bathy = bathy_nc["bathy"].values

        shoreline = find_contour_coordinates(grid_lon, grid_lat, bathy, level=0)
        ## Remove values from the ends of the line
        shoreline = shoreline[10:-10]

        # User defines the shoreline
        smoothed_shoreline, _ = manual_shoreline_definition(
            grid_lon, grid_lat, bathy, shoreline, n_point=1000, smooth_window=20
        )

        # Compute equidistant points along the shoreline
        x_equidistant, y_equidistant = equidistant_points_on_curve(
            smoothed_shoreline[:, 0], smoothed_shoreline[:, 1], distance
        )

        # Generate perpendicular transects
        lines_latlon, lines_UTM = perpendicular_line(
            x_equidistant,
            y_equidistant,
            smoothed_shoreline,
            extension_length=extension_length,
            bathy=bathy_nc['bathy'],
            dummy_length=500
        )
        
        # Process transects into dictionary
        transect_dict = transect_processing(
            lines_latlon, lines_UTM, bathy_nc["bathy"], elevation_threshold
        )

        # Plot the bathymetry and transects
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))

        elev_min = round(bathy.min() / 10) * 10
        elev_max = round(bathy.max() / 10) * 10
        elev_delta = round((elev_max - elev_min) / 11)

        plot_topobathymetry_and_contours(
            grid_lon, grid_lat, bathy,
            elev_min=elev_min,
            elev_max=elev_max,
            elev_delta=elev_delta,
            z0_contour=shoreline,
            cmap='viridis',
            ax=axs
        )

        for df in transect_dict.values():
            line_lon = df['Lon']
            line_lat = df['Lat']
            axs.plot(line_lon, line_lat, 'r--')

        axs.scatter(x_equidistant, y_equidistant, c='k')
        axs.plot(
            smoothed_shoreline[:, 0], smoothed_shoreline[:, 1],
            ls='--', lw=2, c='y', label='Shoreline defined by User'
        )

        axs.set_xlim(grid_lon.min(), grid_lon.max())
        axs.set_ylim(grid_lat.min(), grid_lat.max())
        axs.legend()
        axs.set_aspect('equal')
        axs.set_title(f'Number of transects: {len(transect_dict)}')

        # Show plot
        plt.show(block=False)

        #Plot the elevation profiles for transects
        plot_transect_elevations(transect_dict)

        # Ask user whether to continue or exit
        user_input = input("\nAre you satisfied with the results? (y/n): ").strip().lower()
        if user_input == 'y':
            # Save transects as HDF5 file
            with pd.HDFStore(f'../Data/{city}/Transects_data.h5', mode='w') as store:
                for key, df in transect_dict.items():
                    store[key] = df
            print("\nFinalizing process...")
            break
        else:
            plt.close('all')
            print("\nRepeating process with new inputs...\n")

#--------------------------------------------------------
#
# RUN
#
#--------------------------------------------------------

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generate shoreline transects from bathymetry data.")
    parser.add_argument("--city", type=str, required=True, help="City to create the Path to the bathymetry NetCDF file")
    parser.add_argument("--extension_length", type=int, required=True, help="Length of the transects in meters")
    parser.add_argument("--distance", type=int, required=True, help="Distance between equidistant points")
    parser.add_argument("--elevation_threshold", type=int, required=True, help="Elevation threshold for filtering transects")

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.city, args.extension_length, args.distance, args.elevation_threshold)