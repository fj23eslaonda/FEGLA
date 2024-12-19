#--------------------------------------------------------
#
#   Packages
#
#--------------------------------------------------------
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Functions.Processing import (
    find_contour_coordinates, 
    create_spline_function,
    plot_topobathymetry_and_contours,
    equidistant_points_on_curve,
    perpendicular_line,
    create_interpolation_function,
    evaluate_transect,
    elevation_filter
)
import cProfile
import pstats
#--------------------------------------------------------
#
#   Loading data
#
#--------------------------------------------------------
def main():
    ## Load bathy
    bathy_nc = xr.open_dataset('../Data/Coquimbo/BaquedanoBathy.nc')

    ## longitude and latitude
    grid_lon = bathy_nc['lon'].values - 360
    grid_lat = bathy_nc['lat'].values
    ## Bathy, negative is associated to bathy.
    bathy    = bathy_nc['bathy'].values*-1
    ## Mesh
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    ## shoreline coordinate
    shoreline = find_contour_coordinates(grid_lon, grid_lat, bathy, level=0)
    ## Shoreline approximation by spline
    _, _, smoothed_points = create_spline_function(shoreline, 'cubic', 
                                                   100, reduction_factor=40, 
                                                    smooth_window=10)

    #--------------------------------------------------------
    #
    #   Creating transects
    #
    #--------------------------------------------------------
    ## Dict to storage transects
    transects_new = dict()

    ## Transect setup
    extension_length   = 5000 
    n_points_transect  = extension_length
    distance           = 50
    # Use average latitude for conversion to meters
    avg_lat            = np.mean(smoothed_points[:, 1])
    elev               = 30

    ## Equidistant points on shoreline curve
    x_equidistant, y_equidistant = equidistant_points_on_curve(shoreline[:,0], shoreline[:,1], distance)
    ## Perpendicular line at intersected point
    ## Units: meters UTM 
    initial_points, end_points = perpendicular_line(x_equidistant, y_equidistant, smoothed_points, extension_length)
    ## Bathy interpolation
    maps                   = [mesh_lon, mesh_lat, bathy]
    interpolation_function = create_interpolation_function(maps)

    ## Interpolating each transect into bathy
    for ix, (initial_point, end_point) in tqdm(enumerate(zip(initial_points, end_points)), total=len(initial_points)):
        ## Transect interpolation using bathy
        deltax_lon, deltay_lat, deltax, deltay, elevation = evaluate_transect(initial_point, end_point,
                                                    extension_length, n_points_transect, interpolation_function, avg_lat)
        ## Create dataframe for transect to storage
        df_transect_interp = pd.DataFrame(zip(deltax_lon, deltay_lat, deltax, deltay, elevation), columns=['lon','lat', 'x', 'y','Elevation'])
        ## Filter for elevation
        df_transect_interp = df_transect_interp[df_transect_interp['Elevation']<=elev]
        ## Save new transect
        transects_new[f'T{str(ix+1).zfill(3)}'] = df_transect_interp

    #--------------------------------------------------------
    #
    #   Computing slope per transect
    #
    #--------------------------------------------------------
    ## Apply elevation filter
    transects_new = {nombre: elevation_filter(df, elev) for nombre, df in transects_new.items()}
    
    ## Computing slopes
    slopes = []
    for ix, df in transects_new.items():
        maxelev = df['Elevation'][:50].max()
        slopes.append(maxelev/50)

    # Calculatin percentile
    p5 = np.percentile(slopes, 5)
    p50 = np.percentile(slopes, 50)
    p95 = np.percentile(slopes, 95)

    #--------------------------------------------------------
    #
    #   Save transects as .h5 file
    #
    #--------------------------------------------------------
    # Guardar el diccionario en un archivo HDF5
    Ntransect = len(transects_new.keys())
    with pd.HDFStore(f'../Data/Coquimbo/Transects_N{Ntransect}_LonLat.h5') as store:
        for key, df in transects_new.items():
            store[key] = df

    #--------------------------------------------------------
    #
    #   Bathy plot
    #
    #--------------------------------------------------------
    # Plot topobathy and shoreline
    fig, ax = plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy, 
                                            elev_min   = -150, 
                                            elev_max   = 60, 
                                            elev_delta = 15, 
                                            z0_contour = shoreline,
                                            cmap       = 'Spectral')
                                            
    ## plotting shoreling
    ax.plot(smoothed_points[:,0], smoothed_points[:,1], 
            ls='--', lw=2, c='black', label='Shoreline spline')

    x_equidistant, y_equidistant = equidistant_points_on_curve(shoreline[:,0], shoreline[:,1], distance=500)
    ax.scatter(x_equidistant, y_equidistant)
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    #--------------------------------------------------------
    #
    #   Bathy and transects plot
    #
    #--------------------------------------------------------
    ## Plot setup
    fig, axs = plt.subplots(1,1, figsize=(12,10))

    plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy, 
                                    elev_min   = -90, 
                                    elev_max   = 240, 
                                    elev_delta = 30, 
                                    z0_contour = shoreline,
                                    cmap       = 'Spectral',
                                    ax         = axs)


    for line_start, line_end in zip(initial_points, end_points):

        # Convert lon/lat to meters
        lat_to_meters = 111000  # 111 km per degree
        lon_to_meters = 111000 * np.cos(np.radians(avg_lat))
        plt.plot([line_start[0] / lon_to_meters, line_end[0] / lon_to_meters],
                [line_start[1] / lat_to_meters, line_end[1] / lat_to_meters], 'r--')

    #axs.set_aspect('equal', adjustable='box')
    axs.set_xlim(grid_lon.min(), grid_lon.max())
    axs.set_ylim(grid_lat.min(), grid_lat.max())
    axs.legend()
    ax.set_aspect('equal')
    plt.show()

    #--------------------------------------------------------
    #
    #   Slope histrogram
    #
    #--------------------------------------------------------
    # Crear el histograma usando seaborn
    plt.figure(figsize=(6, 6))
    sns.histplot(slopes, bins=30, kde=False, edgecolor='black')
    plt.axvline(p5, color='red', linestyle='dashed', linewidth=1.5, label='P5')
    plt.axvline(p50, color='red', linestyle='-', linewidth=1.5, label='P50')
    plt.axvline(p95, color='red', linestyle='dashed', linewidth=1.5, label='P95')
    plt.xlabel('Slopes')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # main()  # Llama a tu función principal
    
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs()  # Opcional: limpia rutas de archivos
    # stats.sort_stats("cumulative")  # Ordena por tiempo acumulado
    # stats.print_stats(20) 