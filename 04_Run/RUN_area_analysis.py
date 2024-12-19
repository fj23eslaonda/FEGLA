#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent 
sys.path.append(str(current_dir.parent))
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from Functions.FEGLA_engine import transect_processing#, find_maxhorizontalflood, find_steeper_slope
from Functions.FEGLA_processing import (
    load_height_from_simulations, 
    find_contour_coordinates,
    create_spline_function,
    plot_topobathymetry_and_contours,
    get_boundary_points
    )
import gc
import seaborn as sns
import argparse
import pickle
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

#--------------------------------------------------------
#
# Load data
#
#--------------------------------------------------------
city = 'Coquimbo'

#--------------------------------------------------------
## Load height from simulations
scen_tran_all = load_height_from_simulations(f'../Data/{city}/Height_for_simulation.h5')
scen_tran_all = transect_processing(scen_tran_all, list(scen_tran_all.keys()), 0.03, 30)

#--------------------------------------------------------
## Load height from FEGLA
main_path    = Path.cwd() / 'Results'
results_list = list(main_path.glob(f'*{city}*F0_0.9_FR_0.5_linear*'))

results_dic = {}
for results in results_list:
    name = str(results).split('_')[-2]
    with open(results, 'rb') as f:
        data = pickle.load(f)
    results_dic[name] = data

#--------------------------------------------------------
## Load bathy
bathy_nc = xr.open_dataset(f'../Data/{city}/BaquedanoBathy.nc')

## longitude and latitude
grid_lon = bathy_nc['grid_lon'].values - 360
grid_lat = bathy_nc['grid_lat'].values
## Bathy
bathy    = bathy_nc['BaquedanoBathy'].values * -1
## Mesh
mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

## shoreline coordinate
shoreline = find_contour_coordinates(grid_lon, grid_lat, bathy, level=0)
## Shoreline approximation by spline
spline_function_x, spline_function_y, smoothed_points = create_spline_function(shoreline, 'cubic', 100, reduction_factor=40, smooth_window=10)

hmax_local_nc = xr.open_dataset(f'../Data/{city}/Maps_TodoFlowDepth_mw_9.1Baquedano.nc')
# Plot parameters
hmin, hmax = 0, 10
height_interval = 1


#--------------------------------------------------------
#
# Data Processing
#
#--------------------------------------------------------
for key, DFresults in results_dic.items():
    maxScen = 50
    #scenList = [f'S{str(ix).zfill(3)}' for ix in range(1, maxScen+1)]
    scenList = ['S015']

    for scen in tqdm(scenList, total=len(scenList), desc='Computing...'):
        heightList       = list()
        
        resultsScen = {key: value for key, value in DFresults.items() if key.startswith(scen)}
        dataScen    = {key: value for key, value in scen_tran_all.items() if key.startswith(scen)}

        ##----------------------------------------------------------
        ## AREA ESTIMATION
        ##----------------------------------------------------------

        for key, df in resultsScen.items():
            heightList.append(df['height_iteration'][-1])

        try:
            polygonSim, aream2Sim, polygonFEGLA, aream2FEGLA = get_boundary_points(dataScen, heightList)
        

            # Create the plot with a reasonable figure size
            fig1, ax1 = plt.subplots(figsize=(12, 10))

            # Plot topobathymetry and shoreline
            plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy,
                                                elev_min=-90,
                                                elev_max=240,
                                                elev_delta=30,
                                                z0_contour=shoreline,
                                                cmap='gray', ax=ax1)
            
            # Create filled contour plot for height

            # cf_height = ax.contourf(grid_lon, grid_lat, hmax_local_nc['FlowDepth'].values[int(scenList[0][1:])-1], 
            #                         levels=np.arange(hmin, hmax, height_interval), 
            #                         cmap='viridis', extend='both', alpha=1)
            #cbar_ax_height = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            #cbar_height = fig.colorbar(cf_height, cax=cbar_ax_height)
            #cbar_height.set_label('Height (m)')
            
            # Plot flood map boundary
            xSim, ySim = polygonSim.exterior.xy
            ax1.add_patch(MplPolygon(np.c_[xSim, ySim], closed=True, edgecolor='cyan', fill=False, linewidth=2, label='HySEA'))
            ax1.text(0.05, 0.1, f"HySEA Area: {aream2Sim/1e6:.2f} km²", transform=ax1.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
            
            xFEGLA, yFEGLA = polygonFEGLA.exterior.xy
            ax1.add_patch(MplPolygon(np.c_[xFEGLA, yFEGLA], closed=True, edgecolor='m', fill=False, linewidth=2, label='FEGLA'))
            ax1.text(0.05, 0.15, f"FEGLA Area: {aream2FEGLA/1e6:.2f} km²", transform=ax1.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
            
            # Display the plot
            ax1.set_aspect('equal', adjustable='box')
            ax1.legend()
            
            plt.savefig(f'./Figures/Coquimbo_Areas_FEGLA_HYSEA_{scen}.png', dpi=300, bbox_inches='tight', pad_inches=0)
            #plt.close(fig1)
        except:
            pass
        #plt.show()

        ## HERE HEIGHT MAPS ARE CODING

        latList, lonList          = list(), list()
        heightFEGLA, heightSim  = list(), list()

        for jx, (key, df) in enumerate(dataScen.items()):
            minlen   = np.min([len(heightList[jx]), len(df['height'].dropna().values)])
            #Remain minimum value
            DFdummy = df.iloc[:minlen].copy()

            latList.append(DFdummy['lat'].values)
            lonList.append(DFdummy['lon'].values)
            heightFEGLA.append(heightList[jx][:minlen])
            heightSim.append(DFdummy['height'].values)
        
        heightError    = np.concatenate(heightSim)-np.concatenate(heightFEGLA)/np.concatenate(heightSim)
        heightErrorAbs = np.abs(heightError)
        percenHeight = np.nanpercentile(heightError, [5, 50, 95])

        fig2, ax2 = plt.subplots(figsize=(12, 10))

        # Plot topobathymetry and shoreline
        plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy,
                                            elev_min=-90,
                                            elev_max=240,
                                            elev_delta=30,
                                            z0_contour=shoreline,
                                            cmap='gray', ax=ax2)
        
        ax2.plot(df['lon'], df['lat'], c='red', ls='--', lw=0.5, alpha=0.5)

        dfHeight = pd.DataFrame({'lon': np.concatenate(lonList),
                                 'lat': np.concatenate(latList), 
                                 'heightError': heightError})
        
        if not dfHeight['heightError'].isna().all():
            scatter=sns.scatterplot(x='lon', y='lat', hue='heightError', palette='viridis', 
                            data=dfHeight, legend=None, s=2, edgecolor='none', ax=ax2)
                # Create a colorbar
            norm = Normalize(vmin=0, vmax=15)
            sm = ScalarMappable(norm=norm, cmap='viridis')
            sm.set_array([])  # Only needed for ScalarMappable

            # Add the colorbar to the figure
            cbar = fig2.colorbar(sm, ax=ax2)
            cbar.set_label('Height Error (%)')
        else:
            sns.scatterplot(x='lon', y='lat', data=dfHeight, legend=None, s=2, edgecolor='none')
        
        ax2.add_patch(MplPolygon(np.c_[xSim, ySim], closed=True, edgecolor='cyan', fill=False, linewidth=2, label='HySEA'))
        ax2.text(0.05, 0.1, f"HySEA Area: {aream2Sim/1e6:.2f} km²", transform=ax2.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        ax2.add_patch(MplPolygon(np.c_[xFEGLA, yFEGLA], closed=True, edgecolor='m', fill=False, linewidth=2, label='FEGLA'))
        ax2.text(0.05, 0.15, f"FEGLA Area: {aream2FEGLA/1e6:.2f} km²", transform=ax2.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        # Display the plot
        #ax2.set_aspect('equal', adjustable='box')
        ax2.legend()
        plt.savefig(f'./Figures/Coquimbo_HeightError_FEGLA_HYSEA_{scen}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.close(fig2)

        #plt.show()

        
        # Create a new figure with two horizontal subplots
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

        # Plot histogram for heightError
        ax3.hist(heightError, bins=50, color='black', alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Height Error')
        ax3.set_xlabel('Height Error (%)')
        ax3.set_ylabel('Frequency')

        # Add vertical lines for percentiles
        ax3.axvline(percenHeight[0], color='red', linestyle='--', linewidth=2, label='5th Percentile')
        ax3.axvline(percenHeight[1], color='red', linestyle='--', linewidth=2, label='50th Percentile (Median)')
        ax3.axvline(percenHeight[2], color='red', linestyle='--', linewidth=2, label='95th Percentile')
        print(f"Percentiles: 5th = {percenHeight[0]}, 50th = {percenHeight[1]}, 95th = {percenHeight[2]}")
        # Add a legend for the percentiles
        ax3.legend(loc='upper left')

        # Plot histogram for heightErrorAbs
        #ax4.hist(np.abs(heightError), bins=30, color='green', alpha=0.7, edgecolor='black')
        #ax4.set_title('Distribution of Absolute Height Error')
        #ax4.set_xlabel('Absolute Height Error (%)')
        #ax4.set_ylabel('Frequency')

        # Adjust layout for better readability
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'./Figures/HeightError_Histograms_{scen}.png', dpi=300, bbox_inches='tight', pad_inches=0)

    

        # Show the figure
       # plt.show()
                

        fig4, ax4 = plt.subplots(figsize=(12, 10))

        # Plot topobathymetry and shoreline
        plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy,
                                            elev_min=-90,
                                            elev_max=240,
                                            elev_delta=30,
                                            z0_contour=shoreline,
                                            cmap='gray', ax=ax4)
        
        ax4.plot(df['lon'], df['lat'], c='red', ls='--', lw=0.5, alpha=0.5)

        dfHeight = pd.DataFrame({'lon': np.concatenate(lonList),
                                 'lat': np.concatenate(latList), 
                                 'heightError': np.concatenate(heightSim)})
        
        if not dfHeight['heightError'].isna().all():
            scatter=sns.scatterplot(x='lon', y='lat', hue='heightError', palette='viridis', 
                            data=dfHeight, legend=None, s=2, edgecolor='none')
                # Create a colorbar
            norm = Normalize(vmin=percenHeight[0], vmax=percenHeight[2])
            sm = ScalarMappable(norm=norm, cmap='viridis')
            sm.set_array([])  # Only needed for ScalarMappable

            # Add the colorbar to the figure
            cbar = fig4.colorbar(sm, ax=ax4)
            cbar.set_label('Height Error (%)')
        else:
            sns.scatterplot(x='lon', y='lat', data=dfHeight, legend=None, s=2, edgecolor='none')
        
        ax4.add_patch(MplPolygon(np.c_[xSim, ySim], closed=True, edgecolor='cyan', fill=False, linewidth=2, label='HySEA'))
        ax4.text(0.05, 0.1, f"HySEA Area: {aream2Sim/1e6:.2f} km²", transform=ax4.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        ax4.add_patch(MplPolygon(np.c_[xFEGLA, yFEGLA], closed=True, edgecolor='m', fill=False, linewidth=2, label='FEGLA'))
        ax4.text(0.05, 0.15, f"FEGLA Area: {aream2FEGLA/1e6:.2f} km²", transform=ax4.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        # Display the plot
        #ax2.set_aspect('equal', adjustable='box')
        ax4.legend()
        plt.savefig(f'./Figures/Coquimbo_Heigh_HYSEA_{scen}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.close(fig2)






        fig5, ax5 = plt.subplots(figsize=(12, 10))

        # Plot topobathymetry and shoreline
        plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy,
                                            elev_min=-90,
                                            elev_max=240,
                                            elev_delta=30,
                                            z0_contour=shoreline,
                                            cmap='gray', ax=ax5)
        
        ax5.plot(df['lon'], df['lat'], c='red', ls='--', lw=0.5, alpha=0.5)

        dfHeight = pd.DataFrame({'lon': np.concatenate(lonList),
                                 'lat': np.concatenate(latList), 
                                 'heightError': np.concatenate(heightFEGLA)})
        
        if not dfHeight['heightError'].isna().all():
            scatter=sns.scatterplot(x='lon', y='lat', hue='heightError', palette='viridis', 
                            data=dfHeight, legend=None, s=2, edgecolor='none')
                # Create a colorbar
            norm = Normalize(vmin=percenHeight[0], vmax=percenHeight[2])
            sm = ScalarMappable(norm=norm, cmap='viridis')
            sm.set_array([])  # Only needed for ScalarMappable

            # Add the colorbar to the figure
            cbar = fig5.colorbar(sm, ax=ax5)
            cbar.set_label('Height Error (%)')
        else:
            sns.scatterplot(x='lon', y='lat', data=dfHeight, legend=None, s=2, edgecolor='none')
        
        ax5.add_patch(MplPolygon(np.c_[xSim, ySim], closed=True, edgecolor='cyan', fill=False, linewidth=2, label='HySEA'))
        ax5.text(0.05, 0.1, f"HySEA Area: {aream2Sim/1e6:.2f} km²", transform=ax5.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        ax5.add_patch(MplPolygon(np.c_[xFEGLA, yFEGLA], closed=True, edgecolor='m', fill=False, linewidth=2, label='FEGLA'))
        ax5.text(0.05, 0.15, f"FEGLA Area: {aream2FEGLA/1e6:.2f} km²", transform=ax5.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        # Display the plot
        #ax2.set_aspect('equal', adjustable='box')
        ax5.legend()
        plt.savefig(f'./Figures/Coquimbo_Heigh_FEGLA_{scen}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.close(fig2)



                


                





#         ## Compute optimized Height map
#         grid_x_for_scenario, grid_y_for_scenario, grid_z_for_scenario, area_for_scenario = compute_Fr_and_flood_areas(scen_idx,
#                                                                                                                     points_for_scenario,
#                                                                                                                     boundary_for_scenario,
#                                                                                                                     h_geoclaw=False,
#                                                                                                                     Fr = False,
#                                                                                                                     surface=True)

#         ## Compute GEOCLAW Height map
#         geo_grid_x_for_scenario, geo_grid_y_for_scenario, geo_grid_z_for_scenario, geo_area_for_scenario = compute_Fr_and_flood_areas(scen_idx,
#                                                                                                                                     points_for_scenario,
#                                                                                                                                     boundary_for_scenario,
#                                                                                                                                     h_geoclaw=True,
#                                                                                                                                     Fr = False,
#                                                                                                                                     surface=True)


# #print(results_dic.keys())
# #print('\n')
# #print(results_dic['0.5'].keys())
# print(results_dic['0.5']['S042_T079'].keys())




# points_for_scenario, boundary_for_scenario = get_Fr_points_and_boundary_for_scenario(tran_idx, scen_idx, scen_tran_all, all_results)

# ## Compute optimized Height map
# grid_x_for_scenario, grid_y_for_scenario, grid_z_for_scenario, area_for_scenario = compute_Fr_and_flood_areas(scen_idx,
#                                                                                                               points_for_scenario,
#                                                                                                               boundary_for_scenario,
#                                                                                                               h_geoclaw=False,
#                                                                                                               Fr = False,
#                                                                                                               surface=True)

# ## Compute GEOCLAW Height map
# geo_grid_x_for_scenario, geo_grid_y_for_scenario, geo_grid_z_for_scenario, geo_area_for_scenario = compute_Fr_and_flood_areas(scen_idx,
#                                                                                                                               points_for_scenario,
#                                                                                                                               boundary_for_scenario,
#                                                                                                                               h_geoclaw=True,
#                                                                                                                               Fr = False,
#                                                                                                                               surface=True)

# ## Compute Optimized Froude map
# Fr_grid_x_for_scenario, Fr_grid_y_for_scenario, Fr_grid_z_for_scenario, Fr_area_for_scenario = compute_Fr_and_flood_areas(scen_idx,
#                                                                                                                           points_for_scenario,
#                                                                                                                           boundary_for_scenario,
#                                                                                                                           h_geoclaw=False,
#                                                                                                                           Fr = True,
#                                                                                                                           surface=True)





# s_ix = 'S018'
# geo_boundary_x, geo_boundary_y = boundary_for_scenario[s_ix]
# geo_grid_x = geo_grid_x_for_scenario[s_ix]
# geo_grid_y = geo_grid_y_for_scenario[s_ix]
# geo_grid_z = geo_grid_z_for_scenario[s_ix]
# geo_area   = geo_area_for_scenario[s_ix]

# city = 'Coquimbo'
# plot_surface_and_area_for_scenario(grid_lon, grid_lat, bathy, shoreline,
#                                    geo_boundary_x, geo_boundary_y, geo_grid_x, geo_grid_y, geo_grid_z, geo_area,
#                                    city, surface=True, cmap_scale=[0, 15, 1])



# s_ix = 'S012'
# boundary_x, boundary_y = boundary_for_scenario[s_ix]
# grid_x = grid_x_for_scenario[s_ix]
# grid_y = grid_y_for_scenario[s_ix]
# grid_z = grid_z_for_scenario[s_ix]
# area   = area_for_scenario[s_ix]

# city = 'Coquimbo'
# plot_surface_and_area_for_scenario(grid_lon, grid_lat, bathy, shoreline,
#                                    boundary_x, boundary_y, grid_x, grid_y, grid_z, area, city,
#                                    surface=True, cmap_scale=[0, 15, 1])
                                                                                                                            