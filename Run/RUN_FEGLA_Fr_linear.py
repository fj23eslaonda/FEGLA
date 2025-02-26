#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import os
import sys
import multiprocessing
import numpy as np
import pandas as pd
import gc
import json
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
# Add the parent directory of `Functions` to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Functions.FEGLA_engine import (
    transect_processing, 
    find_maxhorizontalflood,
    create_batches_from_combinations,
    load_batch_from_hdf5
    )

#--------------------------------------------------------
#
# MODEL ENGINE
#
#--------------------------------------------------------

g              = 9.81  # gravity constant
delta_x        = -1.0  # delta x for computations
tolerance      = 0.01  # error tolerance for convergence
max_iterations = 100   # maximum number of iterations allowed

#--------------------------------------------------------
# Wrapper function to parallelize transect processing
#--------------------------------------------------------
def process_transect_wrapper(args):
    idx, dataframes, F0, FR = args
    return process_transect(idx, dataframes, F0, FR)
#--------------------------------------------------------
# Load inputs
#--------------------------------------------------------
def load_params(json_file):
    with open(json_file, 'r') as file:
        params = json.load(file)
    return params
#--------------------------------------------------------
# Main function to process a single transect
#--------------------------------------------------------
def process_transect(idx, scen_tran_all, F0, FR):
    print('================')
    print('Processing :', idx)
    
    # Initialize iteration lists for each transect
    height_iteration = []
    all_XRmax        = []
    all_XRmin        = []
    error_iteration  = []

    flood_transect = scen_tran_all[idx]
    iteration      = 0

    # Extract initial variables
    z_initial          = flood_transect['elevation'].values
    manning_initial    = flood_transect['manning'].values
    h0                 = flood_transect['hmax'].values

    # Check if all hmax values are NaN (Transect Not Flooded)
    if np.all(np.isnan(h0)):
        print(f"Transect {idx} was NOT flooded. Returning NaN values.")
        return idx, {
            'height_iteration': np.full((1, len(z_initial)), np.nan, dtype=np.float32),
            'error_iteration': np.array([np.nan], dtype=np.float32),
            'all_XRmax': np.array([np.nan], dtype=np.float32),
            'all_XRmin': np.array([np.nan], dtype=np.float32),
            'R0': np.nan
        }

    # Continue processing if the transect was flooded
    distance_initial   = flood_transect['cum_distance'].values
    R0   = h0[0] + 0.5 * F0**2 * h0[0]

    # Find initial X_max and X_min
    X_max, _ = find_maxhorizontalflood(flood_transect, R0)
    X_min    = 0
    X_R      = np.mean([X_max, X_min])

    previous_error = None

    while iteration < max_iterations:
        print(f'iteration: {iteration}')
        print('----------------')

        # Recalculate N and dx based on XR
        # Calculate N so that dx is as close to 1 as possible
        N       = int(np.round(X_R))  
        delta_x = - X_R / N  # New dx based on XR and N

        # Create new distance array based on the new dx
        new_distance = np.linspace(0, X_R, N + 1)

        # Reinterpolate z and manning values based on the new distances
        z        = np.interp(new_distance, distance_initial, z_initial)
        manning  = np.interp(new_distance, distance_initial, manning_initial)

        distance_dummy  = new_distance[new_distance <= X_R]

        Fr   = F0 + (FR-F0)*distance_dummy/X_R

        h_opti_list     = np.full(distance_dummy.size, np.nan)
        velocity_list   = np.full(distance_dummy.size, np.nan)
        h_opti_list[-1] = 0  # Initial boundary condition

        # Iterate from right (runup) to left (towards the coast)
        for i in range(len(distance_dummy) - 1, 0, -1):
            z_next, z_prev      = z[i - 1], z[i]
            h_prev              = h_opti_list[i]
            Fr_next, Fr_prev    = Fr[i - 1], Fr[i]
            manning_coeff       = manning[i - 1]

            u_prev           = Fr_prev * np.sqrt(g * h_prev) if h_prev >= 0 else 0
            velocity_list[i] = u_prev

            ## Energy and loss approximation
            energy_i = z_prev + h_prev + 1/2 * Fr_prev**2*h_prev
            loss_i   = (g * Fr_next**2 * manning_coeff**2 * delta_x) / (h_prev**(1/3)) if h_prev > 0 else 0
            h_next   = 1 / (1 + 1/2 * Fr_next**2) * (energy_i - loss_i - z_next)

            # Ensure h_next is not negative
            h_next = max(h_next, 0)
            
            h_opti_list[i - 1] = h_next
            velocity_list[i - 1] = Fr_next * np.sqrt(g * h_next) if h_next >= 0 else 0

        # Calculate error and adjust X_R
        h0_opti  = h_opti_list[0]
        h0_error = np.abs(h0_opti - h0[0]) / h0[0]

        # Save error and handle iteration adjustments
        error_iteration.append(h0_error)

        if previous_error is not None:
            error_diff = np.abs(previous_error - h0_error)
            if error_diff < 0.001:
                print("Adjusting R0 by 10% due to small error change.")
                R0 *= 1.1  
                X_max, _  = find_maxhorizontalflood(flood_transect, R0)
                X_min     = 0
                X_R       = np.mean([X_max, X_min])

        previous_error = h0_error  # Update previous error for the next iteration
        
        # Save iteration values
        height_iteration.append(np.array(h_opti_list.copy(), dtype=np.float32))
        all_XRmax.append(X_max)
        all_XRmin.append(X_min)

        # Update final results with the latest iteration values
        print(f'h0_error: {h0_error:.5f}, h0: {h0[0]:.5f}, h0_opti:{h0_opti:.5f}')
        print(f'XR_max : {X_max}')
        print(f'XR_min : {X_min}')
        print(f'XR: {X_R}')

        if np.round(h0_error,2) <= tolerance:
            break
        if h0_opti < h0[0]:
            X_min = X_R
        else:
            X_max = X_R

        iteration += 1
        X_R = 0.5 * (X_max + X_min)

    # Return all the results
    return idx, {
        'height_iteration': height_iteration[-1] if height_iteration else np.full((1, len(z_initial)), np.nan, dtype=np.float32),
        'error_iteration': np.array(error_iteration, dtype=np.float32),
        'all_XRmax': np.array(all_XRmax, dtype=np.float32),
        'all_XRmin': np.array(all_XRmin, dtype=np.float32),
        'R0': R0
    }


#--------------------------------------------------------
# Main function to process all transects in parallel
#--------------------------------------------------------
def main(params, scenarios_to_run=None, transect_to_run=None):
    city       = params['city']
    filepath   = params['filepath']  
    outpath    = Path(params['outpath'])   
    batch_size = params['batch_size'] 
    manning    = params['manning']
    F02try     = params['F0']
    FR2try     = params['FR']

    # Load the dictionary of dataframes from the HDF5 file
    with pd.HDFStore(filepath, mode='r') as store:
        all_keys = [key.strip('/') for key in store.keys()]  # Get all keys from the HDF5 file, without the leading '/'

    # Extract existing combinations of scenarios and transects
    available_combinations = sorted(all_keys)

    # Filter the available combinations based on scenarios_to_run and transect_to_run
    if scenarios_to_run is not None:
        available_combinations = [key for key in available_combinations if key.split('_')[0] in scenarios_to_run]
    
    if transect_to_run is not None:
        available_combinations = [key for key in available_combinations if key.split('_')[1] in transect_to_run]

    if not available_combinations:
        raise ValueError("No valid combinations of scenarios and transects found based on the provided filters.")

    # Create batches based on the existing valid combinations
    batch_list = create_batches_from_combinations(available_combinations, batch_size)

    for FR in FR2try:
        for F0 in F02try:
            print(f"Run code for F0: {F0} and FR: {FR}")
            all_results = dict()  # Initialize an empty dictionary to store results

            for batch in tqdm(batch_list, desc="Processing batches", total=len(batch_list)):
                # Load data for the current batch
                dataframes = load_batch_from_hdf5(filepath, batch)
                idx_batch  = list(dataframes.keys())

                # Process the transects
                dataframes = transect_processing(dataframes, idx_batch, manning_coeff=manning)

                # Parallel processing of transects
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    args = [(idx, dataframes, F0, FR) for idx in idx_batch]
                    results = pool.map(process_transect_wrapper, args, chunksize=12)

                # Collect results for each scenario
                for idx, res in results:
                    if res is not None:
                        all_results[idx] = res  # Store results in the dictionary
                
                del dataframes  # Delete the dataframes to free up memory
                gc.collect()  # Run garbage collection to free up memory
            
            results_folder = outpath / city
            # Create the folder if it doesn't exist
            results_folder.mkdir(parents=True, exist_ok=True)
            with open(results_folder / f'{city}_all_results_F0_{F0}_FR_{FR}_linear.pkl', 'wb') as f:
                pickle.dump(all_results, f)

#--------------------------------------------------------
# Main script entry point
#--------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the simulation with specified parameters.")
    parser.add_argument('--params', type=str, help="Path to the JSON parameters file.")
    args = parser.parse_args()

    # Load parameters from the JSON file
    params = load_params(args.params)
    # Execute main function
    # scenarios_to_run = ['S050'], only to run scenario N50
    # transects_to_run = ['T040'], only to run transect T40
    main(params)#, scenarios_to_run=['S001'], transect_to_run=['T008'])
