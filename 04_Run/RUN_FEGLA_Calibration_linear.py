#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import pandas as pd
from tqdm import tqdm
from Functions.FEGLA_engine import transect_processing
from Functions.FEGLA_processing import load_height_from_simulations, load_results_for_iteration, remove_nan_values
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#--------------------------------------------------------
#
# LOAD DATA
#
#--------------------------------------------------------
city = 'Cartagena'

scen_tran_all = load_height_from_simulations(f'../Data/{city}/Height_for_simulation.h5')
scen_tran_all = transect_processing(scen_tran_all, list(scen_tran_all.keys()), 0.03, 30)

main_path    = Path.cwd() / 'Results'
results_list = list(main_path.glob(f'*{city}*linear*'))

results_dic = {}
for results in results_list:
    name = str(results).split('_')[-2]
    with open(results, 'rb') as f:
        data = pickle.load(f)
    results_dic[name] = data

#--------------------------------------------------------
#
# RMSE COMPUTING
#
#--------------------------------------------------------
rmse_Fr08 = []
rmse_Fr09 = []
rmse_Fr10 = []
rmse_Fr11 = []
rmse_Fr12 = []
rmse_Fr13 = []
rmse_Fr14 = []
rmse_Fr15 = []

for key in tqdm(scen_tran_all.keys()):
    height_sim = scen_tran_all[key]['height'].values

    height_Fr08 = results_dic['Fr0.8'][key]['height_iteration'][-1]
    len_Fr08 = np.min([len(height_sim), len(height_Fr08)])
    h_sim_Fr08, h_Fr08 = remove_nan_values(height_sim[:len_Fr08], height_Fr08[:len_Fr08])
    rmse_Fr08.append(np.sqrt(mean_squared_error(h_sim_Fr08, h_Fr08)))

    height_Fr09 = results_dic['Fr0.9'][key]['height_iteration'][-1]
    len_Fr09 = np.min([len(height_sim), len(height_Fr09)])
    h_sim_Fr09, h_Fr09 = remove_nan_values(height_sim[:len_Fr09], height_Fr09[:len_Fr09])
    rmse_Fr09.append(np.sqrt(mean_squared_error(h_sim_Fr09, h_Fr09)))

    height_Fr10 = results_dic['Fr1.0'][key]['height_iteration'][-1]
    len_Fr10 = np.min([len(height_sim), len(height_Fr10)])
    h_sim_Fr10, h_Fr10 = remove_nan_values(height_sim[:len_Fr10], height_Fr10[:len_Fr10])
    rmse_Fr10.append(np.sqrt(mean_squared_error(h_sim_Fr10, h_Fr10)))

    height_Fr11 = results_dic['Fr1.1'][key]['height_iteration'][-1]
    len_Fr11 = np.min([len(height_sim), len(height_Fr11)])
    h_sim_Fr11, h_Fr11 = remove_nan_values(height_sim[:len_Fr11], height_Fr11[:len_Fr11])
    rmse_Fr11.append(np.sqrt(mean_squared_error(h_sim_Fr11, h_Fr11)))

    height_Fr12 = results_dic['Fr1.2'][key]['height_iteration'][-1]
    len_Fr12 = np.min([len(height_sim), len(height_Fr12)])
    h_sim_Fr12, h_Fr12 = remove_nan_values(height_sim[:len_Fr12], height_Fr12[:len_Fr12])
    rmse_Fr12.append(np.sqrt(mean_squared_error(h_sim_Fr12, h_Fr12)))

    height_Fr13 = results_dic['Fr1.3'][key]['height_iteration'][-1]
    len_Fr13 = np.min([len(height_sim), len(height_Fr13)])
    h_sim_Fr13, h_Fr13 = remove_nan_values(height_sim[:len_Fr13], height_Fr13[:len_Fr13])
    rmse_Fr13.append(np.sqrt(mean_squared_error(h_sim_Fr13, h_Fr13)))

    height_Fr14 = results_dic['Fr1.4'][key]['height_iteration'][-1]
    len_Fr14 = np.min([len(height_sim), len(height_Fr14)])
    h_sim_Fr14, h_Fr14 = remove_nan_values(height_sim[:len_Fr14], height_Fr14[:len_Fr14])
    rmse_Fr14.append(np.sqrt(mean_squared_error(h_sim_Fr14, h_Fr14)))

    height_Fr15 = results_dic['Fr1.5'][key]['height_iteration'][-1]
    len_Fr15 = np.min([len(height_sim), len(height_Fr15)])
    h_sim_Fr15, h_Fr15 = remove_nan_values(height_sim[:len_Fr15], height_Fr15[:len_Fr15])
    rmse_Fr15.append(np.sqrt(mean_squared_error(h_sim_Fr15, h_Fr15)))

#--------------------------------------------------------
#
# PLOT
#
#--------------------------------------------------------
data = [rmse_Fr08, rmse_Fr09, rmse_Fr10, rmse_Fr11, 
        rmse_Fr12, rmse_Fr13, rmse_Fr14, rmse_Fr15]

means = [np.mean(dat) for dat in data]
std_devs = [np.std(dat) for dat in data]

plt.rc('font', size=13)
plt.figure(figsize=(12, 6))
plt.errorbar(x=[f'{0.8+0.1*i:.2f}' for i in range(8)], y=means, yerr=std_devs, fmt='o', capsize=5, color='blue', ecolor='black', elinewidth=2, markeredgewidth=2)
plt.xlim(-0.5, 7.5)
#plt.ylim(0, 2)

for i, mean in enumerate(means):
    plt.text(i-0.35, mean, f'Media: {mean:.3f} \n Std: {std_devs[i]:.2f}', ha='left', va='bottom', rotation=90)

plt.xlabel('$F_0$')
plt.ylabel('RMSE [m]')
plt.grid(True)
#plt.savefig('Figures/Calibracion_FR_linear.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()