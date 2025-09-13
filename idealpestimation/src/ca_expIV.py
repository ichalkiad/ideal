import os 

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["NUMBA_NUM_THREADS"] = "20"

import ipdb
import pathlib
import jsonlines
import numpy as np
import random
from idealpestimation.src.utils import pickle, \
                                        time, timedelta, \
                                            rank_and_plot_solutions, print_threadpool_info, \
                                                clean_up_data_matrix, load_matrix
from idealpestimation.src.efficiency_monitor import Monitor
from idealpestimation.src.ca import CA_custom, do_correspondence_analysis
import pandas as pd



def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, temperature_rate=[0, 1], temperature_steps=[1e-3], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, country=None, diff_iter=None, disp=False,
        theta_true=None, year=None, min_sigma_e=None, fastrun=False,
        max_restarts=2, max_partial_restarts=2, max_halving=2, plot_online=False, seedint=None):
        

        DIR_out = "{}/{}/estimation_CA_{}/".format(data_location, country, year)
        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
                
        # load data            
        mappings, node_to_index_start, index_to_node_start, \
                    node_to_index_end, index_to_node_end, Y = load_matrix("{}/Y_{}_{}".format(data_location, country, year), K, J)                
        Y = Y.astype(np.int8).reshape((K, J), order="F")    

        ca = CA_custom(
            n_components=d,
            n_iter=2,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=seedint
        )
     
        
        rows_all_le_1 = np.array((Y.max(axis=1) < 1).toarray().ravel())
        k_idx = np.argwhere(rows_all_le_1).ravel()
        cols_all_le_1 = np.array((Y.max(axis=0) < 1).toarray().ravel())
        j_idx = np.argwhere(cols_all_le_1).ravel()
        if len(j_idx) > 0:
            Y_new = Y[:, j_idx]
        else:
            Y_new = Y
        if len(k_idx) > 0:
            Y_new = Y_new[k_idx, :]
        
        K_new = K - len(k_idx.flatten())
        J_new = J - len(j_idx.flatten())
        # try:
        #     assert K == K_new
        #     assert J == J_new
        # except:
        #     print(country)        
        # return 

        parameter_space_dim_new = (K_new+J_new)*d + J_new + K_new + 2
        Y = Y_new
        Y = Y.todense()
        if country == "us":
            y_idx_subsample = np.random.choice(np.arange(0, Y.shape[0]), size=int(np.round(0.5*Y.shape[0])))            
            Y = Y[y_idx_subsample, :]
            K = Y.shape[0]
            # note that if Users IDs are needed, they must be stored here and retrieved when needed - subsampling changes order
            parameter_space_dim = (K+J)*d + J + K + 2
            print(Y.shape)

        param_positions_dict = dict()            
        k = 0
        for param in parameter_names:
            if param == "X":
                param_positions_dict[param] = (k, k + K*d)                       
                k += K*d    
            elif param in ["Z"]:
                param_positions_dict[param] = (k, k + J*d)                                
                k += J*d
            elif param in ["Phi"]:            
                param_positions_dict[param] = (k, k + J*d)                                
                k += J*d
            elif param == "beta":
                param_positions_dict[param] = (k, k + K)                                   
                k += K    
            elif param == "alpha":
                param_positions_dict[param] = (k, k + J)                                       
                k += J    
            elif param == "gamma":
                param_positions_dict[param] = (k, k + 1)                                
                k += 1
            elif param == "delta":
                param_positions_dict[param] = (k, k + 1)                                
                k += 1
            elif param == "sigma_e":
                param_positions_dict[param] = (k, k + 1)                                
                k += 1

        args = (DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,                     
                parameter_space_dim, None, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)   
        

        # start efficiency monitoring - interval in seconds
        print_threadpool_info()
        monitor = Monitor(interval=0.01, fastprogram=True)
        monitor.start()            
        try:            
            t_start = time.time()            
            theta_hat = do_correspondence_analysis(ca, Y, param_positions_dict, args, plot_online=plot_online, seedint=seedint)
            t_end = time.time()
            monitor.stop()
            wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
            efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                        avg_threads, max_threads, avg_processes, max_processes)
            elapsedtime = str(timedelta(seconds=t_end - t_start))   
        
            theta = [(theta_hat, None, None, None, None, None, None, None, None, True)]
            rank_and_plot_solutions(theta, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, dst_func, param_positions_dict, 
                                    DIR_out, args, seedint=seedint, get_RT_error=False, plot_solutions=False, compute_posterior=False)
        except:
            print("Failed: {}".format(country))
            return
        

if __name__ == "__main__":
    
    
    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1
    
    countries = ["finland", "france", "germany", "netherlands", "poland", "uk"] #["us"]
    dataspace = "/mnt/hdd2/ioannischalkiadakis/epodata_rsspaper/"

    for year in [2023, 2020]: #2020
        for country in countries:

            datasets_names = [file.name for file in pathlib.Path(dataspace).iterdir() if file.is_file() and (country in file.name and str(year) in file.name and "mappings" in file.name)]
            if len(datasets_names) == 0:
                continue

            K = int(datasets_names[0].split("_")[3].replace("K", ""))
            J = int(datasets_names[0].split("_")[4].replace("J", ""))
            
            print(parallel, K, J, elementwise, evaluate_posterior, country, year)
            
            parameter_names = ["X", "Z"]
            d = 2  
            prior_loc_x = np.zeros((d,))
            prior_scale_x = np.eye(d)
            prior_loc_z = np.zeros((d,))
            prior_scale_z = np.eye(d)
            prior_loc_phi = np.zeros((d,))
            prior_scale_phi = np.eye(d)
            prior_loc_alpha = 0
            prior_scale_alpha = 1    
            prior_loc_beta = 0
            prior_scale_beta = 1
            prior_loc_gamma = 0
            prior_scale_gamma = 1    
            prior_loc_delta = 0
            prior_scale_delta = 1        
            # a
            prior_loc_sigmae = 3
            # b
            prior_scale_sigmae = 0.5
            max_signal2noise_ratio = 25 # in dB   # max snr

            min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
            print(min_sigma_e)

            tol = 1e-6    
            data_location = dataspace
            total_running_processes = 30      

            args = (None, total_running_processes, data_location, None, parameter_names, J, K, d, None, None, tol,                     
                    None, None, None, None, None, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                    prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                    prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                    None, None, None, min_sigma_e, None)  

            # full, with status quo
            # parameter_space_dim = (K+2*J)*d + J + K + 3
            # no status quo
            parameter_space_dim = (K+J)*d + J + K + 2
            theta_true = np.zeros((parameter_space_dim,))
            print("Parameter space dimensionality: {}".format(parameter_space_dim))
            main(J=J, K=K, d=d, total_running_processes=total_running_processes, 
                data_location=data_location, parallel=parallel, 
                parameter_names=parameter_names, country=country, 
                dst_func=None, parameter_space_dim=parameter_space_dim, 
                penalty_weight_Z=None, constant_Z=None, retries=None, 
                elementwise=elementwise, evaluate_posterior=evaluate_posterior, 
                temperature_rate=None, temperature_steps=None, L=None, tol=tol, 
                prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, 
                prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
                prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, 
                prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
                prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, 
                prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma, 
                prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta,         
                prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
                gridpoints_num=None, diff_iter=None, disp=None, theta_true=theta_true,
                year=year, min_sigma_e=min_sigma_e, fastrun=None,
                max_restarts=None, max_partial_restarts=None, 
                max_halving=None, plot_online=None, seedint=seed_value)
    


    