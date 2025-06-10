import os 

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMBA_NUM_THREADS"] = "2"

import ipdb
import pathlib
import jsonlines
import numpy as np
import random
from idealpestimation.src.utils import pickle, \
                                        time, timedelta, \
                                            rank_and_plot_solutions, print_threadpool_info, \
                                                clean_up_data_matrix
from idealpestimation.src.efficiency_monitor import Monitor
from idealpestimation.src.ca import CA_custom, do_correspondence_analysis
import pandas as pd



def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, temperature_rate=[0, 1], temperature_steps=[1e-3], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False,
        theta_true=None, percentage_parameter_change=1, min_sigma_e=None, fastrun=False,
        max_restarts=2, max_partial_restarts=2, max_halving=2, plot_online=False, seedint=None):
        
        m = trialsmin        
        print(trialsmin)
        DIR_out = "{}/{}/estimation_CA/".format(data_location, m)
        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
        
        # load data    
        with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
            Y = pickle.load(f)
        Y = Y.astype(np.int8).reshape((K, J), order="F")    

        ca = CA_custom(
            n_components=d,
            n_iter=5,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=seedint
        )

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

        with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
            for result in f.iter(type=dict, skip_invalid=True):
                for param in parameter_names:
                    theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
        
        
        Y, K, J, theta_true, param_positions_dict, parameter_space_dim = clean_up_data_matrix(Y, K, J, d, theta_true, parameter_names, param_positions_dict)
        
        
        
        args = (DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,                     
                parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)   
        

        # start efficiency monitoring - interval in seconds
        print_threadpool_info()
        monitor = Monitor(interval=0.005)
        monitor.start()            

        t_start = time.time()            
        theta_hat = do_correspondence_analysis(ca, Y, param_positions_dict, args, plot_online=plot_online, seedint=seedint)
        t_end = time.time()
        monitor.stop()
        wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
            avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
        efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                    avg_threads, max_threads, avg_processes, max_processes)
        elapsedtime = str(timedelta(seconds=t_end - t_start))   
    
        theta = [(theta_hat, None, None, None, None, None, None, None, None)]
        rank_and_plot_solutions(theta, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, dst_func, param_positions_dict, 
                                DIR_out, args, seedint=seedint, get_RT_error=False)


if __name__ == "__main__":
    
    
    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1

    dataspace = "/linkhome/rech/genpuz01/umi36fq/idealdata_slurm_test/"    
    parameter_vector_idx = 0 #int(os.environ["SLURM_ARRAY_TASK_ID"])    
    parameter_grid = pd.read_csv("/linkhome/rech/genpuz01/umi36fq/slurm_experimentI_ca_test.csv", header=None)
    parameter_vector = parameter_grid.iloc[parameter_vector_idx].values

    Mmin = int(parameter_vector[0])
    M = int(Mmin + 1)
    K = int(parameter_vector[1])
    J = int(parameter_vector[2])
    sigma_e_true = parameter_vector[3]

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes, elementwise, evaluate_posterior)
    
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]
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
    data_location = "{}/data_K{}_J{}_sigmae{}/".format(dataspace, K, J, str(sigma_e_true).replace(".", ""))
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
        parameter_names=parameter_names, optimisation_method=None, 
        dst_func=None, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, trialsmax=M, 
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
        percentage_parameter_change=None, min_sigma_e=min_sigma_e, fastrun=None,
        max_restarts=None, max_partial_restarts=None, 
        max_halving=None, plot_online=None, seedint=seed_value)
    


    