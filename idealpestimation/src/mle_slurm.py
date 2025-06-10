import os 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import ipdb
import pathlib
import numpy as np
import math
import random
from idealpestimation.src.parallel_manager import jsonlines
from idealpestimation.src.utils import  time, timedelta, print_threadpool_info
from idealpestimation.src.efficiency_monitor import Monitor
from idealpestimation.src.mle import estimate_mle
import pandas as pd
        

def main(J=2, K=2, d=1, N=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        niter=None, parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, 
        constant_Z=0.0, retries=10, min_sigma_e=None, prior_loc_x=None, prior_scale_x=None, 
        prior_loc_z=None, prior_scale_z=None, prior_loc_phi=None, prior_scale_phi=None,
        prior_loc_beta=None, prior_scale_beta=None, prior_loc_alpha=None, prior_scale_alpha=None,
        prior_loc_gamma=None, prior_scale_gamma=None, prior_loc_delta=None, prior_scale_delta=None, 
        prior_loc_sigmae=None, prior_scale_sigmae=None, param_positions_dict=None, rng=None, batchsize=None, data_start=0, data_end=0):

    
    DIR_top = data_location      
    m = trialsmin

    theta_true = np.zeros((parameter_space_dim,))    
    with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
        for result in f.iter(type=dict, skip_invalid=True):
            for param in parameter_names:
                theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 

        
    subdataset_name = "dataset_{}_{}".format(data_start, data_end)
    DIR_out = "{}/{}/{}/{}/estimation/".format(DIR_top, m, batchsize, subdataset_name)
    print(DIR_out)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True) 
    args = (DIR_out, data_location, subdataset_name, None, optimisation_method, 
            parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, min_sigma_e,
            prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng, batchsize, theta_true)
    estimate_mle(args)                  

   
    return DIR_out


if __name__ == "__main__":

    seed_value = 9125
    random.seed(seed_value)
    np.random.seed(seed_value)
    rng = np.random.default_rng()

    parallel = False
    total_running_processes = 1

    dataspace = "/linkhome/rech/genpuz01/umi36fq/idealdata_slurm_test/"    
    parameter_vector_idx = 0 # int(os.environ["SLURM_ARRAY_TASK_ID"])    
    parameter_grid = pd.read_csv("{}/slurm_experimentI_mle_test.csv".format(dataspace), header=None)
    parameter_vector = parameter_grid.iloc[parameter_vector_idx].values

    Mmin = parameter_vector[0]
    M = Mmin + 1
    K = parameter_vector[1]
    J = parameter_vector[2]
    sigma_e_true = parameter_vector[3]
    batchsize = parameter_vector[4]
    data_start = parameter_vector[5]
    data_end = parameter_vector[6]

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes, data_start, data_end)
    
    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = 200
    penalty_weight_Z = 0.0
    constant_Z = 0.0
    retries = 30
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
    prior_scale_delta= 1        
    prior_loc_sigmae = 3
    prior_scale_sigmae = 0.5
    data_location = "{}/data_K{}_J{}_sigmae{}/".format(dataspace, K, J, str(sigma_e_true).replace(".", ""))
    
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 3
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 2
    max_signal2noise_ratio = 25 # in dB   # max snr
    min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    # for distributing per N rows
    N = math.ceil(parameter_space_dim/J)
    print("Observed data points per data split: {}".format(N*J))    
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
    
    
    # start efficiency monitoring - interval in seconds
    print_threadpool_info()
    monitor = Monitor(interval=0.5)
    monitor.start()   
    t_start = time.time()  
    dir_out = main(J=J, K=K, d=d, N=N, total_running_processes=total_running_processes, 
                                                data_location=data_location, parallel=parallel, 
                                                parameter_names=parameter_names, optimisation_method=optimisation_method, 
                                                dst_func=dst_func, niter=niter, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, 
                                                trialsmax=M, penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=retries, min_sigma_e=min_sigma_e,
                                                prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, 
                                                prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, prior_loc_phi=prior_loc_phi, 
                                                prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
                                                prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, prior_loc_gamma=prior_loc_gamma, 
                                                prior_scale_gamma=prior_scale_gamma, prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta, 
                                                prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae, param_positions_dict=param_positions_dict, 
                                                rng=rng, batchsize=batchsize, data_start=data_start, data_end=data_end)   
    t_end = time.time()
    monitor.stop()
    wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
        avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
    efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                avg_threads, max_threads, avg_processes, max_processes)
    elapsedtime = str(timedelta(seconds=t_end - t_start))   
    out_file = "{}/efficiency_metrics.jsonl".format(dir_out)
    with open(out_file, 'a') as f:         
        writer = jsonlines.Writer(f)
        writer.write({"wall_duration": wall_duration, 
                    "avg_total_cpu_util": avg_total_cpu_util, 
                    "max_total_cpu_util": max_total_cpu_util, 
                    "avg_total_ram_residentsetsize_MB": avg_total_ram_residentsetsize_MB, 
                    "max_total_ram_residentsetsize_MB": max_total_ram_residentsetsize_MB,
                    "avg_threads": avg_threads, 
                    "max_threads": max_threads, 
                    "avg_processes": avg_processes, 
                    "max_processes": max_processes})
    
    