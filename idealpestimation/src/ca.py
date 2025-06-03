import os 

os.environ["OMP_NUM_THREADS"] = "500"
os.environ["MKL_NUM_THREADS"] = "500"
os.environ["OPENBLAS_NUM_THREADS"] = "500"
os.environ["NUMBA_NUM_THREADS"] = "500"
os.environ["JAX_NUM_THREADS"] = "1000"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=500"

import sys
import ipdb
import jax
import pathlib
import jsonlines
import numpy as np
import random
from idealpestimation.src.parallel_manager import jsonlines
from idealpestimation.src.utils import pickle, optimisation_dict2params,\
                                        time, timedelta, parse_input_arguments, \
                                            rank_and_plot_solutions, print_threadpool_info, \
                                                get_min_achievable_mse_under_rotation_trnsl
from idealpestimation.src.efficiency_monitor import Monitor
import prince


def do_correspondence_analysis(ca_runner, Y, param_positions_dict, args, plot_online=False):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    estimated_theta = []

    params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
    X_true = np.asarray(params_true["X"]) # d x K       
    Z_true = np.asarray(params_true["Z"]) # d x J          

    ca_out = ca_runner.fit(Y)
    
    Xhat = ca_out.row_coordinates(Y).values.T
    Zhat = ca_out.column_coordinates(Y).values.T
    theta_hat = np.zeros((parameter_space_dim,))
    for param in parameter_names:
        if param == "X":
            theta_hat[param_positions_dict[param][0]:param_positions_dict[param][1]] = Xhat.reshape((K*d,), order="F") 
        elif param == "Z":
            theta_hat[param_positions_dict[param][0]:param_positions_dict[param][1]] = Zhat.reshape((J*d,), order="F")
        else:
            # Note: passing true parameter values for compatibility with rank_and_plot_solutions
            theta_hat[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()

    _, _, mse_x, mse_x_nonRT, meanrelerror_x, meanrelerror_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=Xhat)
    _, _, mse_z, mse_z_nonRT, meanrelerror_z, meanrelerror_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Zhat)
        
    estimated_theta.append((theta_hat, mse_x, mse_z, mse_x_nonRT, mse_z_nonRT, meanrelerror_x, meanrelerror_z, meanrelerror_x_nonRT, meanrelerror_z_nonRT))

    return estimated_theta

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
        
        for m in range(trialsmin, trialsmax, 1):
            DIR_out = "{}/{}/estimation_CA/".format(data_location, m)
            pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
            
            # load data    
            with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
                Y = pickle.load(f)
            Y = Y.astype(np.int8).reshape((K, J), order="F")    

            # done internally in prince.CA
            # ypp = np.sum(Y)
            # vec_ones = np.ones((J,1))
            # wm = (Y @ vec_ones)/ypp
            # wm_diag = np.diag(1/np.sqrt(wm))
            # wn = (vec_ones.T @ Y)/ypp
            # wn_diag = np.diag(1/np.sqrt(wn))
            # Smat = (wm_diag @ (Y - ypp *(wm @ wn)) @ wn_diag)/ypp

            ca = prince.CA(
                n_components=d,
                n_iter=10,
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
            theta = do_correspondence_analysis(ca, Y, param_positions_dict, args, plot_online=plot_online)
            t_end = time.time()
            monitor.stop()
            wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
            efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                        avg_threads, max_threads, avg_processes, max_processes)
            elapsedtime = str(timedelta(seconds=t_end - t_start))   
            rank_and_plot_solutions(theta, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, dst_func, param_positions_dict, DIR_out, args)


if __name__ == "__main__":
    
    # python idealpestimation/src/ca.py --trials 1 --K 30 --J 10 --sigmae 001

    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    args = parse_input_arguments()
    
    if args.trials is None or args.K is None or args.J is None or args.sigmae is None:
        parallel = False
        Mmin = 0
        M = 1
        K = 30
        J = 10
        sigma_e_true = 0.01
        total_running_processes = 20   
        elementwise = True
        evaluate_posterior = True
    else:
        parallel = args.parallel
        trialsstr = args.trials
        if "-" in trialsstr:
            trialsparts = trialsstr.split("-")
            Mmin = int(trialsparts[0])
            M = int(trialsparts[1])
        else:
            Mmin = 0
            M = int(trialsstr)
        K = args.K
        J = args.J
        total_running_processes = args.total_running_processes
        sigma_e_true = args.sigmae
        elementwise = args.elementwise
        evaluate_posterior = args.evaluate_posterior

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes, elementwise, evaluate_posterior)
    
    # if not parallel:
    #     jax.default_device = jax.devices("cpu")[0]
    #     jax.config.update("jax_traceback_filtering", "off")
    
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
    #/home/ioannischalkiadakis/ideal
    # data_location = "./idealpestimation/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))
    data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata_rsspaper/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e_true).replace(".", ""))
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
    


    