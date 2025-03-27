import os 
import sys
import ipdb
import jax
import pathlib
import pickle
import jsonlines
import numpy as np
import random
from idealpestimation.src.parallel_manager import jsonlines, ProcessManager
from idealpestimation.src.icm_annealing_posteriorpower import icm_posterior_power_annealing
from idealpestimation.src.utils import time, timedelta, parse_input_arguments, rank_and_plot_solutions


class ProcessManagerSyntheticDataAnnealing(ProcessManager):
    def __init__(self, max_processes):
        super().__init__(max_processes)
    
    def worker_process(self, args):

        current_pid = os.getpid()
        with self.execution_counter.get_lock():
            self.execution_counter.value += 1
            self.shared_dict[current_pid] = self.execution_counter.value
        
        t0 = time.time()
        k_prev, s, Y_annealed, temperature_rate, temperature_steps, percentage_parameter_change,\
            fastrun, True, y_rows, theta_part_annealing, theta, elapsedtime, param_positions_dict,\
                DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,\
                    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x,\
                        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,\
                            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae,\
                                gridpoints_num, diff_iter, disp, min_sigma_e, theta_true = args
        DIR_out_icm = "{}/{}/".format(DIR_out, s)
        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)

        icm_args = (DIR_out_icm, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,\
                    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x,\
                        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,\
                            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae,\
                                gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)
        
        theta = icm_posterior_power_annealing(Y_annealed, param_positions_dict, icm_args,
                                    temperature_rate=temperature_rate, temperature_steps=temperature_steps, 
                                    percentage_parameter_change=percentage_parameter_change, 
                                    fastrun=fastrun, data_annealing=False, annealing_rows=None, 
                                    theta_part_annealing=None)
        elapsedtime = str(timedelta(seconds=time.time()-t0))   
        # get highest-likelihood solution and feed into icm_posterior_power_annealing to initialise theta for next iteration
        theta_part_annealing = rank_and_plot_solutions(theta, elapsedtime, Y_annealed, J, y_rows, d, parameter_names, 
                                                    dst_func, param_positions_dict, DIR_out_icm, icm_args)     
        


def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, temperature_rate=[0, 1], temperature_steps=[0.1], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False,
        theta_true=None, percentage_parameter_change=1, min_sigma_e=None, fastrun=False):
        
        for m in range(trialsmin, trialsmax, 1):
            if elementwise:
                if evaluate_posterior:                    
                    DIR_out = "{}/{}/estimation_ICM_data_annealing_evaluate_posterior_elementwise/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_gamma1/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_gamma1_perturb/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_annealing/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_annealing_perturb/".format(data_location, m)
                else:
                    DIR_out = "{}/{}/estimation_ICM_data_annealing_differentiate_posterior_elementwise/".format(data_location, m)
            else:
                if evaluate_posterior:
                    DIR_out = "{}/{}/estimation_ICM_data_annealing_evaluate_posterior_vector_test/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector_gamma1/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector_gamma1_perturb/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector_annealing/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector_annealing_perturb/".format(data_location, m)
                else:
                    raise NotImplementedError("At the moment we only evaluate the posterior with vector parameters - minimising coordinate-wise is more efficient.")
            pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
            
            # load data    
            with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
                Y = pickle.load(f)
            Y = Y.astype(np.int8).reshape((K, J), order="F")    

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
            
            if elementwise and not evaluate_posterior:
                # growing data window + differentiate posterior
                theta_part_annealing = None
                for s in range(temperature_rate[0], temperature_rate[1], temperature_steps[0]):
                    y_rows = int(round(s*K))
                    Y_annealed = Y[0:y_rows, :]            
                    t0 = time.time()
                    theta = icm_posterior_power_annealing(Y_annealed, param_positions_dict, args,
                                                temperature_rate=temperature_rate, temperature_steps=temperature_steps, 
                                                percentage_parameter_change=percentage_parameter_change, 
                                                fastrun=fastrun, data_annealing=True, annealing_rows=y_rows, 
                                                theta_part_annealing=theta_part_annealing)
                    elapsedtime = str(timedelta(seconds=time.time()-t0))   
                    # get highest-likelihood solution and feed into icm_posterior_power_annealing to initialise theta for next iteration
                    theta_part_annealing = rank_and_plot_solutions(theta, elapsedtime, Y_annealed, J, y_rows, d, parameter_names, 
                                                                dst_func, param_positions_dict, DIR_out, args)                
            else:
                if not parallel:
                    raise AttributeError("Set the parallel flag for parallel data annealing.")
                # for parallel data annealing: store partial solution
                manager = ProcessManagerSyntheticDataAnnealing(total_running_processes)  
                try:   
                    manager.create_results_dict(optim_target="all")  
                    t0 = time.time()
                    k_prev = 0
                    while True:       
                        for s in range(temperature_rate[0], temperature_rate[1], temperature_steps[0]):
                            y_rows = int(round(s*K))
                            Y_annealed = Y[k_prev:y_rows, :]       

                            worker_args = (k_prev, s, Y_annealed, temperature_rate, temperature_steps, percentage_parameter_change, 
                                        fastrun, True, y_rows-k_prev, theta_part_annealing, theta, elapsedtime, param_positions_dict, 
                                        DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,                     
                                        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                                        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                                        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                                        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)
                            #####  parallelisation with Parallel Manager #####
                            manager.cleanup_finished_processes()
                            current_count = manager.current_process_count()                                
                            print(f"Currently running processes: {current_count}")
                            manager.print_shared_dict() 
                            while current_count == total_running_processes:
                                manager.cleanup_finished_processes()
                                current_count = manager.current_process_count()                                                                                                                                                   
                            if current_count < total_running_processes:
                                manager.spawn_process(args=(worker_args,))      
                            k_prev += y_rows                                            
                            # Wait before next iteration
                            time.sleep(1)  
                            ################################################## 
                        if manager.all_processes_complete.is_set():
                            break  
                    elapsedtime = str(timedelta(seconds=time.time()-t0))
                except KeyboardInterrupt:
                    # On Ctrl-C stop all processes
                    print("\nShutting down gracefully...")
                    if parallel:
                        manager.cleanup_all_processes()
                        manager.print_shared_dict()  # Final print of shared dictionary                    
                    sys.exit(0)      



if __name__ == "__main__":

    # python idealpestimation/src/icm_annealing_posteriorpower.py --trials 1 --K 30 --J 10 --sigmae 05 --elementwise --evaluate_posterior  --total_running_processes 5

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
        elementwise = False
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
    # before halving annealing rate
    percentage_parameter_change = 0.2

    if not parallel:
        jax.default_device = jax.devices("cpu")[0]
        jax.config.update("jax_traceback_filtering", "off")
    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = 50
    penalty_weight_Z = 0.0
    constant_Z = 0.0
    retries = 20
    diff_iter = None
    disp = False
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma" , "sigma_e"]
    d = 2  
    gridpoints_num = 15 #30
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
    temperature_steps = [0.1, 1]
    temperature_rate = [0.1]

    fastrun = True
    max_signal2noise_ratio = 25 # in dB   # max snr

    min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
    print(min_sigma_e)

    tol = 1e-6    
    #/home/ioannischalkiadakis/ideal
    # data_location = "./idealpestimation/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))
    # data_location = "/mnt/hdd2/ioannischalkiadakis/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))
    data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata_mmtest_polarisedregime/data_K{}_J{}_sigmae{}_5poles/".format(K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 30                 
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 3
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    main(J=J, K=K, d=d, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, trialsmax=M, 
        penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=retries, 
        elementwise=elementwise, evaluate_posterior=evaluate_posterior, temperature_rate=temperature_rate, temperature_steps=temperature_steps, L=niter, tol=tol, 
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
        prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, 
        prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma, 
        prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta,         
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
        gridpoints_num=gridpoints_num, diff_iter=diff_iter, disp=disp, theta_true=theta_true,
        percentage_parameter_change=percentage_parameter_change, min_sigma_e=min_sigma_e, fastrun=fastrun)
    
    