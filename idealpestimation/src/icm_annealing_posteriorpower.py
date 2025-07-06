import os 

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMBA_NUM_THREADS"] = "1"
# os.environ["JAX_NUM_THREADS"] = "500"
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=500"

import sys
import ipdb
# import jax
import pathlib
import numpy as np
import random
import itertools
from scipy.optimize import minimize
import plotly.graph_objects as go
from idealpestimation.src.parallel_manager import jsonlines, ProcessManager
from idealpestimation.src.utils import pickle, sample_theta_curr_init, \
                                        create_constraint_functions_icm, \
                                            update_annealing_temperature, \
                                                compute_and_plot_mse, time, datetime, \
                                                    timedelta, parse_input_arguments, \
                                                        halve_annealing_rate_upd_schedule,\
                                                            plot_posteriors_during_estimation, \
                                                                get_parameter_name_and_vector_coordinate,\
                                                                    get_posterior_for_optimisation_vec,optimisation_dict2params,\
                                                                        rank_and_plot_solutions, get_evaluation_grid, check_convergence,\
                                                                            data_annealing_init_theta_given_theta_prev, plot_posterior_vec_runtimes, \
                                                                                print_threadpool_info, error_polarisation_plots, clean_up_data_matrix
from idealpestimation.src.efficiency_monitor import Monitor


class ProcessManagerSynthetic(ProcessManager):
    def __init__(self, max_processes):
        super().__init__(max_processes)
    
    def worker_process(self, args):

        current_pid = os.getpid()
        with self.execution_counter.get_lock():
            self.execution_counter.value += 1
            self.shared_dict[current_pid] = self.execution_counter.value
        
        t0 = time.time()
        f, gridpoint, gamma_annealing, l, DIR_out, param, idx, vector_coordinate = args
        posterior_eval = -f(gridpoint)
        
        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = current_pid
        grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsedtime = timedelta(seconds=time.time()-t0)            
        total_seconds = int(elapsedtime.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60            
        grid_and_optim_outcome["elapsedtime"] = f"{hours}:{minutes:02d}:{seconds:02d}"       
        grid_and_optim_outcome["elapsedtime_hours"] = hours       
        grid_and_optim_outcome["elapsedtime_seconds"] = str(elapsedtime)   
        # time_obj = datetime.strptime(grid_and_optim_outcome["elapsedtime_seconds"], '%H:%M:%S.%f')
        # hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
        grid_and_optim_outcome["elapsedtime_hours"] = hours  
        if isinstance(gridpoint, float):
            grid_and_optim_outcome["gridpoint"] = gridpoint
        else:      
            grid_and_optim_outcome["gridpoint"] = list(gridpoint)
        grid_and_optim_outcome["posterior"] = posterior_eval.tolist()
        # print(posterior_eval.tolist())
        grid_and_optim_outcome["gamma_annealing"] = gamma_annealing
        grid_and_optim_outcome["icm_iteration"] = l
            
        if vector_coordinate is not None:
            out_file = "{}/{}/{}/posterior_evaluation_anneal_power_{}_param_idx_{}_vectorcoordinate_{}.jsonl".format(DIR_out, param, l, gamma_annealing, idx, vector_coordinate)
        else:
            out_file = "{}/{}/{}/posterior_evaluation_anneal_power_{}_param_idx_{}.jsonl".format(DIR_out, param, l, gamma_annealing, idx)
        pathlib.Path("{}/{}/{}/".format(DIR_out, param, l)).mkdir(parents=True, exist_ok=True)     
        self.append_to_json_file(grid_and_optim_outcome, output_file=out_file)


def optimise_posterior_elementwise(param, idx, vector_index_in_param_matrix, vector_coordinate, Y, gamma, theta_curr, param_positions_dict, l, args, debug=False):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
 
    theta_test_in = theta_curr.copy()
    # Negate its result for the negative log posterior likelihood
    f = get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta_test_in.copy(), gamma, param_positions_dict, args)
    # get grid depending on param
    grid, _ = get_evaluation_grid(param, vector_coordinate, args)      
    if evaluate_posterior:                  
        # parallel eval grid
        if parallel:
            manager = ProcessManagerSynthetic(total_running_processes)  
            try:   
                manager.create_results_dict(optim_target="all")  
                t0 = time.time()  
                while True:                    
                    for gridpoint in grid:
                        worker_args = (f, gridpoint, gamma, l, DIR_out, param, idx, vector_coordinate) 
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
                        # Wait before next iteration
                        time.sleep(1)  
                        ################################################## 
                    if manager.all_processes_complete.is_set():
                        break                    
                if manager.all_processes_complete.is_set():                        
                    out_folder = "{}/{}/{}/".format(DIR_out, param, l)
                    min_f = np.inf
                    param_estimate = None                    
                    with jsonlines.open("{}/posterior_evaluation_anneal_power_{}_param_idx_{}_vectorcoordinate_{}.jsonl".format(out_folder, 
                                                                                                gamma, idx, vector_coordinate), mode="r") as f: 
                        for result in f.iter(type=dict, skip_invalid=True): 
                            if not np.isnan(result["posterior"]) and result["posterior"] < min_f:
                                min_f = result["posterior"]
                                param_estimate = result["gridpoint"]
                            else:
                                continue
                elapsedtime = str(timedelta(seconds=time.time()-t0))   
                # time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                # hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)               
            except KeyboardInterrupt:
                # On Ctrl-C stop all processes
                print("\nShutting down gracefully...")
                if parallel:
                    manager.cleanup_all_processes()
                    manager.print_shared_dict()  # Final print of shared dictionary                    
                sys.exit(0)  
        else:
            min_f = np.inf
            param_estimate = None            
            if debug:
                t0 = time.time()  
                grideval = []          
                for gridpoint in grid:
                    posterior_eval = -f(gridpoint)
                    grideval.append(posterior_eval)
                    if posterior_eval < min_f:
                        min_f = posterior_eval
                        param_estimate = gridpoint
                elapsedtime = str(timedelta(seconds=time.time()-t0))   
                # time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                # hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)       
                print("Loop elapsed time: {}".format(elapsedtime))
            
            t0 = time.time()
            grid_list = np.asarray([gp for gp in grid])   
            grideval_map = np.asarray(list(map(f, grid_list)))        
            if debug:
                param_estimate_map = grid_list[np.argmax(grideval_map)]
            else:                
                param_estimate = grid_list[np.argmax(grideval_map)]
            elapsedtime = str(timedelta(seconds=time.time()-t0))  
            if debug:
                print("Map elapsed time: {}".format(elapsedtime))                         
                assert(np.allclose(grideval, -grideval_map))
                assert(np.allclose(param_estimate, param_estimate_map))   
    else:
        # use previously evaluation grid as starting points grid
        retry = 0
        minus_f = lambda x: -f(x)
        t0 = time.time()
        while retry < retries:   
            # print("Retry: {}".format(retry))
            gridpoint = random.choice(grid)
            optimize_kwargs = {
                'method': optimisation_method,
                'x0': np.asarray([gridpoint]),                
                'jac': '3-point'
            }                               
            bounds = create_constraint_functions_icm(parameter_space_dim, vector_coordinate, param=param, param_positions_dict=param_positions_dict, args=args)    
            if diff_iter is not None:
                result = minimize(minus_f, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxiter":diff_iter, "maxls":1000000})
            else:
                result = minimize(minus_f, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxls":1000000})
            
            param_estimate = result.x[0]          
            
            if result.success:
                break
            else:              
                grid.remove(gridpoint)
                retry += 1
        elapsedtime = str(timedelta(seconds=time.time()-t0))   
        # time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
        # hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)    
    
    theta_test_out = theta_curr.copy()
    theta_test_out[idx] = param_estimate
    
    return theta_test_out, elapsedtime

def optimise_posterior_vector(param, idx, Y, gamma, theta_curr, param_positions_dict, l, args, debug=False):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    theta_test_in = theta_curr.copy()
    if evaluate_posterior:
        # Negate its result for the negative log posterior likelihood        
        f = get_posterior_for_optimisation_vec(param, Y, idx, idx, None, theta_test_in.copy(), gamma, param_positions_dict, args)
        # get grid depending on param
        grid, _ = get_evaluation_grid(param, None, args)        
        # parallel eval grid
        if parallel:
            manager = ProcessManagerSynthetic(total_running_processes)  
            try:   
                manager.create_results_dict(optim_target="all")  
                t0 = time.time()
                while True:                    
                    for gridpoint in grid:
                        worker_args = (f, gridpoint, gamma, l, DIR_out, param, idx, None) 
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
                        # Wait before next iteration
                        time.sleep(1)  
                        ################################################## 
                    if manager.all_processes_complete.is_set():
                        break  
                elapsedtime = str(timedelta(seconds=time.time()-t0))   
                # time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                # hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)                   
                if manager.all_processes_complete.is_set():                        
                    out_folder = "{}/{}/{}/".format(DIR_out, param, l)
                    min_f = np.inf
                    param_estimate = None
                    with jsonlines.open("{}/posterior_evaluation_anneal_power_{}_param_idx_{}.jsonl".format(out_folder, gamma, idx), mode="r") as f: 
                        for result in f.iter(type=dict, skip_invalid=True): 
                            if not np.isnan(result["posterior"]) and result["posterior"] < min_f:
                                min_f = result["posterior"]
                                param_estimate = result["gridpoint"]
                            else:
                                continue
            except KeyboardInterrupt:
                # On Ctrl-C stop all processes
                print("\nShutting down gracefully...")
                if parallel:
                    manager.cleanup_all_processes()
                    manager.print_shared_dict()  # Final print of shared dictionary                    
                sys.exit(0)  
        else:
            min_f = np.inf
            param_estimate = None
            if debug:
                grideval = []
                t0 = time.time()
                for gridpoint in grid:                
                    posterior_eval = -f(gridpoint)
                    grideval.append(posterior_eval)
                    if posterior_eval < min_f:
                        min_f = posterior_eval
                        param_estimate = gridpoint            
                elapsedtime = str(timedelta(seconds=time.time()-t0))   
                # time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                # hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)       
                print("Loop elapsed time: {}".format(elapsedtime))            
                # to refill grid
                grid, _ = get_evaluation_grid(param, None, args)    
            
            t0 = time.time()     
            grid_list = np.asarray([gp for gp in grid])   
            grideval_map = np.asarray(list(map(f, grid_list)))        
            if debug:
                param_estimate_map = grid_list[np.argmax(grideval_map)]
            else:                
                param_estimate = grid_list[np.argmax(grideval_map)]           
            elapsedtime = str(timedelta(seconds=time.time()-t0))  
            if debug:
                print("Map elapsed time: {}".format(elapsedtime))                         
                assert(np.allclose(grideval, -grideval_map))
                assert(np.allclose(param_estimate, param_estimate_map))              
    else:
        raise NotImplementedError("More efficient to differentiate posterior coordinate-wise.")

    theta_test_out = theta_curr.copy()
    if param in ["X", "Z", "Phi"]:
        theta_test_out[param_positions_dict[param][0] + idx*d:param_positions_dict[param][0] + (idx + 1)*d] = param_estimate.copy()
    else:
        # scalar updates and coordinate-wise updates of alphas/betas
        theta_test_out[param_positions_dict[param][0] + idx:param_positions_dict[param][0] + (idx + 1)] = param_estimate

    return theta_test_out, elapsedtime

def icm_posterior_power_annealing_debug(Y, param_positions_dict, args, temperature_rate=None, temperature_steps=None, 
                                plot_online=True, percentage_parameter_change=1, seedint=1234):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    iiidx = 0
    temperature_steps_init = temperature_steps.copy()
    temperature_rate_init = temperature_rate.copy()
    while iiidx < parameter_space_dim:
        
        target_param1, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=iiidx, d=d)                    
        t0 = time.time()
        theta_samples_list = None
        base2exponent = 10
        idx_all = None
        theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict, 
                                                                    args, samples_list=theta_samples_list, idx_all=idx_all)
        
        gamma = 0.01
        l = 0
        i = 0    
        theta_prev = np.zeros((parameter_space_dim,))

        all_gammas = []
        for gidx in range(len(temperature_steps_init[1:])):
            upperlim = temperature_steps_init[1+gidx]        
            start = gamma if gidx==0 else all_gammas[-1]        
            all_gammas.extend(np.arange(start, upperlim, temperature_rate_init[gidx]))            
        N = len(all_gammas)
        print("Annealing schedule: {},{}".format(N, iiidx))       

        #######################################            
        perturb = False
        testparam = target_param1
        if not elementwise and testparam not in ["X", "Z", "Phi"]:
            if target_param1 in ["alpha"]:
                teststep = J
            elif target_param1 in ["beta"]:
                teststep = K
            else:
                # skip
                teststep = 1
            iiidx += teststep
            continue
        if testparam in ["alpha", "beta"]:
            vector_index = vector_coordinate
        else:
            vector_index = vector_index_in_param_matrix
        if elementwise:
            vector_test_coordinate = vector_coordinate
        else:
            vector_test_coordinate = None
        if vector_test_coordinate is not None:
            if testparam in ["X", "Z", "Phi"]:
                testidx = vector_index*d + vector_test_coordinate
            else:
                testidx = vector_test_coordinate
        else:
            testidx = vector_index
        if testparam is not None:
            theta_perturb = theta_true.copy()
            theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] += 0.1*np.linalg.norm(theta_true[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]], ord=2)
            theta_curr = theta_true.copy()
            for param in parameter_names:
                if param == testparam:
                    if vector_test_coordinate is not None:    
                        if perturb:
                            theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                        theta_curr[param_positions_dict[testparam][0]+testidx] = -4
                    else:   
                        if perturb:
                            theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                        theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d] = [-4]*d                        
                else:
                    if perturb:
                        theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_perturb[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                    else:
                        # fixed to true
                        continue
            all_gammas = [1]*len(all_gammas)
            gamma = 1
        #######################################
        
        delta_rate_prev = None
        mse_theta_full = []
        err_theta_full = []
        fig_xz = None
        mse_x_list = []
        mse_z_list = []
        err_x_list = []
        err_z_list = []
        err_x_nonRT_list = []
        err_z_nonRT_list = []
        per_param_sq_ers = dict()
        per_param_ers = dict()
        plotting_thetas = dict()
        per_param_heats = dict()
        fig_posteriors = dict()
        fig_posteriors_annealed = dict()
        for param in parameter_names:
            plotting_thetas[param] = []
            per_param_sq_ers[param] = []        
            per_param_ers[param] = []        
            fig_posteriors[param] = None
            fig_posteriors_annealed[param] = None
            if param in ["gamma", "delta", "sigma_e"]:
                continue
            else:
                per_param_heats["{}_sq_e".format(param)] = []
                per_param_heats["{}_e".format(param)] = []
        per_param_heats["theta_sq_e"] = []
        per_param_heats["theta_e"] = []
        per_param_sq_ers["X_rot_translated_mseOverMatrix"] = []
        per_param_sq_ers["Z_rot_translated_mseOverMatrix"] = []        
        per_param_sq_ers["X_mseOverMatrix"] = []
        per_param_sq_ers["Z_mseOverMatrix"] = []   
        per_param_ers["X_rot_translated_errOverMatrix"] = []
        per_param_ers["Z_rot_translated_errOverMatrix"] = []
        per_param_ers["X_errOverMatrix"] = []
        per_param_ers["Z_errOverMatrix"] = []   
        plot_restarts = []
        xbox = []
        total_iter = 1   
        halving_rate = 0 
        restarts = 0
        max_restarts = 3
        max_halving = 2
        estimated_thetas = []
        plot_online = False
        # to plot X before it has moved for the first time
        fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, theta_curr.copy(), total_iter, fig_posteriors, 
                                                                                            fig_posteriors_annealed, gamma, param_positions_dict, args, 
                                                                                            plot_arrows=True, testparam=testparam, testidx=testidx, testvec=vector_index) 

        converged = False
        while ((L is not None and l < L)) and (not converged):
            converged = False
            random_restart = False
            if elementwise:
                i = 0                    
                while i < parameter_space_dim and not converged:                                            
                    target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=i, d=d)                    
                    theta_test, _ = optimise_posterior_elementwise(target_param, i, vector_index_in_param_matrix, vector_coordinate, 
                                                                Y, gamma, theta_curr.copy(), param_positions_dict, L, args)                   
                    theta_curr = theta_test.copy()                    
                    gamma, delta_rate = update_annealing_temperature(gamma, total_iter, temperature_rate, temperature_steps, all_gammas)                   
                    mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_xz, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                         err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list = \
                                    compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, 
                                        param_positions_dict=param_positions_dict, plot_online=plot_online, mse_theta_full=mse_theta_full, err_theta_full=err_theta_full, 
                                        fig_xz=fig_xz, mse_x_list=mse_x_list, mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, 
                                        mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list, err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list,
                                        per_param_sq_ers=per_param_sq_ers, per_param_ers=per_param_ers,
                                        per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, seedint=seedint)       

                    #########################
                    if testparam is not None:
                        for param in parameter_names:
                            if param == testparam:
                                if vector_test_coordinate is not None:    
                                    testidx_val_tmp = theta_curr[param_positions_dict[testparam][0]+testidx]
                                    if perturb:
                                        theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                    else:
                                        theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_true[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                    theta_curr[param_positions_dict[testparam][0] + testidx] = testidx_val_tmp
                                else:
                                    continue                                
                            else:
                                if perturb:
                                    # +/- 10% of the norm of the parameter
                                    theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_perturb[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                                else:
                                    theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                    #########################                        
                    delta_rate_prev = delta_rate                            
                    # check is correct here as we are only changing the testparam/testidx in debug mode - all the rest are fixed to true
                    converged, delta_theta, random_restart = check_convergence(elementwise, theta_curr, theta_prev, param_positions_dict, i, 
                                                                            parameter_space_dim=parameter_space_dim, testparam=testparam, 
                                                                            testidx=testidx, p=percentage_parameter_change, tol=tol)                        
                    total_iter += 1   
                    i += 1       

                fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, 
                                                                                                                    theta_curr.copy(), total_iter, fig_posteriors, 
                                                                                                                    fig_posteriors_annealed, gamma, 
                                                                                                                    param_positions_dict, args, plot_arrows=True,
                                                                                                                    testparam=testparam, testidx=testidx)                          
                    
                if random_restart and random_restart < max_restarts: 
                    halved = False
                    if (halving_rate <= max_halving):                           
                        gamma, delta_rate_prev, temperature_rate, all_gammas, N = halve_annealing_rate_upd_schedule(N, gamma, 
                                                                                delta_rate_prev, temperature_rate, temperature_steps, all_gammas,  
                                                                                testparam=testparam)                    
                        halving_rate += 1          
                        halved = True
                    plot_restarts.append((l, total_iter, halved, "fullrestart"))                              
                    restarts += 1                    
                    # keep solution
                    estimated_thetas.append(theta_curr)                                       
                    # random restart
                    theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                                    args, samples_list=theta_samples_list, idx_all=idx_all)                        
                    gamma = 0.01 
                    converged = False
                    ################################
                    # as when starting, just not set testparam at the edge of the grid
                    if testparam is not None:
                        for param in parameter_names:
                            if param == testparam:
                                if vector_test_coordinate is not None:    
                                    testidx_val_tmp = theta_curr[param_positions_dict[testparam][0]+testidx]
                                    if perturb:
                                        theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                    else:
                                        theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_true[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                    theta_curr[param_positions_dict[testparam][0] + testidx] = testidx_val_tmp
                                else:
                                    # leave to theta_curr param section
                                    continue         
                            else:
                                theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                                if perturb:
                                    theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_perturb[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                        all_gammas = [1]*len(all_gammas)
                        gamma = 1
                    ################################
                    theta_prev = np.zeros((parameter_space_dim,))     
                else:                      
                    theta_prev = theta_curr.copy()        
                    l += 1   
                print(l, total_iter, converged, random_restart, restarts)
            else:
                for target_param in parameter_names:                     
                    if target_param in ["X", "beta"]:  
                        param_no = K                                                      
                    elif target_param in ["Z", "Phi", "alpha"]:
                        param_no = J                                                        
                    else:
                        # scalars
                        param_no = 1                    
                    for idx in range(param_no):
                        theta_test, _ = optimise_posterior_vector(target_param, idx, Y, gamma, theta_curr.copy(), 
                                                                param_positions_dict, L, args)     
                        theta_curr = theta_test.copy()
                        gamma, delta_rate = update_annealing_temperature(gamma, total_iter, temperature_rate, 
                                                                        temperature_steps, all_gammas)                 
                        mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_xz, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                             err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list = \
                                        compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, param_positions_dict=param_positions_dict,
                                            plot_online=plot_online, mse_theta_full=mse_theta_full, err_theta_full=err_theta_full, fig_xz=fig_xz, mse_x_list=mse_x_list, 
                                            mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list,
                                            err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list, per_param_sq_ers=per_param_sq_ers, 
                                            per_param_ers=per_param_ers, per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, seedint=seedint)                            
                        #########################
                        if testparam is not None:
                            for param in parameter_names:
                                if param == testparam:
                                    if vector_test_coordinate is not None:    
                                        raise NotImplementedError("Why here?!")                                           
                                    else:
                                        testidx_val_tmp = theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d].copy()                                     
                                        if perturb:
                                            theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                        else:
                                            theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_true[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                        theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d] = testidx_val_tmp.copy()
                                else:
                                    if perturb:
                                        # +/- 10% of the norm of the parameter
                                        theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_perturb[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                                    else:
                                        theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()   
                        #########################         
                        # check is correct here as we are only changing the testparam/testidx in debug mode - all the rest are fixed to true  
                        converged, delta_theta, random_restart = check_convergence(elementwise, theta_curr, theta_prev, param_positions_dict, param_positions_dict[target_param][1], 
                                                                            parameter_space_dim=parameter_space_dim, d=d, testparam=testparam, 
                                                                            testidx=testidx, p=percentage_parameter_change, tol=tol)
                        delta_rate_prev = delta_rate                                  
                        total_iter += 1    
                        if converged:
                            break
                    if converged:
                        break
                        
                fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, theta_curr.copy(), total_iter, fig_posteriors, 
                                                                                                        fig_posteriors_annealed, gamma, param_positions_dict, args, 
                                                                                                        plot_arrows=True, testparam=testparam, testidx=testidx, testvec=vector_index)   
                
                if random_restart and restarts < max_restarts:     
                    halved = False              
                    if (halving_rate <= max_halving):    
                        gamma, delta_rate_prev, temperature_rate, all_gammas, N = halve_annealing_rate_upd_schedule(N, gamma, 
                                                                                    delta_rate_prev, temperature_rate, temperature_steps, all_gammas,  
                                                                                    testparam=testparam)                    
                        halving_rate += 1
                        halved = True
                    plot_restarts.append((l, total_iter, halved, "fullrestart"))                              
                    restarts += 1                    
                    # keep solution
                    estimated_thetas.append(theta_curr)                                       
                    # random restart
                    theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                                    args, samples_list=theta_samples_list, idx_all=idx_all)                       
                    gamma = 0.01 
                    converged = False
                    ################################
                    # as when starting, just not set testparam at the edge of the grid
                    if testparam is not None:
                        for param in parameter_names:
                            if param == testparam:
                                if vector_test_coordinate is not None:  
                                    raise NotImplementedError("Why here?!")                                              
                                else:
                                    testidx_val_tmp = theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d].copy()                                  
                                    if perturb:
                                        theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_perturb[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                    else:
                                        theta_curr[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]] = theta_true[param_positions_dict[testparam][0]:param_positions_dict[testparam][1]].copy()
                                    theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d] = testidx_val_tmp.copy()
                            else:
                                if perturb:
                                    theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_perturb[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                                else:
                                    theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()
                        all_gammas = [1]*len(all_gammas)
                        gamma = 1
                    ################################
                    theta_prev = np.zeros((parameter_space_dim,))
                else:                    
                    theta_prev = theta_curr.copy()   
                    l += 1 
                print(l, total_iter, converged, random_restart, restarts, halving_rate)                                      
                           
            
        mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_xz, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
             err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list = \
                        compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, param_positions_dict=param_positions_dict,
                            plot_online=True, mse_theta_full=mse_theta_full, err_theta_full=err_theta_full, fig_xz=fig_xz, mse_x_list=mse_x_list, 
                            mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list,
                            err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list, per_param_sq_ers=per_param_sq_ers, 
                            per_param_ers=per_param_ers, per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, seedint=seedint)  
        fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, theta_curr.copy(), total_iter, fig_posteriors, 
                                                                                        fig_posteriors_annealed, gamma, param_positions_dict, args, 
                                                                                        plot_arrows=True, testparam=testparam, testidx=testidx, testvec=vector_index) 
        
        if converged and (len(estimated_thetas)==0 or (not np.all(np.isclose(theta_curr, estimated_thetas[-1][0])))):
            estimated_thetas.append((theta_curr, mse_x_list[-1], mse_z_list[-1], mse_x_nonRT_list[-1], mse_z_nonRT_list[-1]))
        
        elapsedtime = str(timedelta(seconds=time.time()-t0)) 
        rank_and_plot_solutions(estimated_thetas, elapsedtime, None, Y, J, K, d, parameter_names, dst_func, param_positions_dict, DIR_out, args)

        if elementwise:
            teststep = 1
        else:
            if target_param1 in ["X", "Z", "Phi"]:
                teststep = d
            elif target_param1 in ["alpha"]:
                teststep = J
            elif target_param1 in ["beta"]:
                teststep = K
            else:
                # skip
                teststep = 1
        iiidx += teststep
        temperature_rate = temperature_rate_init.copy()
        temperature_steps = temperature_steps_init.copy()
        
    return estimated_thetas


def icm_posterior_power_annealing(Y, param_positions_dict, args, temperature_rate=None, temperature_steps=None, 
                                plot_online=False, percentage_parameter_change=1, fastrun=False, 
                                data_annealing=False, annealing_prev=None, theta_part_annealing=None, 
                                max_restarts=2, max_partial_restarts=2, max_halving=2, seedint=1234):
  
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args


    rng = np.random.default_rng()
    testparam = None
    testidx = None 
    vector_index = None
    theta_samples_list = None
    base2exponent = 5
    idx_all = None

    # # try loading icm data annealing solution and init from there, else random
    # try:
    #     theta_curr = np.zeros((parameter_space_dim,))
    #     init_solution_dir = "{}/{}/estimation_ICM_data_annealing_evaluate_posterior_elementwise/".format(data_location, m)
    #     init_file = "{}/params_out_global_theta_hat_test.jsonl".format(init_solution_dir)
    #     with jsonlines.open(init_file, "r") as f:
    #         for result in f.iter(type=dict, skip_invalid=True):
    #             for param in parameter_names:
    #                 theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
    #     print("Loaded starting point from ICM data tempering solution.")
    #     theta_samples_list = []
    #     idx_all = []
    #     loaded_icm_data_tempering = True
    # except:
    theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict, 
                                                                args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)
       
    gamma = 0.1
    l = 0
    i = 0    
    theta_prev = np.zeros((parameter_space_dim,))

    if data_annealing:
        gamma = 1
        all_gammas = np.ones((10,)).tolist()
        N = len(all_gammas)
        if theta_part_annealing is not None:
            theta_curr = data_annealing_init_theta_given_theta_prev(theta_curr.copy(), theta_part_annealing, 
                                                                    K, J, d, param_positions_dict, 
                                                                    parameter_names, annealing_prev, 
                                                                    diff_elementwise=(elementwise and (not evaluate_posterior)))
    else:
        all_gammas = []
        for gidx in range(len(temperature_steps[1:])):
            upperlim = temperature_steps[1+gidx]        
            start = gamma if gidx==0 else all_gammas[-1]        
            all_gammas.extend(np.arange(start, upperlim, temperature_rate[gidx]))            
        N = len(all_gammas)
        print("Annealing schedule: {}".format(N))       

    # print_probab_per_coord_iter = 0.25
    # print_probab_per_full_scan_iter = 0.6

    delta_rate_prev = None
    mse_theta_full = []
    err_theta_full = []
    fig_x = None
    fig_z = None
    fig_x_err = None
    fig_z_err = None
    per_param_sq_ers = dict()
    per_param_ers = dict()
    plotting_thetas = dict()
    per_param_heats = dict()
    fig_posteriors = dict()
    fig_posteriors_annealed = dict()
    for param in parameter_names:
        plotting_thetas[param] = []
        per_param_sq_ers[param] = []        
        per_param_ers[param] = []        
        fig_posteriors[param] = None
        fig_posteriors_annealed[param] = None
        if param in ["gamma", "delta", "sigma_e"]:
            continue
        else:
            per_param_heats["{}_sq_e".format(param)] = []
            per_param_heats["{}_e".format(param)] = []
    per_param_heats["theta_sq_e"] = []
    per_param_heats["theta_e"] = []   
    per_param_sq_ers["X_rot_translated_mseOverMatrix"] = []
    per_param_sq_ers["Z_rot_translated_mseOverMatrix"] = []
    per_param_sq_ers["X_mseOverMatrix"] = []
    per_param_sq_ers["Z_mseOverMatrix"] = []    
    per_param_ers["X_rot_translated_errOverMatrix"] = []
    per_param_ers["Z_rot_translated_errOverMatrix"] = []
    per_param_ers["X_errOverMatrix"] = []
    per_param_ers["Z_errOverMatrix"] = []    
    plot_restarts = []
    xbox = []
    total_iter = 1   
    halving_rate = 0 
    restarts = 0
    estimated_thetas = []

    mse_x_list = []
    mse_z_list = []
    err_x_list = []
    err_z_list = []
    mse_x_nonRT_list = []
    mse_z_nonRT_list = []
    err_x_nonRT_list = []
    err_z_nonRT_list = []
    xbox = []

    if plot_online and not fastrun:
        # to plot X before it has moved for the first time
        fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, theta_curr.copy(), 
                                                                                    total_iter, fig_posteriors, fig_posteriors_annealed, gamma, 
                                                                                    param_positions_dict, args, plot_arrows=True, testparam=testparam, 
                                                                                    testidx=testidx, testvec=vector_index) 
    converged = False
    random_restart = False  
    
    while ((L is not None and l < L)) and (not converged):
        converged = False
        random_restart = False
        if elementwise:
            i = 0
            # t00 = time.time()
            while i < parameter_space_dim:                                            
                target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=i, d=d) 
                
                if ( ( target_param == "gamma" or target_param == "sigma_e" ) and (l > 10 and not (l in [14, 19, 24])) ): #( l > 10  and not (l % 5 == 0) ) ):  
                        i += 1 
                        continue                         
                # t00 = time.time()
                theta_test, _ = optimise_posterior_elementwise(target_param, i, vector_index_in_param_matrix, vector_coordinate, 
                                                            Y, gamma, theta_curr.copy(), param_positions_dict, L, args)    
                # milliseconds = (time.time()-t00)*1000                     
                # out_file = "/mnt/hdd2/ioannischalkiadakis/timings_element_parallel_500.jsonl"
                # with open(out_file, 'a') as f:         
                #     writer = jsonlines.Writer(f)
                #     writer.write({"K":K, "J":J, "block_size_rows": None, 
                #                 "block_size_cols":None, "parallel": 1, 
                #                 "milliseconds":milliseconds})
                
                theta_curr = theta_test.copy()                    
                gamma, delta_rate = update_annealing_temperature(gamma, total_iter, temperature_rate, temperature_steps, all_gammas)      
                # if data_annealing:
                #     assert gamma == 1

                if plot_online:
                    subset_coord2plot = None
                    # subset_coord2plot = [10, 75, 82, 115, 120, 121]
                    mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, _fig_x, _fig_z, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                         err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, _fig_x_err, _fig_z_err = \
                                compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, 
                                    param_positions_dict=param_positions_dict, plot_online=plot_online, 
                                    mse_theta_full=mse_theta_full, err_theta_full=err_theta_full,
                                    fig_x=None, fig_z=None, fig_x_err=None, fig_z_err=None, 
                                    mse_x_list=mse_x_list, mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, 
                                    mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list, 
                                    err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list,
                                    per_param_sq_ers=per_param_sq_ers, per_param_ers=per_param_ers,
                                    per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, 
                                    fastrun=fastrun, target_param=target_param, subset_coord2plot=subset_coord2plot, seedint=seedint)      
                delta_rate_prev = delta_rate                                                                    
                total_iter += 1   
                i += 1   
                if i % 20000 == 0:
                    # milliseconds = (time.time()-t00)*1000  
                    # print(milliseconds)
                    print(i, l, target_param)                          
            
            # last entry in mse lists in the same, has been stored twice
            if plot_online: # and l==4: ################################################################
                subset_coord2plot = None
                # subset_coord2plot = [10, 75, 82, 115, 120, 121]
                mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, \
                    fig_x, fig_z, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                    err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, fig_x_err, fig_z_err = \
                                compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter+1, args=args, 
                                    param_positions_dict=param_positions_dict, plot_online=plot_online, 
                                    mse_theta_full=mse_theta_full, err_theta_full=err_theta_full,
                                    fig_x=fig_x, fig_z=fig_z, fig_x_err=fig_x_err, fig_z_err=fig_z_err, 
                                    mse_x_list=mse_x_list, mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, 
                                    mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list, 
                                    err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list,
                                    per_param_sq_ers=per_param_sq_ers, per_param_ers=per_param_ers,
                                    per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, 
                                    fastrun=fastrun, subset_coord2plot=subset_coord2plot, seedint=seedint)
            # plot posteriors during estimation   
            if plot_online and not fastrun: # and l==4:      ########################################################
                fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, 
                                                                                                        theta_curr.copy(), total_iter, fig_posteriors, 
                                                                                                        fig_posteriors_annealed, gamma, 
                                                                                                        param_positions_dict, args, plot_arrows=True,
                                                                                                        testparam=testparam, testidx=testidx)  

            converged, delta_theta, random_restart = check_convergence(elementwise, theta_curr, theta_prev, param_positions_dict, i, 
                                                                        parameter_space_dim=parameter_space_dim, testparam=testparam, 
                                                                        testidx=testidx, p=percentage_parameter_change, tol=tol)
        else:
            for target_param in parameter_names:                  
                # limit costly updates 
                if ((target_param == "gamma" or target_param == "sigma_e") and (l == 0 or not (l % 5 == 0))):                    
                    continue                      
                if target_param in ["X", "beta"]:  
                    param_no = K                                                      
                elif target_param in ["Z", "Phi", "alpha"]:
                    param_no = J                                                        
                else:
                    # scalars
                    param_no = 1                    
                for idx in range(param_no):
                    
                    # t00 = time.time()
                    theta_test, _ = optimise_posterior_vector(target_param, idx, Y, gamma, theta_curr.copy(), 
                                                            param_positions_dict, L, args)                       
                    # milliseconds = (time.time()-t00)*1000                     
                    # out_file = "/mnt/hdd2/ioannischalkiadakis/timings_vector_parallel_1k.jsonl"
                    # with open(out_file, 'a') as f:         
                    #     writer = jsonlines.Writer(f)
                    #     writer.write({"K":K, "J":J, "block_size_rows": None, 
                    #                 "block_size_cols":None, "parallel": 1, 
                    #                 "milliseconds":milliseconds})
            
                    theta_curr = theta_test.copy()
                    gamma, delta_rate = update_annealing_temperature(gamma, total_iter, temperature_rate, 
                                                                    temperature_steps, all_gammas)
                    if plot_online: 
                        subset_coord2plot = None
                        # subset_coord2plot = [10, 75, 82, 115, 120, 121]                       
                        mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, _fig_x, _fig_z,\
                            per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                            err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, fig_x_err, fig_z_err = \
                                compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, 
                                                    param_positions_dict=param_positions_dict,
                                                    plot_online=plot_online, mse_theta_full=mse_theta_full, err_theta_full=err_theta_full, 
                                                    fig_x=None, fig_z=None, fig_x_err=None, fig_z_err=None, mse_x_list=mse_x_list, 
                                                    mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, mse_z_nonRT_list=mse_z_nonRT_list, 
                                                    err_x_list=err_x_list, err_z_list=err_z_list, err_x_nonRT_list=err_x_nonRT_list, 
                                                    err_z_nonRT_list=err_z_nonRT_list, per_param_sq_ers=per_param_sq_ers, per_param_ers=per_param_ers, 
                                                    per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, fastrun=fastrun, 
                                                    target_param=target_param, subset_coord2plot=subset_coord2plot)     

                    delta_rate_prev = delta_rate                                                                        
                    total_iter += 1    
                    if total_iter % 20000 == 0:
                        print(total_iter, l)                  
            
            # last entry in mse lists in the same, has been stored twice
            if plot_online:        
                subset_coord2plot = None
                # subset_coord2plot = [10, 75, 82, 115, 120, 121]        
                mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_x, fig_z, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                    err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, fig_x_err, fig_z_err = \
                                    compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter+1, args=args, 
                                        param_positions_dict=param_positions_dict, plot_online=plot_online, 
                                        mse_theta_full=mse_theta_full, err_theta_full=err_theta_full, fig_x=fig_x, 
                                        fig_z=fig_z, fig_x_err=fig_x_err, fig_z_err=fig_z_err, mse_x_list=mse_x_list, 
                                        mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, 
                                        mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list, 
                                        err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list, 
                                        per_param_sq_ers=per_param_sq_ers, per_param_ers=per_param_ers,
                                        per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, 
                                        fastrun=fastrun, subset_coord2plot=subset_coord2plot)  
            
            # plot posteriors during estimation 
            if plot_online and not fastrun:       
                fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, 
                                                                                    theta_curr.copy(), total_iter, fig_posteriors, 
                                                                                    fig_posteriors_annealed, gamma, param_positions_dict, args, 
                                                                                    plot_arrows=True, testparam=testparam, testidx=testidx, 
                                                                                    testvec=vector_index)                               
            converged, delta_theta, random_restart = check_convergence(elementwise, theta_curr, theta_prev, param_positions_dict, total_iter, 
                                                                        parameter_space_dim=parameter_space_dim, d=d, testparam=testparam, 
                                                                        testidx=testidx, p=percentage_parameter_change, tol=tol) 
            
        if random_restart and restarts < max_partial_restarts + max_restarts:   
            restarts += 1  
            if not plot_online:
                estimated_thetas.append((theta_curr, None, None, None, None, None, None, None, None))
            else:
                subset_coord2plot = None
                # subset_coord2plot = [10, 75, 82, 115, 120, 121]
                mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_x, fig_z, per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
                    err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, fig_x_err, fig_z_err = \
                    compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, param_positions_dict=param_positions_dict,
                                        plot_online=plot_online, mse_theta_full=mse_theta_full, err_theta_full=err_theta_full, fig_x=fig_x, 
                                        fig_z=fig_z, fig_x_err=fig_x_err, fig_z_err=fig_z_err, mse_x_list=mse_x_list, mse_z_list=mse_z_list, 
                                        mse_x_nonRT_list=mse_x_nonRT_list, mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, 
                                        err_z_list=err_z_list, err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list,
                                        per_param_sq_ers=per_param_sq_ers, per_param_ers=per_param_ers, per_param_heats=per_param_heats, 
                                        xbox=xbox, plot_restarts=plot_restarts, fastrun=fastrun, subset_coord2plot=subset_coord2plot, seedint=seedint) 
                if fastrun is False:
                    estimated_thetas.append((theta_curr, mse_x_list[-1], mse_z_list[-1], mse_x_nonRT_list[-1], mse_z_nonRT_list[-1], \
                                        err_x_list[-1], err_z_list[-1], err_x_nonRT_list[-1], err_z_nonRT_list[-1])) 
                else:
                    estimated_thetas.append((theta_curr, None, None, None, None, None, None, None, None))
                mse_x_list = []
                mse_z_list = []
                mse_x_nonRT_list = []
                mse_z_nonRT_list = []
                err_x_list = []
                err_z_list = []                
                err_x_nonRT_list = []
                err_z_nonRT_list = []
            converged = False
            halved = False            
            theta_prev = theta_curr.copy()                   
            if restarts <= max_partial_restarts:                        
                # random restart, only for unconverged coordinates
                uncovergedpart = np.argwhere(delta_theta > tol)
                theta_curr_full_upd, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                                args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)
                theta_curr[uncovergedpart] = theta_curr_full_upd[uncovergedpart].copy()    
                if data_annealing:
                    gamma = 1
                else:                  
                    gamma = 0.1
                plot_restarts.append((l, total_iter, halved, "partialrestart"))
            else:                                                                        
                # random restart, completely from scratch
                theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                                args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)                     
                if data_annealing:
                    gamma = 1
                else: 
                    gamma = 0.1
                theta_prev = np.zeros((parameter_space_dim,))  
                plot_restarts.append((l, total_iter, halved, "fullrestart"))
        else:
            halved = False
            if ((halving_rate < max_halving) and (not data_annealing) and (l >= 15)):              
                gamma, delta_rate_prev, temperature_rate, all_gammas, N = halve_annealing_rate_upd_schedule(N, gamma, 
                                                                        delta_rate_prev, temperature_rate, temperature_steps, all_gammas,  
                                                                        testparam=testparam)                    
                halving_rate += 1 
                halved = True
            
            theta_prev = theta_curr.copy() 
            if restarts >= max_partial_restarts + max_restarts:
                random_restart = False

        print("Annealing schedule repeat no.: {}".format(l))
        print("Total conditional posterior evaluations: {}".format(total_iter))
        print("Convergence: {}".format(converged))
        print("Random restart: {}".format(random_restart))
        l += 1           

    # last update of gamma, sigma_e
    # if elementwise:
    #     i = 0                 
    #     while i < parameter_space_dim:     
    #         target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=i, d=d) 
    #         if not (target_param == "gamma" or target_param == "sigma_e"):
    #             i += 1
    #             continue                   
    #         theta_test, _ = optimise_posterior_elementwise(target_param, i, vector_index_in_param_matrix, vector_coordinate, 
    #                                                     Y, gamma, theta_curr.copy(), param_positions_dict, L, args)                   
    #         theta_curr = theta_test.copy()
    #         i += 1
    # else:
    #     for target_param in ["gamma", "sigma_e"]:              
    #         idx = 0
    #         theta_test, _ = optimise_posterior_vector(target_param, idx, Y, gamma, theta_curr.copy(), 
    #                                                 param_positions_dict, L, args)     
    #         theta_curr = theta_test.copy()

    subset_coord2plot = None
    # subset_coord2plot = [10, 75, 82, 115, 120, 121]
    mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_x, fig_z, \
        per_param_sq_ers, per_param_ers, per_param_heats, xbox, \
        err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, fig_x_err, fig_z_err = \
                    compute_and_plot_mse(theta_true, theta_curr, l, iteration=total_iter, args=args, 
                                        param_positions_dict=param_positions_dict, plot_online=plot_online, mse_theta_full=mse_theta_full, 
                                        err_theta_full=err_theta_full, fig_x=fig_x, fig_z=fig_z, fig_x_err=fig_x_err, 
                                        fig_z_err=fig_z_err, mse_x_list=mse_x_list, mse_z_list=mse_z_list, mse_x_nonRT_list=mse_x_nonRT_list, 
                                        mse_z_nonRT_list=mse_z_nonRT_list, err_x_list=err_x_list, err_z_list=err_z_list, 
                                        err_x_nonRT_list=err_x_nonRT_list, err_z_nonRT_list=err_z_nonRT_list, per_param_sq_ers=per_param_sq_ers, 
                                        per_param_ers=per_param_ers, per_param_heats=per_param_heats, xbox=xbox, plot_restarts=plot_restarts, 
                                        fastrun=fastrun, subset_coord2plot=subset_coord2plot, seedint=seedint)  
    if plot_online and not fastrun:
        fig_posteriors, fig_posteriors_annealed, plotting_thetas = plot_posteriors_during_estimation(Y, total_iter, plotting_thetas, theta_curr.copy(), i, fig_posteriors, 
                                                                                    fig_posteriors_annealed, gamma, param_positions_dict, args, 
                                                                                    plot_arrows=True, testparam=testparam, testidx=testidx, testvec=vector_index) 
    
    
    if converged and (len(estimated_thetas)==0 or (not np.all(np.isclose(theta_curr, estimated_thetas[-1][0])))):
        if fastrun is False:
            estimated_thetas.append((theta_curr, mse_x_list[-1], mse_z_list[-1], mse_x_nonRT_list[-1], mse_z_nonRT_list[-1],\
                                err_x_list[-1], err_z_list[-1], err_x_nonRT_list[-1], err_z_nonRT_list[-1]))
        else:
            estimated_thetas.append((theta_curr, None, None, None, None, None, None, None, None))
    
    
    return estimated_thetas

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
        max_restarts=2, max_partial_restarts=2, max_halving=2, plot_online=False, seedint=1234):
        
        for m in range(trialsmin, trialsmax, 1):
            if elementwise:
                if evaluate_posterior:                    
                    DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_icmptest/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_gamma1/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_gamma1_perturb/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_annealing/".format(data_location, m)
                    # DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise_annealing_perturb/".format(data_location, m)
                else:
                    DIR_out = "{}/{}/estimation_ICM_differentiate_posterior_elementwise/".format(data_location, m)
            else:
                if evaluate_posterior:
                    DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector/".format(data_location, m)
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
            
            # start efficiency monitoring - interval in seconds
            print_threadpool_info()
            monitor = Monitor(interval=30)
            monitor.start()            

            t_start = time.time()            
            thetas = icm_posterior_power_annealing(Y, param_positions_dict, args, temperature_rate=temperature_rate, 
                                                temperature_steps=temperature_steps, plot_online=plot_online, 
                                                percentage_parameter_change=percentage_parameter_change, fastrun=fastrun,
                                                data_annealing=False, annealing_prev=None, theta_part_annealing=None,
                                                max_restarts=max_restarts, max_partial_restarts=max_partial_restarts, 
                                                max_halving=max_halving, seedint=seedint)
            t_end = time.time()
            monitor.stop()
            wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
            efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                        avg_threads, max_threads, avg_processes, max_processes)
            elapsedtime = str(timedelta(seconds=t_end - t_start))   

            thetas_and_errors = []
            params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)            
            for solution in thetas:
                theta_curr = solution[0]
                params_hat = optimisation_dict2params(theta_curr, param_positions_dict, J, K, d, parameter_names)
                for param in ["X", "Z"]:
                    if param == "X":                        
                        X_hat_vec = np.asarray(params_hat[param]).reshape((d*K,), order="F")       
                        X_true_vec = np.asarray(params_true[param]).reshape((d*K,), order="F")       
                        x_rel_err = (X_true_vec - X_hat_vec)/X_true_vec
                        x_sq_err = x_rel_err**2
                    elif param == "Z":                               
                        Z_hat_vec = np.asarray(params_hat[param]).reshape((d*J,), order="F")         
                        Z_true_vec = np.asarray(params_true[param]).reshape((d*J,), order="F")       
                        z_rel_err = (Z_true_vec - Z_hat_vec)/Z_true_vec
                        z_sq_err = z_rel_err**2
                thetas_and_errors.append((theta_curr, None, None, np.mean(x_sq_err), np.mean(z_sq_err), None, None, np.mean(x_rel_err), np.mean(z_rel_err)))

            rank_and_plot_solutions(thetas_and_errors, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, 
                                    dst_func, param_positions_dict, DIR_out, args, seedint=seedint, get_RT_error=True)


if __name__ == "__main__":

    # datain = "/mnt/hdd2/ioannischalkiadakis/idealdata_plotstest/data_K30_J10_sigmae001/"
    # estimation_folder = "estimation_ICM_evaluate_posterior_elementwise_May2"
    # error_polarisation_plots(datain, estimation_folder, M=1, K=30, J=10, d=2)

    # files = ["/mnt/hdd2/ioannischalkiadakis/timings_element_parallel_500.jsonl", "/mnt/hdd2/ioannischalkiadakis/timings_element_parallel_1k.jsonl", "/mnt/hdd2/ioannischalkiadakis/timings_vector_parallel_1k.jsonl", "/mnt/hdd2/ioannischalkiadakis/timings_vector_numba.jsonl"]
    # names = ["element-parallel-block-500", "element-parallel-block-1k", "vector-parallel-block-1k", "vector-numba-only"]
    # plot_posterior_vec_runtimes(files, names, outdir="/mnt/hdd2/ioannischalkiadakis/idealdata_testmle/data_K10000_J100_sigmae0001/0/")
    # import sys
    # sys.exit(0)

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
    # before halving annealing rate
    percentage_parameter_change = 0.2

    # if not parallel:
    #     jax.default_device = jax.devices("cpu")[0]
    #     jax.config.update("jax_traceback_filtering", "off")
    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    penalty_weight_Z = 0.0
    constant_Z = 0.0
    retries = 20
    diff_iter = None
    disp = False
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]
    d = 2  
    gridpoints_num = 30 #, 15, 30
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
    temperature_steps = [0, 1, 2, 5, 10]
    temperature_rate = [1e-3, 1e-2, 1e-1, 1]

    # temperature_steps = [0, 1, 2, 5, 10]
    # temperature_rate = [0.1, 0.2, 0.5, 1] 

    niter = 10
    fastrun = True
    max_restarts = 2
    max_partial_restarts = 2
    max_halving = 2
    plot_online = False
    max_signal2noise_ratio = 25 # in dB   # max snr

    min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
    print(min_sigma_e)

    tol = 1e-6    
    #/home/ioannischalkiadakis/ideal
    # data_location = "./idealpestimation/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))
    data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata_plotstest/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e_true).replace(".", ""))
    # data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata_testmle/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 30      

    # with open("/mnt/hdd2/ioannischalkiadakis/idealdata_testmle/data_K10000_J100_sigmae0001/0/Utilities.pickle", "rb") as f:
    #     utilities = pickle.load(f)
    # ipdb.set_trace()
    # fig = go.Figure()
    # sigma_e_true_num = 0.001
    # fig.add_trace(go.Scatter(y=utilities.reshape((K*J,))/sigma_e_true_num))
    # fig.add_hline(y=3/sigma_e_true_num)
    # fig.add_hline(y=-3/sigma_e_true_num)
    # fig.show()
    # ipdb.set_trace()


    args = (None, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, None, tol,                     
            None, None, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
            gridpoints_num, diff_iter, disp, min_sigma_e, None)  

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
        elementwise=elementwise, evaluate_posterior=evaluate_posterior, temperature_rate=temperature_rate, 
        temperature_steps=temperature_steps, L=niter, tol=tol, 
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
        prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, 
        prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma, 
        prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta,         
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
        gridpoints_num=gridpoints_num, diff_iter=diff_iter, disp=disp, theta_true=theta_true,
        percentage_parameter_change=percentage_parameter_change, min_sigma_e=min_sigma_e, fastrun=fastrun,
        max_restarts=max_restarts, max_partial_restarts=max_partial_restarts, max_halving=max_halving, plot_online=plot_online, seedint=seed_value)
    
    ## tig = TruncatedInverseGamma(alpha=prior_loc_sigmae, beta=prior_scale_sigmae, lower=1e-5, upper=max_sigma_e)
    ## x = np.linspace(0, 2*max_sigma_e, 500)
    ## pdfs = tig.pdf(x)
    ## fig = go.Figure(data=go.Scatter(
    ##                     x=x,
    ##                     y=pdfs,
    ##                     mode='lines',
    ##                     name=param,
    ##                     line=dict(
    ##                         color='royalblue',
    ##                         width=2
    ##                     ),
    ##                     hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
    ##     ))
    ## fig.show()
    
    # For posterior plotting: run in vector mode
    # args = (data_location, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, niter, tol,                     
    #         parameter_space_dim, 0, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
    #         prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
    #         prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
    #         gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)
    # with open("{}/{}/Y.pickle".format(data_location, 0), "rb") as f:
    #         Y = pickle.load(f)
    # Y = Y.astype(np.int8).reshape((K, J), order="F")    
    
    # outdir = "{}/posterior_plots_update/".format(data_location)
    # pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)     
    # param_positions_dict = dict()            
    # k = 0
    # for param in parameter_names:
    #     if param == "X":
    #         param_positions_dict[param] = (k, k + K*d)                       
    #         k += K*d    
    #     elif param in ["Z"]:
    #         param_positions_dict[param] = (k, k + J*d)                                
    #         k += J*d
    #     elif param in ["Phi"]:            
    #         param_positions_dict[param] = (k, k + J*d)                                
    #         k += J*d
    #     elif param == "beta":
    #         param_positions_dict[param] = (k, k + K)                                   
    #         k += K    
    #     elif param == "alpha":
    #         param_positions_dict[param] = (k, k + J)                                       
    #         k += J    
    #     elif param == "gamma":
    #         param_positions_dict[param] = (k, k + 1)                                
    #         k += 1
    #     elif param == "delta":
    #         param_positions_dict[param] = (k, k + 1)                                
    #         k += 1
    #     elif param == "sigma_e":
    #         param_positions_dict[param] = (k, k + 1)                                
    #         k += 1
    # with jsonlines.open("{}/0/synthetic_gen_parameters.jsonl".format(data_location), "r") as f:
    #     for result in f.iter(type=dict, skip_invalid=True):
    #         for param in parameter_names:
    #             theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
    # theta_curr = theta_true.copy()
    
    # for param in parameter_names:
    #     outdir = "{}/posterior_plots_update/{}/".format(data_location, param)
    #     pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)     
    #     if param in ["X", "beta"]:
    #         for i in range(K):                
    #             if param == "X":
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=i, vector_coordinate=None, 
    #                                         theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args, 
    #                                         true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d])
    #                 for j in range(d):
    #                     plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=i, vector_coordinate=j, 
    #                                             theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args,
    #                                             true_param=theta_true[param_positions_dict[param][0]+i*d+j])
    #             else:
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=i, vector_coordinate=i, 
    #                                         theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args,
    #                                         true_param=theta_true[param_positions_dict[param][0]+i])
    #     elif param in ["Z", "Phi", "alpha"]:
    #         for j in range(J):
    #             if param in ["Phi", "Z"]:
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=j, vector_coordinate=None, 
    #                                         theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args, 
    #                                         true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d])
    #                 for i in range(d):
    #                     plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=j, vector_coordinate=i, 
    #                                             theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args,
    #                                             true_param=theta_true[param_positions_dict[param][0]+j*d+i])
    #             else:
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=j, vector_coordinate=j, 
    #                                         theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args,
    #                                         true_param=theta_true[param_positions_dict[param][0]+j])
    #     else:
    #         plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=None, vector_coordinate=0, 
    #                                 theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args,
    #                                 true_param=theta_true[param_positions_dict[param][0]])



    