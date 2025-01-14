import os 
import sys
import time 
import ipdb
import jax
import pathlib
import pickle
import jsonlines
import numpy as np
import random
import itertools
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
from idealpestimation.src.parallel_manager import jsonlines, ProcessManager
from idealpestimation.src.utils import log_conditional_posterior_x_vec,  \
                                        log_conditional_posterior_phi_vec, \
                                            log_conditional_posterior_z_vec, \
                                                log_conditional_posterior_alpha_j, \
                                                    log_conditional_posterior_beta_i, \
                                                      log_conditional_posterior_gamma, \
                                                        log_conditional_posterior_delta, \
                                                          log_conditional_posterior_mu_e, \
                                                            log_conditional_posterior_sigma_e, \
                                                              log_conditional_posterior_x_il, \
                                                                log_conditional_posterior_phi_jl, \
                                                                    log_conditional_posterior_z_jl, \
                                                                        qmc, fix_plot_layout_and_save, \
                                                                            create_constraint_functions_icm, \
                                                                                optimisation_dict2paramvectors



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
        grid_and_optim_outcome["elapsedtime_seconds"] = str(timedelta(seconds=time.time()-t0))   
        time_obj = datetime.strptime(grid_and_optim_outcome["elapsedtime_seconds"], '%H:%M:%S.%f')
        hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
        grid_and_optim_outcome["elapsedtime_hours"] = hours  
        if isinstance(gridpoint, float):
            grid_and_optim_outcome["gridpoint"] = gridpoint
        else:      
            grid_and_optim_outcome["gridpoint"] = list(gridpoint)
        grid_and_optim_outcome["posterior"] = posterior_eval.tolist()[0]
        # print(posterior_eval.tolist()[0])
        grid_and_optim_outcome["gamma_annealing"] = gamma_annealing
        grid_and_optim_outcome["icm_iteration"] = l
            
        if vector_coordinate is not None:
            out_file = "{}/{}/{}/posterior_evaluation_anneal_power_{}_param_idx_{}_vectorcoordinate_{}.jsonl".format(DIR_out, param, l, gamma_annealing, idx, vector_coordinate)
        else:
            out_file = "{}/{}/{}/posterior_evaluation_anneal_power_{}_param_idx_{}.jsonl".format(DIR_out, param, l, gamma_annealing, idx)
        pathlib.Path("{}/{}/{}/".format(DIR_out, param, l)).mkdir(parents=True, exist_ok=True)     
        self.append_to_json_file(grid_and_optim_outcome, output_file=out_file)


def get_parameter_name_and_vector_coordinate(param_positions_dict, i, d):

    for param in param_positions_dict.keys():
        if param_positions_dict[param][0] <= i and param_positions_dict[param][1] > i: # not equal to upper limit
            target_param = param
            index_in_flat_vector = i - param_positions_dict[param][0]            
            if target_param in ["X", "Z", "Phi"]:                
                column = index_in_flat_vector // d
                row = index_in_flat_vector % d
                # print(target_param, i, column, row)
                return target_param, column, row
            else:
                # alpha, beta are vectors, not flattened matrices
                return target_param, index_in_flat_vector, index_in_flat_vector
        else:
            continue
    raise AttributeError("Should never reach this point.")

def get_evaluation_grid(param, vector_coordinate, args):

    grid_width_std = 3
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args
    
    if param == "alpha":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, gridpoints_num).tolist()
    elif param == "beta":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, gridpoints_num).tolist()
    elif param == "gamma":            
        grid = np.linspace(-2, 2, gridpoints_num).tolist()
        if 0.0 in grid:
            grid.remove(0.0)
    elif param == "delta":                    
        grid = np.linspace(-2, 2, gridpoints_num).tolist()
        if 0.0 in grid:
            grid.remove(0.0)        
    elif param == "mu_e":
        grid = np.linspace(-2, 2, gridpoints_num).tolist()
    elif param == "sigma_e":
        grid = np.linspace(0.0001, 1.5, gridpoints_num).tolist()
    else:        
        if d == 1 or elementwise:
            if param == "Phi":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], gridpoints_num).tolist()                
            elif param == "Z":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], gridpoints_num).tolist()                
            elif param == "X":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], gridpoints_num).tolist()                
        elif (d > 1 and d <= 5):
            if param == "Phi":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
            elif param == "Z":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
            elif param == "X":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
        else:
            raise NotImplementedError("Use a Sobol sequence to generate a grid in such high dimensional space.")

    return grid

def plot_posterior_elementwise(outdir, param, Y, idx, vector_coordinate, theta_curr, gamma, param_positions_dict, args):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args
    
    f = get_posterior_for_optimisation_vec(param=param, Y=Y, idx=idx, vector_coordinate=vector_coordinate, theta_curr=theta_curr, 
                                           gamma=gamma, param_positions_dict=param_positions_dict, args=args)
    
    if elementwise and vector_coordinate is None:
        xx_ = np.linspace(-2, 2, 30)
        xx = itertools.product(*[xx_, xx_])    
        yy = np.asarray([f(x)[0] for x in xx]).flatten()
    else:
        if param == "gamma":
            xx_ = np.linspace(-5, 5, 300)
        elif param == "mu_e":
            xx_ = np.linspace(-10, 10, 500)
        elif param == "sigma_e":
            xx_ = np.linspace(0.0001, 1.5, 300)
        else:
            xx_ = np.linspace(-2, 2, 300)
        yy = np.asarray([f(x)[0] for x in xx_]).flatten()
        
    if vector_coordinate is not None:
        fig = go.Figure(data=go.Scatter(
                        x=[xxx for xxx in xx_],
                        y=yy,
                        mode='lines',
                        name=param,
                        line=dict(
                            color='royalblue',
                            width=2
                        ),
                        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
    else:        
        fig = go.Figure(data=
                    go.Contour(
                        x=xx_,
                        y=xx_,
                        z=yy,
                        colorscale='Hot',
                        contours=dict(
                                    # start=np.min(yy)-10,
                                    # end=np.max(yy)+10,
                                    # size=0.1,
                                    showlabels=True
                                ),
                        colorbar=dict(
                            title='Value',
                            titleside='right'
                        )
                    )
                )
        fig.update_layout(                
                xaxis_title='x1',
                yaxis_title='x2',                
        )
    # fig.show()
        
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)    
    savename = "{}/{}_idx_{}_vector_coord_{}.html".format(outdir, param, idx, vector_coordinate)
    fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
    
def get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta_curr, gamma, param_positions_dict, args):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args

    if param == "X":
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_x_il(x, vector_coordinate, vector_index_in_param_matrix, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, 
                                                                  prior_loc_x[vector_coordinate], prior_scale_x[vector_coordinate, vector_coordinate], gamma)
        else:
            post2optim = lambda x: log_conditional_posterior_x_vec(x, idx, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x, prior_scale_x, gamma)  
    elif param == "Z":
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_z_jl(x, vector_coordinate, vector_index_in_param_matrix, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z[vector_coordinate], 
                                                                prior_scale_z[vector_coordinate, vector_coordinate], gamma, constant_Z, penalty_weight_Z)
        else:
            post2optim = lambda x: log_conditional_posterior_z_vec(x, idx, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z, prior_scale_z, gamma, 
                                                                constant_Z, penalty_weight_Z)
    elif param == "Phi":            
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_phi_jl(x, vector_coordinate, vector_index_in_param_matrix, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi[vector_coordinate], 
                                                                    prior_scale_phi[vector_coordinate, vector_coordinate], gamma)
        else:
            post2optim = lambda x: log_conditional_posterior_phi_vec(x, idx, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi, prior_scale_phi, gamma)
    elif param == "beta":
        if elementwise and isinstance(vector_coordinate, int):
            idx = vector_coordinate
        post2optim = lambda x: log_conditional_posterior_beta_i(x, idx, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_beta, prior_scale_beta, gamma)
    elif param == "alpha":
        if elementwise and isinstance(vector_coordinate, int):
            idx = vector_coordinate
        post2optim = lambda x: log_conditional_posterior_alpha_j(x, idx, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_alpha, prior_scale_alpha, gamma)
    elif param == "gamma":
        post2optim = lambda x: log_conditional_posterior_gamma(x, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=gamma) 
    elif param == "delta":
        post2optim = lambda x: log_conditional_posterior_delta(x, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, gamma)    
    elif param == "mu_e":
        post2optim = lambda x: log_conditional_posterior_mu_e(x, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, gamma)         
    elif param == "sigma_e":
        post2optim = lambda x: log_conditional_posterior_sigma_e(x, Y, theta_curr, J, K, d, parameter_names, dst_func, param_positions_dict, gamma) 
        
    return post2optim

def optimise_posterior_elementwise(param, idx, vector_index_in_param_matrix, vector_coordinate, Y, gamma, theta_curr, param_positions_dict, l, args):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args
    
    # Negate its result for the negative log posterior likelihood
    f = get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta_curr, gamma, param_positions_dict, args)
    # get grid depending on param
    grid = get_evaluation_grid(param, vector_coordinate, args)      
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
                    time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                    hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)   
                    print(elapsedtime)  
                    
                    # ipdb.set_trace()
            
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
            t0 = time.time()            
            for gridpoint in grid:
                posterior_eval = -f(gridpoint)
                if posterior_eval < min_f:
                    min_f = posterior_eval
                    param_estimate = gridpoint
            elapsedtime = str(timedelta(seconds=time.time()-t0))   
            time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
            hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)   
            print(elapsedtime)                 
            
            # ipdb.set_trace()
    
    else:
        # use previously evaluation grid as starting points grid
        retry = 0
        minus_f = lambda x: -f(x)
        t0 = time.time()
        while retry < retries:   
            print("Retry: {}".format(retry))
            gridpoint = random.choice(grid)
            optimize_kwargs = {
                'method': optimisation_method,
                'x0': np.asarray([gridpoint]),                
                'jac': '3-point'
            }                               
            bounds = create_constraint_functions_icm(parameter_space_dim, vector_coordinate, param=param, param_positions_dict=param_positions_dict, args=args)    
            if diff_iter is not None:
                result = minimize(minus_f, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxiter":diff_iter, "maxls":20})
            else:
                result = minimize(minus_f, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxls":20})
            
            param_estimate = result.x[0]          
            
            if result.success:
                break
            else:              
                grid.remove(gridpoint)
                retry += 1
       
    
    theta_curr[idx] = param_estimate
    
    return theta_curr

def optimise_posterior_vector(param, idx, Y, gamma, theta_curr, param_positions_dict, l, args):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args
    
    if evaluate_posterior:
        # Negate its result for the negative log posterior likelihood        
        f = get_posterior_for_optimisation_vec(param, Y, idx, idx, None, theta_curr, gamma, param_positions_dict, args)
        # get grid depending on param
        grid = get_evaluation_grid(param, None, args)        
        # parallel eval grid
        if parallel:
            manager = ProcessManagerSynthetic(total_running_processes)  
            try:   
                manager.create_results_dict(optim_target="all")    
                while True:                    
                    for gridpoint in grid:
                        worker_args = (f, gridpoint, gamma, l, DIR_out, param, idx) 
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
            t0 = time.time()
            for gridpoint in grid:
                posterior_eval = -f(gridpoint)
                if posterior_eval < min_f:
                    min_f = posterior_eval
                    param_estimate = gridpoint            
            elapsedtime = str(timedelta(seconds=time.time()-t0))   
            time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
            hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)                    
            # print(elapsedtime)
    else:
        raise NotImplementedError("More efficient to differentiate posterior coordinate-wise.")

    if param in ["X", "Z", "Phi"]:
        theta_curr[param_positions_dict[param][0] + idx*d:param_positions_dict[param][0] + (idx + 1)*d] = param_estimate
    else:
        # scalar updates and coordinate-wise updates of alphas/betas
        theta_curr[param_positions_dict[param][0] + idx:param_positions_dict[param][0] + (idx + 1)] = param_estimate

    return theta_curr


def icm_posterior_power_annealing(Y, param_positions_dict, args):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args

    # gamma = 1
    l = 0
    gamma = 3*delta_n**l
    delta_theta = np.inf
    theta_prev = np.zeros((parameter_space_dim,))

    sampler = qmc.Sobol(d=parameter_space_dim, scramble=False)   
    theta_curr = sampler.random_base2(m=4)[5]       
    
    delta_theta = np.inf
    while (L is not None and l < L) or abs(delta_theta) > tol:
        for n in range(1, N+1, 1):
            print(n, gamma)
            if elementwise:
                for i in range(parameter_space_dim):                    
                    target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=i, d=d)
                    theta_curr = optimise_posterior_elementwise(target_param, i, vector_index_in_param_matrix, vector_coordinate, Y, gamma, theta_curr, param_positions_dict, L, args)                                        
            else:
                for param in parameter_names:            
                    if param in ["X", "beta"]:  
                        for idx in range(K):
                            theta_curr = optimise_posterior_vector(param, idx, Y, gamma, theta_curr, param_positions_dict, L, args)                                                            
                    elif param in ["Z", "Phi", "alpha"]:
                        for idx in range(J):
                            theta_curr = optimise_posterior_vector(param, idx, Y, gamma, theta_curr, param_positions_dict, L, args)                                                            
                    else:
                        # scalars
                        theta_curr = optimise_posterior_vector(param, 0, Y, gamma, theta_curr, param_positions_dict, L, args)
            # gamma += delta_n
            gamma = 5*delta_n**l

        if L is not None:
            l += 1    
        print(theta_prev)    
        print(theta_curr)
        delta_theta = np.sum((theta_curr - theta_prev)**2)
        theta_prev = theta_curr.copy()       
        print(L, delta_theta)
    
    return theta_curr

def main(J=2, K=2, d=1, N=100, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trials=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, delta_n=0.1, L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False):

        for m in range(trials):
            if elementwise:
                if evaluate_posterior:
                    DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise/".format(data_location, m)
                else:
                    DIR_out = "{}/{}/estimation_ICM_differentiate_posterior_elementwise/".format(data_location, m)
            else:
                if evaluate_posterior:
                    DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector/".format(data_location, m)
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
                elif param == "mu_e":
                    param_positions_dict[param] = (k, k + 1)                                
                    k += 1
                elif param == "sigma_e":
                    param_positions_dict[param] = (k, k + 1)                                
                    k += 1

            args = (DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol,                     
                    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                    prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, gridpoints_num, 
                    diff_iter, disp)   
            t0 = time.time()
            theta = icm_posterior_power_annealing(Y, param_positions_dict, args)
            print(theta)

            params_out = dict()
            params_out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            params_out["elapsedtime"] = str(timedelta(seconds=time.time()-t0))   
            time_obj = datetime.strptime(params_out["elapsedtime"], '%H:%M:%S.%f')
            hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
            params_out["elapsedtime_hours"] = hours
            params_out["param_positions_dict"] = param_positions_dict
            for param in parameter_names:
                params_out[param] = theta[param_positions_dict[param][0]:param_positions_dict[param][1]]
                if isinstance(params_out[param], np.ndarray):
                    params_out[param] = params_out[param].tolist()
            out_file = "{}/params_out_global_theta_hat.jsonl".format(DIR_out)
            with open(out_file, 'a') as f:         
                writer = jsonlines.Writer(f)
                writer.write(params_out)

if __name__ == "__main__":

    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    parallel = False
    if not parallel:
        jax.default_device = jax.devices("cpu")[0]
        jax.config.update("jax_traceback_filtering", "off")
    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = 20
    penalty_weight_Z = 1000.0
    constant_Z = 0.0
    elementwise = True
    evaluate_posterior = True
    retries = 10
    diff_iter = None
    disp = True
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "mu_e", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "mu_e", "sigma_e"]
    M = 1
    K = 30
    J = 10
    d = 2  
    gridpoints_num = 20
    prior_loc_x = np.zeros((d,))
    prior_scale_x = np.eye(d)
    prior_loc_z = np.zeros((d,))
    prior_scale_z = np.eye(d)
    prior_loc_phi = np.zeros((d,))
    prior_scale_phi = np.eye(d)
    prior_loc_beta = 0
    prior_scale_beta = 0.5
    prior_loc_alpha = 0
    prior_scale_alpha = 0.5
    annealing_schedule_duration = 10
    # if using exponential annealing schedule
    delta_n = 0.9
    #delta_n = 0.05
    tol = 1e-6    
    sigma_e_true = 1      
    data_location = "/home/ioannischalkiadakis/ideal/idealpestimation/data_K{}_J{}_sigmae{}_nopareto/".format(K, J, str(sigma_e_true).replace(".", ""))
    # data_location = "/home/ioannis/Dropbox (Heriot-Watt University Team)/ideal/idealpestimation/data_K{}_J{}_sigmae{}_nopareto/".format(K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 200                 
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 4
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 3
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    main(J=J, K=K, d=d, N=annealing_schedule_duration, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, parameter_space_dim=parameter_space_dim, trials=M, 
        penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=retries, 
        elementwise=elementwise, evaluate_posterior=evaluate_posterior, delta_n=delta_n, L=niter, tol=tol, 
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
        prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, gridpoints_num=gridpoints_num, diff_iter=diff_iter, disp=disp)

    # args = (data_location, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, annealing_schedule_duration, delta_n, niter, tol,                     
    #                 parameter_space_dim, 0, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
    #                 prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, gridpoints_num, 
    #                 diff_iter, disp) 
    # with open("{}/{}/Y.pickle".format(data_location, 0), "rb") as f:
    #         Y = pickle.load(f)
    # Y = Y.astype(np.int8).reshape((K, J), order="F")    
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
    #     elif param == "mu_e":
    #         param_positions_dict[param] = (k, k + 1)                                
    #         k += 1
    #     elif param == "sigma_e":
    #         param_positions_dict[param] = (k, k + 1)                                
    #         k += 1
    # theta_curr = np.zeros((parameter_space_dim,))
    # with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_location), "r") as f:
    #     for result in f.iter(type=dict, skip_invalid=True):
    #         for param in parameter_names:
    #             theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
    # outdir = "{}/posterior_plots/".format(data_location)
    # pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)     
    
    # for param in parameter_names:
    #     outdir = "{}/posterior_plots/{}/".format(data_location, param)
    #     pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)     
    #     if param in ["X", "beta"]:
    #         for i in range(K):                
    #             if param == "X":
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=i, vector_coordinate=None, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)
    #                 for j in range(d):
    #                     plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=i, vector_coordinate=j, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)
    #             else:
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=i, vector_coordinate=i, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)
    #     elif param in ["Z", "Phi", "alpha"]:
    #         for j in range(J):
    #             if param in ["Phi", "Z"]:
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=j, vector_coordinate=None, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)
    #                 for i in range(d):
    #                     plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=j, vector_coordinate=i, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)
    #             else:
    #                 plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=j, vector_coordinate=j, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)
    #     else:
    #         plot_posterior_elementwise(outdir=outdir, param=param, Y=Y, idx=None, vector_coordinate=0, theta_curr=theta_curr, gamma=1, param_positions_dict=param_positions_dict, args=args)



    