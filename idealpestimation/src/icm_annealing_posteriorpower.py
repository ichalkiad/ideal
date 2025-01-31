import os 
import sys
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
from sklearn.decomposition import PCA
from plotly.validators.scatter.marker import SymbolValidator
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
                                                                                update_annealing_temperature, \
                                                                                    compute_and_plot_mse, time, datetime, \
                                                                                        timedelta, parse_input_arguments, \
                                                                                            halve_annealing_rate_upd_schedule, p_ij_arg, negative_loglik, TruncatedInverseGamma



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

    grid_width_std = 5    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true  = args
    gridpoints_num_alpha_beta = gridpoints_num*50
    
    if param == "alpha":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, gridpoints_num_alpha_beta).tolist()
    elif param == "beta":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, gridpoints_num_alpha_beta).tolist()
    elif param == "gamma":            
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, gridpoints_num_alpha_beta).tolist()
        if 0.0 in grid:
            grid.remove(0.0)
    elif param == "delta":                    
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, gridpoints_num_alpha_beta).tolist()
        if 0.0 in grid:
            grid.remove(0.0)        
    elif param == "mu_e":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_mue)+prior_loc_mue, grid_width_std*np.sqrt(prior_scale_mue)+prior_loc_mue, gridpoints_num_alpha_beta).tolist()
    elif param == "sigma_e":
        grid = np.linspace(0.00000000001, grid_width_std*np.sqrt(prior_scale_sigmae)+prior_loc_sigmae, gridpoints_num_alpha_beta).tolist()
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

def plot_posterior_elementwise(outdir, param, Y, idx, vector_coordinate, theta_curr, gamma, param_positions_dict, args, 
                            true_param=None, hat_param=None, iteration=None, fig_in=None):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true  = args

    f = get_posterior_for_optimisation_vec(param=param, Y=Y, idx=idx, vector_index_in_param_matrix=idx, vector_coordinate=vector_coordinate, theta=theta_curr, 
                                           gamma=gamma, param_positions_dict=param_positions_dict, args=args)

    if vector_coordinate is None:
        xx_ = np.linspace(-5, 5, 100)
        xx = itertools.product(*[xx_, xx_])  
        xxlist = [ix for ix in xx]        
        yy = np.asarray(list(map(f, xxlist)))          
        # yy = np.asarray([f(x)[0] for x in xx]).flatten()
    else:        
        if param == "gamma":
            xx_ = np.linspace(-5, 5, 1000)
        elif param == "mu_e":
            xx_ = np.linspace(-100, 100, 1000)
        elif param == "sigma_e":
            xx_ = np.linspace(0.00001, max_sigma_e, 1000)
        else:
            xx_ = np.linspace(-5, 5, 1000)        
        yy = np.asarray(list(map(f, xx_)))        
        # yy = np.asarray([f(x)[0] for x in xx_]).flatten()

    # if fig_in is not None:
    #     fig = fig_in
    # else:
    fig = go.Figure()   
    if vector_coordinate is not None:
        fig.add_trace(go.Scatter(
                        x=xx_,
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
        fig.add_trace(go.Contour(
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
                            title='Posterior',
                            titleside='right'
                        )
                    )
                )
        fig.update_layout(                
                xaxis_title='x1',
                yaxis_title='x2',                
        )
    # if fig_in is None:
    if (idx is None and vector_coordinate==0 and true_param is not None) or (isinstance(idx, int) and isinstance(vector_coordinate, int)):
        # scalar param        
        fig.add_trace(go.Scatter(x=xx_, y=[true_param]*len(xx_), mode="markers", marker_symbol="star", marker_color="red"))
    elif isinstance(idx, int) and vector_coordinate is None:
        # surface plot
        fig.add_trace(go.Scatter(x=[true_param[0]], y=[true_param[1]], mode="markers", marker_symbol="star", marker_color="red"))

    if hat_param is not None and ((idx is None and vector_coordinate==0) or (isinstance(idx, int) and isinstance(vector_coordinate, int))):
        # scalar param        
        fig.add_trace(go.Scatter(x=np.linspace(xx_[0], xx_[-1], num=len(hat_param)), y=hat_param, 
                                mode="markers+lines", marker_symbol="square", marker_color="blue", name="step: {}".format(iteration)))
    elif hat_param is not None and (isinstance(idx, int) and vector_coordinate is None):
        # surface plot
        fig.add_trace(go.Scatter(x=hat_param[:, 0], y=hat_param[:, 1], mode="markers+lines", marker_symbol="square", marker_color="blue", name="step: {}".format(iteration)))
    
    fig.update_layout(hovermode='x unified') 
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)    
    savename = "{}/{}_idx_{}_vector_coord_{}.html".format(outdir, param, idx, vector_coordinate)
    fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="Posterior", title="", showgrid=False, showlegend=True, print_png=True, print_html=True, print_pdf=False)

    return fig

def plot_posteriors_during_estimation(Y, iteration, theta_lists, fig_posteriors, fig_posteriors_annealed, gamma, args):

    all_thetas = np.stack(theta_lists)    

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true  = args

    # non-annealed posterior
    for theta_i in range(parameter_space_dim):       
        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=theta_i, d=d)                    
        fig_posteriors[target_param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=target_param, 
                        Y=Y, idx=vector_index_in_param_matrix, vector_coordinate=vector_coordinate, 
                        theta_curr=theta_true.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[theta_i], 
                        hat_param=all_thetas[:, theta_i], iteration=iteration,
                        fig_in=fig_posteriors[target_param])     
    
    for param in parameter_names:
        if param in ["X", "beta"]:
            for i in range(K):        
                if param=="X":
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=i, vector_coordinate=None, 
                        theta_curr=theta_true.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], iteration=iteration,
                        fig_in=fig_posteriors[param])    
                elif param=="beta":
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=i, vector_coordinate=i, 
                        theta_curr=theta_true.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+i], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+i], iteration=iteration, 
                        fig_in=fig_posteriors[param])    
        elif param in ["Z", "Phi", "alpha"]:
            for j in range(J):
                if param=="alpha":
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=j, vector_coordinate=j, 
                        theta_curr=theta_true.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+j], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+j], iteration=iteration, 
                        fig_in=fig_posteriors[param])    
                else:
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=j, vector_coordinate=None, 
                        theta_curr=theta_true.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], iteration=iteration, 
                        fig_in=fig_posteriors[param])    
        

    # annealed posterior
    for theta_i in range(parameter_space_dim):       
        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=theta_i, d=d)     
        keyname = "param_{}_vindexParammatrix_{}_veccord_{}_gamma_{}".format(target_param, vector_index_in_param_matrix, vector_coordinate, gamma)   
        if keyname in fig_posteriors_annealed.keys():
            fig = fig_posteriors_annealed[keyname]             
        else:
            fig = go.Figure()
        fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=target_param, 
                        Y=Y, idx=vector_index_in_param_matrix, vector_coordinate=vector_coordinate, 
                        theta_curr=theta_true.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[theta_i], 
                        hat_param=all_thetas[:, theta_i], iteration=iteration, 
                        fig_in=fig)     
    
    for param in parameter_names:
        keyname = "param_{}_vindexParammatrix_{}_gamma_{}".format(param, vector_index_in_param_matrix, gamma)   
        if keyname in fig_posteriors_annealed.keys():
            fig = fig_posteriors_annealed[keyname]             
        else:
            fig = go.Figure()
        if param in ["X", "beta"]:
            for i in range(K):   
                if param=="X":     
                    fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, 
                                                                                  idx=i, vector_coordinate=None, 
                        theta_curr=theta_true.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], iteration=iteration, 
                        fig_in=fig)    
                elif param=="beta":
                    fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, 
                                                                                idx=i, vector_coordinate=i, 
                        theta_curr=theta_true.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+i], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+i], iteration=iteration, 
                        fig_in=fig)    
        elif param in ["Z", "Phi", "alpha"]:
            for j in range(J):
                if param=="alpha":
                    fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, 
                                                                                idx=j, vector_coordinate=j, 
                        theta_curr=theta_true.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+j], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+j], iteration=iteration, 
                        fig_in=fig) 
                else:
                    fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, idx=j, 
                                                                                  vector_coordinate=None, 
                        theta_curr=theta_true.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], 
                        hat_param=all_thetas[:, param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], iteration=iteration, 
                        fig_in=fig)   
                
    return fig_posteriors, fig_posteriors_annealed



def get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta, gamma, param_positions_dict, args):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true = args

    if param == "X":
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_x_il(x, vector_coordinate, vector_index_in_param_matrix, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, 
                                                                  prior_loc_x[vector_coordinate], prior_scale_x[vector_coordinate, vector_coordinate], gamma)
        else:
            post2optim = lambda x: log_conditional_posterior_x_vec(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x, prior_scale_x, gamma)  
    elif param == "Z":
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_z_jl(x, vector_coordinate, vector_index_in_param_matrix, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z[vector_coordinate], 
                                                                prior_scale_z[vector_coordinate, vector_coordinate], gamma, constant_Z, penalty_weight_Z)
        else:
            post2optim = lambda x: log_conditional_posterior_z_vec(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z, prior_scale_z, gamma, 
                                                                constant_Z, penalty_weight_Z)
    elif param == "Phi":            
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_phi_jl(x, vector_coordinate, vector_index_in_param_matrix, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi[vector_coordinate], 
                                                                    prior_scale_phi[vector_coordinate, vector_coordinate], gamma)
        else:
            post2optim = lambda x: log_conditional_posterior_phi_vec(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi, prior_scale_phi, gamma)
    elif param == "beta":
        if elementwise and isinstance(vector_coordinate, int):
            idx = vector_coordinate
        post2optim = lambda x: log_conditional_posterior_beta_i(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_beta, prior_scale_beta, gamma)
    elif param == "alpha":
        if elementwise and isinstance(vector_coordinate, int):
            idx = vector_coordinate
        post2optim = lambda x: log_conditional_posterior_alpha_j(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_alpha, prior_scale_alpha, gamma)
    elif param == "gamma":
        post2optim = lambda x: log_conditional_posterior_gamma(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=gamma, prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma) 
    elif param == "delta":
        post2optim = lambda x: log_conditional_posterior_delta(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma, prior_loc_delta, prior_scale_delta)    
    elif param == "mu_e":
        post2optim = lambda x: log_conditional_posterior_mu_e(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma, prior_loc_mue, prior_scale_mue)         
    elif param == "sigma_e":
        post2optim = lambda x: log_conditional_posterior_sigma_e(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma, prior_loc_sigmae, prior_scale_sigmae, max_sigma_e) 
        
    return post2optim

def optimise_posterior_elementwise(param, idx, vector_index_in_param_matrix, vector_coordinate, Y, gamma, theta_curr, param_positions_dict, l, args, debug=False):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true  = args
    
    theta_test_in = theta_curr.copy()
    # Negate its result for the negative log posterior likelihood
    f = get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta_test_in, gamma, param_positions_dict, args)
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
                time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)       
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
        elapsedtime = str(timedelta(seconds=time.time()-t0))   
        time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
        hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)    
    
    theta_test_out = theta_curr.copy()
    theta_test_out[idx] = param_estimate
    
    return theta_test_out, elapsedtime

def optimise_posterior_vector(param, idx, Y, gamma, theta_curr, param_positions_dict, l, args, debug=False):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true = args
    
    theta_test_in = theta_curr.copy()
    if evaluate_posterior:
        # Negate its result for the negative log posterior likelihood        
        f = get_posterior_for_optimisation_vec(param, Y, idx, idx, None, theta_test_in, gamma, param_positions_dict, args)
        # get grid depending on param
        grid = get_evaluation_grid(param, None, args)        
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
                time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)                   
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
                time_obj = datetime.strptime(elapsedtime, '%H:%M:%S.%f')
                hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)       
                print("Loop elapsed time: {}".format(elapsedtime))            
                # to refill grid
                grid = get_evaluation_grid(param, None, args)    
            
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
        theta_test_out[param_positions_dict[param][0] + idx*d:param_positions_dict[param][0] + (idx + 1)*d] = param_estimate
    else:
        # scalar updates and coordinate-wise updates of alphas/betas
        theta_test_out[param_positions_dict[param][0] + idx:param_positions_dict[param][0] + (idx + 1)] = param_estimate

    return theta_test_out, elapsedtime


def rank_and_plot_solutions(estimated_thetas, elapsedtime, Y, J, K, d, parameter_names, dst_func, param_positions_dict, DIR_out):

    computed_negloglik = []
    for theta in estimated_thetas:
        nll = negative_loglik(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z=0, constant_Z=0, debug=False)
        computed_negloglik.append(nll)
    # sort in increasing order, i.e. from best to worst solution
    sorted_idx = np.argsort(np.asarray(computed_negloglik))
    sorted_idx_lst = sorted_idx.tolist()    
    for i in sorted_idx_lst:
        theta = estimated_thetas[i]
        nll = computed_negloglik[i]
        params_out = dict()
        params_out["negativeloglik"] = nll
        params_out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")        
        params_out["elapsedtime"] = elapsedtime        
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

        # plot utilities
        pathlib.Path("{}/solution_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)   
        pij_arg = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)
        fig = go.Figure(data=go.Heatmap(
            z=pij_arg,
            x=[str(i) for i in range(pij_arg.shape[1])],
            y=[str(i) for i in range(pij_arg.shape[0])],
            colorscale="sunsetdark",
            showscale=True,   
            colorbar=dict(thickness=10, title='U'),             
        )) 
        fix_plot_layout_and_save(fig, "{}/solution_plots/utilities_solution_index_{}.html".format(DIR_out, sorted_idx_lst.index(i)), 
                                xaxis_title="Leaders", yaxis_title="Followers", title="Utilities with estimated parameters", 
                                showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
    
    # 2D projection of solutions
    raw_symbols = SymbolValidator().values
    theta_matrix = np.array(estimated_thetas)
    theta_matrix = theta_matrix[sorted_idx, :]
    pca = PCA(n_components=2)
    components = pca.fit_transform(theta_matrix)
    fig = go.Figure()
    for i in range(components.shape[0]):
        fig.add_trace(go.Scatter(x=[components[i, 0]], y=[components[i, 1]], marker_symbol=raw_symbols[i]))
    fig.update(layout_yaxis_range = [np.min(components[:,1])-1,np.max(components[:,1])+1])
    fix_plot_layout_and_save(fig, "{}/solution_plots/project_solutions_2D.html".format(DIR_out, sorted_idx_lst.index(i)), 
                                xaxis_title="PC1", yaxis_title="PC2", title="", 
                                showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
    # fig.show()
 

def sample_theta_curr_init(parameter_space_dim, base2exponent, args, samples_list=None, idx_all=None):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true  = args

    if samples_list is None:
        sampler = qmc.Sobol(d=parameter_space_dim, scramble=False)   
        samples_list = list(sampler.random_base2(m=base2exponent))       
        idx_all = np.arange(0, len(samples_list), 1).tolist()
    
    idx = np.random.choice(idx_all, size=1, replace=False)[0]
    theta_curr = np.asarray(samples_list[idx]).reshape((1, parameter_space_dim))
    idx_all.remove(idx)   

    lbounds = np.zeros((parameter_space_dim,))
    ubounds = np.ones((parameter_space_dim,))
    for param in parameter_names:
        if param == "X":
            # assume homogeneous variance
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_x[0, 0])+prior_loc_x[0]
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_x[0, 0])+prior_loc_x[0]
        elif param == "Z":
            # assume homogeneous variance
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_z[0, 0])+prior_loc_z[0]
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_z[0, 0])+prior_loc_z[0]
        elif param == "Phi":
            # assume homogeneous variance
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_phi[0, 0])+prior_loc_phi[0]
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_phi[0, 0])+prior_loc_phi[0]
        elif param == "alpha":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_alpha)+prior_loc_alpha
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_alpha)+prior_loc_alpha
        elif param == "beta":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_beta)+prior_loc_beta
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_beta)+prior_loc_beta
        elif param == "gamma":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_gamma)+prior_loc_gamma
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_gamma)+prior_loc_gamma
        elif param == "delta":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_delta)+prior_loc_delta
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_delta)+prior_loc_delta
        elif param == "mu_e":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_mue)+prior_loc_mue
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_mue)+prior_loc_mue
        elif param == "sigma_e":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_sigmae) #+prior_loc_gamma
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_sigmae) #+prior_loc_gamma
    
    theta_curr = qmc.scale(theta_curr, lbounds, ubounds).reshape((parameter_space_dim,))

    return theta_curr, samples_list, idx_all

def icm_posterior_power_annealing(Y, param_positions_dict, args, temperature_rate=None, temperature_steps=None, 
                                plot_online=True, percentage_parameter_change=1):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, max_sigma_e, theta_true  = args

    gamma = 0.1
    # T0 = get_T0(Y, J, K, d, parameter_names, dst_func, param_positions_dict, args)#3
    l = 0
    n = 0
    i = 0    
    delta_theta = np.inf
    theta_prev = np.zeros((parameter_space_dim,))

    all_gammas = []
    for gidx in range(len(temperature_steps[1:])):
        upperlim = temperature_steps[1+gidx]        
        start = gamma if gidx==0 else all_gammas[-1]        
        all_gammas.extend(np.arange(start, upperlim, temperature_rate[gidx]))            
    N = len(all_gammas)
    print("Annealing schedule: {}".format(N))
    theta_samples_list = None
    idx_all = None
    base2exponent = 10
    theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, args, samples_list=theta_samples_list, idx_all=idx_all)
    
    delta_rate_prev = None
    delta_theta = np.inf
    fig_theta_full = None
    mse_theta_full = []
    fig_xz = None
    mse_x_list = []
    mse_z_list = []
    per_param_ers = dict()
    per_param_heats = dict()
    per_param_boxes = dict()
    fig_posteriors = dict()
    fig_posteriors_annealed = dict()
    for param in parameter_names:
        per_param_ers[param] = []
        fig_posteriors[param] = None
        fig_posteriors_annealed[param] = None
        if param in ["gamma", "delta", "mu_e", "sigma_e"]:
            continue
        else:
            per_param_heats[param] = []
            per_param_boxes[param] = go.Figure()
    # per_param_ers["theta"] = []
    per_param_ers["X_rot_translated_mseOverMatrix"] = []
    per_param_ers["Z_rot_translated_mseOverMatrix"] = []
    per_param_heats["theta"] = []
    per_param_boxes["theta"] = go.Figure()    
    total_iter = 1   
    halving_rate = 0 
    restarts = 0
    max_restarts = 2
    max_halving = 2
    estimated_thetas = []
    while (L is not None and l < L) and not np.all(np.isclose(delta_theta, tol)) and restarts < max_restarts:
        n = 0
        while n < N and not np.all(np.isclose(delta_theta, tol)) and restarts < max_restarts:
            if elementwise:
                i = 0
                while i < parameter_space_dim and not np.all(np.isclose(delta_theta, tol)):
                    target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=i, d=d)                    
                    theta_test, _ = optimise_posterior_elementwise(target_param, i, vector_index_in_param_matrix, vector_coordinate, Y, gamma, theta_curr, 
                                                                param_positions_dict, L, args)
                    theta_curr = theta_test.copy()
                    gamma, delta_rate = update_annealing_temperature(gamma, total_iter, temperature_rate, temperature_steps, all_gammas)
                    delta_theta_se = (theta_curr - theta_prev)**2
                    # if se close to zero, i.e. parameters are stuck, break, else compute relativised se
                    if np.allclose(np.sum(delta_theta_se), tol):                        
                        break
                    delta_theta = delta_theta_se/np.sum(delta_theta_se)
                    if (delta_rate_prev is not None and delta_rate_prev < delta_rate) or (total_iter % 50 == 0 and total_iter > 1):    
                        plot_online = True
                    else:            
                        plot_online = False                        
                    mse_theta_full, fig_theta_full, mse_x_list, mse_z_list, fig_xz, per_param_ers, per_param_heats, per_param_boxes = \
                                    compute_and_plot_mse(theta_true, theta_curr, n, iteration=total_iter, delta_rate=delta_rate_prev, gamma_n=gamma, args=args, 
                                        param_positions_dict=param_positions_dict, plot_online=plot_online, fig_theta_full=fig_theta_full, mse_theta_full=mse_theta_full, 
                                        fig_xz=fig_xz, mse_x_list=mse_x_list, mse_z_list=mse_z_list, per_param_ers=per_param_ers, per_param_heats=per_param_heats, 
                                        per_param_boxes=per_param_boxes)        
                    if total_iter % 50 == 0:
                        # plot posteriors during estimation        
                        fig_posteriors, fig_posteriors_annealed = plot_posteriors_during_estimation(Y, total_iter, per_param_heats["theta"], fig_posteriors, 
                                                                                                    fig_posteriors_annealed, gamma, args)          
                        ipdb.set_trace()
                    delta_rate_prev = delta_rate      
                    theta_prev = theta_curr.copy()                                            
                    total_iter += 1   
                    i += 1
                    n += 1                    
                half, gamma, delta_rate_prev, temperature_rate, all_gammas, N = halve_annealing_rate_upd_schedule(N, gamma, 
                                                                                    delta_rate_prev, delta_theta, temperature_rate, 
                                                                                    temperature_steps, all_gammas, percentage_parameter_change, tol)
                if half:
                    halving_rate += 1
                    if (halving_rate <= max_halving):                   
                        # delta_rate_prev = None
                        restarts += 1                    
                        # keep solution
                        estimated_thetas.append(theta_curr)                                       
                        # random restart
                        theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, args, samples_list=theta_samples_list, idx_all=idx_all)
                        n = N
                        gamma = 0.1
                        delta_theta = np.inf
                        theta_prev = np.zeros((parameter_space_dim,))
                     
            else:
                for param in parameter_names:                     
                    if param in ["X", "beta"]:  
                        param_no = K                                                      
                    elif param in ["Z", "Phi", "alpha"]:
                        param_no = J                                                        
                    else:
                        # scalars
                        param_no = 1                    
                    for idx in range(param_no):
                        theta_test, _ = optimise_posterior_vector(param, idx, Y, gamma, theta_curr, param_positions_dict, L, args)     
                        theta_curr = theta_test.copy()
                        gamma, delta_rate = update_annealing_temperature(gamma, total_iter, temperature_rate, 
                                                                            temperature_steps, all_gammas)
                        if (delta_rate_prev is not None and delta_rate_prev < delta_rate) or (total_iter % 50 == 0 and total_iter > 1):    
                            plot_online = True
                        else:            
                            plot_online = False                        
                        mse_theta_full, fig_theta_full, mse_x_list, mse_z_list, fig_xz, per_param_ers, per_param_heats, per_param_boxes = \
                                        compute_and_plot_mse(theta_true, theta_curr, n, iteration=total_iter, delta_rate=delta_rate_prev, gamma_n=gamma, args=args, param_positions_dict=param_positions_dict,
                                            plot_online=plot_online, fig_theta_full=fig_theta_full, mse_theta_full=mse_theta_full, fig_xz=fig_xz, mse_x_list=mse_x_list, 
                                            mse_z_list=mse_z_list, per_param_ers=per_param_ers, per_param_heats=per_param_heats, per_param_boxes=per_param_boxes)    
                        if total_iter % 50 == 0:
                            # plot posteriors during estimation        
                            fig_posteriors, fig_posteriors_annealed = plot_posteriors_during_estimation(Y, total_iter, per_param_heats["theta"], fig_posteriors, 
                                                                                                        fig_posteriors_annealed, gamma, args)  
                        delta_rate_prev = delta_rate                     
                        delta_theta_se = (theta_curr - theta_prev)**2
                        # delta_theta = delta_theta_se
                        if np.allclose(np.sum(delta_theta_se), tol):                        
                            break
                        delta_theta = delta_theta_se/np.sum(delta_theta_se)
                        theta_prev = theta_curr.copy()                                                                 
                        total_iter += 1                                             
                        if np.all(np.isclose(delta_theta, tol)):                            
                            break
                    if np.allclose(np.sum(delta_theta_se), tol) or np.all(np.isclose(delta_theta, tol)):                            
                        break                
                half, gamma, delta_rate_prev, temperature_rate, all_gammas, N = halve_annealing_rate_upd_schedule(N, gamma, delta_rate_prev, delta_theta, temperature_rate, 
                                                temperature_steps, all_gammas, percentage_parameter_change, tol)
                if half:
                    halving_rate += 1
                    if (halving_rate <= max_halving):                   
                        # delta_rate_prev = None
                        restarts += 1                    
                        # keep solution
                        estimated_thetas.append(theta_curr)
                        # restart 
                        theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, args, samples_list=theta_samples_list, idx_all=idx_all)                                
                        n = N
                        gamma = 0.1
                        delta_theta = np.inf
                        theta_prev = np.zeros((parameter_space_dim,))
        l += 1            
        mse_theta_full, fig_theta_full, mse_x_list, mse_z_list, fig_xz, per_param_ers, per_param_heats, per_param_boxes= \
                        compute_and_plot_mse(theta_true, theta_curr, n, iteration=total_iter, delta_rate=delta_rate_prev, gamma_n=gamma, args=args, param_positions_dict=param_positions_dict,
                            plot_online=True, fig_theta_full=fig_theta_full, mse_theta_full=mse_theta_full, fig_xz=fig_xz, mse_x_list=mse_x_list, 
                            mse_z_list=mse_z_list, per_param_ers=per_param_ers, per_param_heats=per_param_heats, per_param_boxes=per_param_boxes)  
    
    if not np.all(np.isclose(theta_curr, estimated_thetas[-1])) and np.allclose(np.sum(delta_theta_se), tol):
        estimated_thetas.append(theta_curr)       

    return estimated_thetas

def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, temperature_rate=[0, 1], temperature_steps=[1e-3], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_mue=0, prior_scale_mue=1, prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False,
        theta_true=None, percentage_parameter_change=1, max_sigma_e=None):

        for m in range(trialsmin, trialsmax, 1):
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

            args = (DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,                     
                    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                    prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                    prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, 
                    gridpoints_num, diff_iter, disp, max_sigma_e, theta_true)   
            t0 = time.time()
            theta = icm_posterior_power_annealing(Y, param_positions_dict, args,
                                                   temperature_rate=temperature_rate, temperature_steps=temperature_steps, 
                                                   percentage_parameter_change=percentage_parameter_change)
            elapsedtime = str(timedelta(seconds=time.time()-t0))   
            rank_and_plot_solutions(theta, elapsedtime, Y, J, K, d, parameter_names, dst_func, param_positions_dict, DIR_out)


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
        sigma_e_true = 0.5
        total_running_processes = 20   
        elementwise = False
        evaluate_posterior = True
    else:
        parallel = args.parallel
        trialsstr = args.trials
        if "-" in trialsstr:
            trialsparts = trialsstr.split("")
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
    niter = 5
    penalty_weight_Z = 0.0
    constant_Z = 0.0
    retries = 10
    diff_iter = None
    disp = True
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "mu_e", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma" , "mu_e", "sigma_e"]
    d = 2  
    gridpoints_num = 100
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
    prior_loc_mue = 0
    prior_scale_mue = 1    
    # a
    prior_loc_sigmae = 3
    # b
    prior_scale_sigmae = 0.5
    temperature_steps = [0, 1, 2, 5, 10]
    temperature_rate = [1e-3, 1e-2, 1e-1, 1] #[1e-3, 1e-2, 1e-1, 1]    
    min_signal2noise_ratio = 10 # in dB

    max_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(min_signal2noise_ratio/10)))
    print(max_sigma_e)

    tol = 1e-8    
    sigma_e_true = 0.01
    # data_location = "/home/ioannischalkiadakis/ideal/idealpestimation/data_K{}_J{}_sigmae{}_nopareto/".format(K, J, str(sigma_e_true).replace(".", ""))
    data_location = "/home/ioannis/Dropbox (Heriot-Watt University Team)/ideal/idealpestimation/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 50                 
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
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 4
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 3
    theta_true = np.zeros((parameter_space_dim,))
    with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_location), "r") as f:
        for result in f.iter(type=dict, skip_invalid=True):
            for param in parameter_names:
                theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
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
        prior_loc_mue=prior_loc_mue, prior_scale_mue=prior_scale_mue,
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
        gridpoints_num=gridpoints_num, diff_iter=diff_iter, disp=disp, theta_true=theta_true,
        percentage_parameter_change=percentage_parameter_change, max_sigma_e=max_sigma_e)
    
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
    
    # args = (data_location, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, niter, tol,                     
    #         parameter_space_dim, 0, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
    #         prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
    #         prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_mue, prior_scale_mue, prior_loc_sigmae, prior_scale_sigmae, 
    #         gridpoints_num, diff_iter, disp, max_sigma_e, theta_true) 
    # with open("{}/{}/Y.pickle".format(data_location, 0), "rb") as f:
    #         Y = pickle.load(f)
    # Y = Y.astype(np.int8).reshape((K, J), order="F")    
    
    # theta_curr = theta_true.copy()
    # outdir = "{}/posterior_plots/".format(data_location)
    # pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)     
    
    # for param in parameter_names:
    #     outdir = "{}/posterior_plots/{}/".format(data_location, param)
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



    