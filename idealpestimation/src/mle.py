import os 

os.environ["OMP_NUM_THREADS"] = "500"
os.environ["MKL_NUM_THREADS"] = "500"
os.environ["OPENBLAS_NUM_THREADS"] = "500"
os.environ["NUMBA_NUM_THREADS"] = "500"
os.environ["JAX_NUM_THREADS"] = "500"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=500"

import sys
import ipdb
import pathlib
import pickle
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from jax import hessian
import math
import random
from idealpestimation.src.parallel_manager import ProcessManager, jsonlines
from idealpestimation.src.utils import params2optimisation_dict, \
                                            optimisation_dict2params, \
                                                initialise_optimisation_vector_sobol, \
                                                    visualise_hessian, fix_plot_layout_and_save, \
                                                        get_hessian_diag_jax, get_jacobian, \
                                                            combine_estimate_variance_rule, optimisation_dict2paramvectors,\
                                                            create_constraint_functions, jnp, \
                                                                time, datetime, timedelta, parse_input_arguments, \
                                                                    negative_loglik_coordwise, negative_loglik_coordwise_jax, collect_mle_results, \
                                                                        collect_mle_results_batchsize_analysis, sample_theta_curr_init,\
                                                                        get_parameter_name_and_vector_coordinate, check_convergence, rank_and_return_best_theta,\
                                                                        negative_loglik_coordwise_parallel, plot_loglik_runtimes, parse_timedelta_string,\
                                                                        print_threadpool_info, clean_up_data_matrix
from idealpestimation.src.efficiency_monitor import Monitor


from idealpestimation.src.icm_annealing_posteriorpower import get_evaluation_grid

def optimise_negativeloglik_elementwise(param, idx, vector_index_in_param_matrix, vector_coordinate, Y, theta_curr, 
                                        param_positions_dict, l, args, debug=False, theta_samples_list=None, idx_all=None, base2exponent=10):
    
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, L, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, min_sigma_e, \
        prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi,\
        prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, prior_loc_gamma,\
        prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, _, rng, batchsize = args
    
    niter_minimize = None
    theta_test_in = theta_curr.copy()
    f = lambda x: negative_loglik_coordwise(x, idx, theta_test_in, Y, J, N, d, 
                                            parameter_names, dst_func, param_positions_dict, 
                                            penalty_weight_Z, constant_Z)
    if parallel:
        f_jax = None
    else:    
        f_jax = lambda x: negative_loglik_coordwise_jax(x, idx, theta_test_in, Y, J, N, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z)
    
    retry = 0
    t0 = time.time()
    while retry < retries:   
        # print("Retry: {}".format(retry))
        mle, result = maximum_likelihood_estimator(f, initial_guess=np.asarray([theta_test_in[idx]]), 
                                        variance_method='jacobian', disp=False, 
                                        optimization_method=optimisation_method, 
                                        data=Y, full_hessian=False, diag_hessian_only=False, 
                                        plot_hessian=False, loglikelihood_per_data_point=None, 
                                        niter=niter_minimize, negloglik_jax=f_jax, output_dir=DIR_out, 
                                        subdataset_name=subdataset_name, param_positions_dict=param_positions_dict, 
                                        parallel=parallel, min_sigma_e=min_sigma_e, args=args, target_param=param, target_idx=idx)         
        param_estimate = mle    
        if result.success and result["variance_status"]:
            break
        else:    
            theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                            args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)
            retry += 1
        
    elapsedtime = str(timedelta(seconds=time.time()-t0))       
    theta_test_out = theta_curr.copy()
    theta_test_out[idx] = param_estimate[0]
    
    return theta_test_out, elapsedtime, result, retry


def variance_estimation(estimation_result, loglikelihood=None, loglikelihood_per_data_point=None, 
                        data=None, full_hessian=True, diag_hessian_only=True, nloglik_jax=None, parallel=False):

       
    params = estimation_result.x        
    try:                        
        if params.shape[0] == 1:
            variance = estimation_result.hess_inv.todense()[0, 0] 
            if np.isnan(variance):
                # error status
                return None, None, variance, False
            else:
                # success status
                return None, None, variance, True            
        else:
            if full_hessian:
                if parallel:                    
                    hess = np.linalg.inv(estimation_result.hess_inv * np.ones(len(params)) + 1e-8 * np.eye(len(params)))                    
                    variance = estimation_result.hess_inv * np.ones((len(params), len(params)))
                else:
                    # Use Hessian approximation to compute Fisher Information as the sample Hessian                                                  
                    params_jax = jnp.asarray(params)                                   
                    hess = np.asarray(hessian(nloglik_jax)(params_jax))                    
                    # Add small regularization to prevent singularity
                    variance = -np.linalg.inv(hess + 1e-8 * np.eye(len(params)))            
                if np.any(np.isnan(variance)):                                
                    raise ArithmeticError
                else:
                    return variance, hess, np.diag(variance), True
            if diag_hessian_only:   
                if parallel:
                    variance = np.diag(estimation_result.hess_inv * np.eye(len(params)))
                else:    
                    params_jax = jnp.asarray(params)                  
                    hess_jax = get_hessian_diag_jax(nloglik_jax, params_jax)                                     
                    variance = -1/np.asarray(hess_jax)            
                if np.any(np.isnan(variance)):    
                    if hasattr(estimation_result, 'hess_inv') and estimation_result["hess_inv"] is not None:
                        if not isinstance(estimation_result["hess_inv"], np.ndarray):
                            variance = -np.diag(estimation_result["hess_inv"].todense())
                        else:
                            variance = -np.diag(estimation_result["hess_inv"])
                    else:                            
                        return None, None, variance, False
                else:
                    return None, None, variance, True
    except Exception as e:
        raise ArithmeticError

def maximum_likelihood_estimator(
    likelihood_function, 
    initial_guess=None, 
    variance_method='jacobian', 
    optimization_method='L-BFGS-B',
    data=None, full_hessian=True, diag_hessian_only=True,
    loglikelihood_per_data_point=None, disp=False, niter=None, 
    jac=None, output_dir="/tmp/", plot_hessian=False, negloglik_jax=None, 
    subdataset_name=None, param_positions_dict=None, parallel=False, min_sigma_e=1e-6, 
    args=None, target_param=None, target_idx=None):
    """
    Estimate the maximum likelihood parameter and its variance.

    Parameters:
    -----------
    likelihood_function : callable
        A function that takes a parameter and returns the negative log-likelihood.
        Should be structured such that minimizing it maximizes the likelihood and it operates on the complete set of data

    Returns:
    --------
    tuple
        A tuple containing:
        - Maximum likelihood estimate (MLE) of the parameter
        - Variance of the MLE 
    """
    optimize_kwargs = {
        'method': optimization_method,
        'x0': initial_guess,                
        'jac': '3-point'#,
        # 'hess': '2-point'
    }                   
    # Perform maximum likelihood estimation
    mle = None
    result = None        
    bounds, _ = create_constraint_functions(len(initial_guess), min_sigma_e, args=args, target_param=target_param, target_idx=target_idx)
    if niter is not None:
        result = minimize(likelihood_function, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxiter":niter, "maxfun":100000}) #2000000
    else:
        result = minimize(likelihood_function, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxfun":100000}) #2000000
    
    mle = result.x          

    if result.success:
        try:        
            variance_noninv, hessian_mat, variance_diag, variance_status = variance_estimation(estimation_result=result, 
                                                                                    loglikelihood=likelihood_function,
                                                                                    data=data, full_hessian=full_hessian, 
                                                                                    diag_hessian_only=diag_hessian_only,
                                                                                    loglikelihood_per_data_point=loglikelihood_per_data_point, 
                                                                                    nloglik_jax=negloglik_jax, parallel=parallel)
            result["variance_method"] = variance_method
            result["variance_diag"] = variance_diag
            result["variance_status"] = variance_status
            if full_hessian:
                result["full_hessian"] = hessian_mat
                if plot_hessian:
                    fig = visualise_hessian(hessian_mat)
                    fix_plot_layout_and_save(fig, "{}/hessian_{}_{}.html".format(output_dir, subdataset_name.replace(".pickle", ""), datetime.now().strftime("%Y-%m-%d")),
                                            xaxis_title="", yaxis_title="", title="Full Hessian matrix estimate", showgrid=False, showlegend=False,
                                            print_png=True, print_html=True, print_pdf=False)
                    
        except ArithmeticError as e:            
            # Fallback to zero variance if computation fails
            print(f"Variance estimation failed: {e}")
            variance_diag = np.zeros((mle.shape[0],))
            result["variance_method"] = "{}".format(variance_method)
            result["variance_diag"] = variance_diag
            result["variance_status"] = False
    else:
        result["variance_method"] = None
        result["variance_diag"] = np.zeros((mle.shape[0],))
        result["variance_status"] = False

        
    return mle, result

def estimate_mle(args):

    tol = 1e-6
    current_pid = os.getpid()    
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, L, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, min_sigma_e, \
        prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi,\
        prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, prior_loc_gamma,\
        prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        param_positions_dict, rng, batchsize, theta_true = args
    
    print(subdataset_name)
    # load data    
    with open("{}/{}/{}/{}/{}.pickle".format(data_location, m, batchsize, subdataset_name, subdataset_name), "rb") as f:
        Y = pickle.load(f)


    Y, K, J, theta_true, param_positions_dict, parameter_space_dim = clean_up_data_matrix(Y, K, J, d, theta_true, parameter_names, param_positions_dict)


    from_row = int(subdataset_name.split("_")[1])
    to_row = int(subdataset_name.split("_")[2])
    # since each batch has N rows    
    N = Y.shape[0]
    Y = Y.astype(np.int8).reshape((N, J), order="F")         
    
    idx_all = None
    theta_samples_list = None
    base2exponent = 15    
    parameter_space_dim_theta = (N+J)*d + J + N + 2
    param_positions_dict_theta = dict()            
    k = 0
    for param in parameter_names:
        if param == "X":
            param_positions_dict_theta[param] = (k, k + N*d)                       
            k += N*d    
        elif param in ["Z"]:
            param_positions_dict_theta[param] = (k, k + J*d)                                
            k += J*d
        elif param in ["Phi"]:            
            param_positions_dict_theta[param] = (k, k + J*d)                                
            k += J*d
        elif param == "beta":
            param_positions_dict_theta[param] = (k, k + N)                                   
            k += N    
        elif param == "alpha":
            param_positions_dict_theta[param] = (k, k + J)                                       
            k += J    
        elif param == "gamma":
            param_positions_dict_theta[param] = (k, k + 1)                                
            k += 1
        elif param == "delta":
            param_positions_dict_theta[param] = (k, k + 1)                                
            k += 1
        elif param == "sigma_e":
            param_positions_dict_theta[param] = (k, k + 1)                                
            k += 1    
    theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim_theta, base2exponent, 
                                                                    param_positions_dict_theta, 
                                                                    args, samples_list=theta_samples_list, 
                                                                    idx_all=idx_all, rng=rng)
    args_theta = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, 
                J, N, d, N, dst_func, L,
                parameter_space_dim_theta, m, penalty_weight_Z, constant_Z, retries, parallel, min_sigma_e,
                prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi,
                prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, prior_loc_gamma,
                prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                param_positions_dict_theta, rng, batchsize)   

    max_full_restarts = 3
    full_restarts = 0
    estimated_thetas = []
    t0 = time.time()
    while full_restarts < max_full_restarts:
        l = 0
        estimation_success = True
        var_estimation_success = True
        theta_prev = np.zeros((parameter_space_dim_theta,))
        variance_local_vec = np.zeros((parameter_space_dim_theta,))
        converged = False
        while ((L is not None and l < L)) and (not converged):
            i = 0 
            while i < parameter_space_dim_theta:                                            
                target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict_theta, i=i, d=d)                    
                theta_test, elapsedtime, result, retry = optimise_negativeloglik_elementwise(target_param, i, vector_index_in_param_matrix, vector_coordinate, 
                                                                    Y, theta_curr.copy(), param_positions_dict_theta, L, args_theta, debug=False,
                                                                    theta_samples_list=theta_samples_list, idx_all=idx_all, base2exponent=base2exponent)                   
                theta_curr = theta_test.copy()   
                estimation_success *= result.success 
                var_estimation_success *= result["variance_status"] 
                if target_param in ["alpha", "beta"]:
                    vector_index = vector_coordinate
                else:
                    vector_index = vector_index_in_param_matrix
                if vector_coordinate is not None:
                    if target_param in ["X", "Z", "Phi"]:
                        testidx = vector_index*d + vector_coordinate
                    else:
                        testidx = vector_coordinate
                else:
                    testidx = vector_index
                coord_converged, _, _ = check_convergence(True, theta_curr, theta_prev, param_positions_dict_theta, i, 
                                                        parameter_space_dim=parameter_space_dim_theta, testparam=target_param, 
                                                        testidx=testidx, p=1, tol=tol)   
                if (not coord_converged) or (coord_converged and result["variance_diag"] != 1):
                    # if converged, estimate has not moved, hence Hessian Inv is 1
                    variance_local_vec[i] = result["variance_diag"]
                i += 1  
            converged, delta_theta, random_restart = check_convergence(True, theta_curr, theta_prev, param_positions_dict_theta, i, 
                                                                            parameter_space_dim=parameter_space_dim_theta, testparam=None, 
                                                                            testidx=vector_coordinate, p=1, tol=tol)     
            theta_prev = theta_curr.copy() 
            l += 1
            print("Total full scans of Theta: {}".format(l))
            print("Convergence: {}".format(converged)) 
            print("Random restart: {}".format(random_restart))  
            
            
        if (not converged) or (full_restarts < max_full_restarts):           
            # print(theta_curr)                                                              
            # random restart, completely from scratch
            theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim_theta, base2exponent, param_positions_dict_theta,
                                                                        args_theta, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)                     
            full_restarts += 1
            mle = theta_curr.copy()
            # save current estimate
            if estimation_success and result["variance_status"]:
                params_hat = optimisation_dict2paramvectors(mle, param_positions_dict_theta, J, N, d, parameter_names)
                variance_hat = optimisation_dict2paramvectors(variance_local_vec, param_positions_dict_theta, J, N, d, parameter_names) 
                grid_and_optim_outcome = dict()
                grid_and_optim_outcome["PID"] = current_pid
                grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")                
                elapsedtime = str(timedelta(seconds=time.time()-t0))
                time_obj, hours, minutes, seconds, microsec = parse_timedelta_string(elapsedtime)
                # total_seconds = int(elapsedtime.total_seconds())
                # hours = total_seconds // 3600
                # minutes = (total_seconds % 3600) // 60
                # seconds = total_seconds % 60            
                grid_and_optim_outcome["elapsedtime"] = f"{hours}:{minutes:02d}:{seconds:02d}"       
                grid_and_optim_outcome["elapsedtime_hours"] = hours
                grid_and_optim_outcome["retry"] = retry
                grid_and_optim_outcome["parameter names"] = parameter_names
                grid_and_optim_outcome["local theta"] = [mle.tolist()]
                grid_and_optim_outcome["X"] = params_hat["X"]
                grid_and_optim_outcome["Z"] = params_hat["Z"]
                if "Phi" in params_hat.keys():
                    grid_and_optim_outcome["Phi"] = params_hat["Phi"] 
                grid_and_optim_outcome["alpha"] = params_hat["alpha"]
                grid_and_optim_outcome["beta"] = params_hat["beta"]
                grid_and_optim_outcome["gamma"] = params_hat["gamma"][0]
                if "delta" in params_hat.keys():
                    grid_and_optim_outcome["delta"] = params_hat["delta"][0]        
                grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"][0]
                grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
                if "Phi" in params_hat.keys():
                    grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
                grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
                grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
                if "delta" in params_hat.keys():
                    grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]        
                grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
                
                grid_and_optim_outcome["param_positions_dict"] = param_positions_dict_theta
                grid_and_optim_outcome["mle_estimation_status"] = estimation_success
                grid_and_optim_outcome["variance_estimation_status"] = var_estimation_success
                estimated_thetas.append((mle, variance_local_vec, current_pid, grid_and_optim_outcome["timestamp"], 
                                grid_and_optim_outcome["elapsedtime"], hours, retry, estimation_success, var_estimation_success))
            else:
                grid_and_optim_outcome = dict()
                grid_and_optim_outcome["PID"] = current_pid
                grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                elapsedtime = str(timedelta(seconds=time.time()-t0))            
                time_obj, hours, minutes, seconds, microsec = parse_timedelta_string(elapsedtime)
                # total_seconds = int(elapsedtime.total_seconds())
                # hours = total_seconds // 3600
                # minutes = (total_seconds % 3600) // 60
                # seconds = total_seconds % 60                 
                grid_and_optim_outcome["elapsedtime"] = f"{hours}:{minutes:02d}:{seconds:02d}"       
                grid_and_optim_outcome["elapsedtime_hours"] = hours
                grid_and_optim_outcome["retry"] = retry
                grid_and_optim_outcome["parameter names"] = parameter_names
                grid_and_optim_outcome["local theta"] = None
                grid_and_optim_outcome["X"] = None
                grid_and_optim_outcome["Z"] = None
                if "Phi" in parameter_names:
                    grid_and_optim_outcome["Phi"] = None
                grid_and_optim_outcome["alpha"] = None
                grid_and_optim_outcome["beta"] = None
                grid_and_optim_outcome["gamma"] = None
                if "delta" in parameter_names:
                    grid_and_optim_outcome["delta"] = None        
                grid_and_optim_outcome["sigma_e"] = None

                grid_and_optim_outcome["variance_Z"] = None
                if "Phi" in parameter_names:
                    grid_and_optim_outcome["variance_Phi"] = None
                grid_and_optim_outcome["variance_alpha"] = None
                grid_and_optim_outcome["variance_gamma"] = None
                if "delta" in parameter_names:
                    grid_and_optim_outcome["variance_delta"] = None
                grid_and_optim_outcome["variance_mu_e"] = None
                grid_and_optim_outcome["variance_sigma_e"] = None
                
                grid_and_optim_outcome["param_positions_dict"] = param_positions_dict_theta
                grid_and_optim_outcome["mle_estimation_status"] = result.success
                grid_and_optim_outcome["variance_estimation_status"] = None
            out_file = "{}/estimationallresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)
            with open(out_file, 'a') as f:         
                writer = jsonlines.Writer(f)
                writer.write(grid_and_optim_outcome)

    mle = theta_curr.copy()
    if estimation_success and result["variance_status"]:
        params_hat = optimisation_dict2paramvectors(mle, param_positions_dict_theta, J, N, d, parameter_names)
        variance_hat = optimisation_dict2paramvectors(variance_local_vec, param_positions_dict_theta, J, N, d, parameter_names) 
        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = current_pid
        grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsedtime = str(timedelta(seconds=time.time()-t0))            
        time_obj, hours, minutes, seconds, microsec = parse_timedelta_string(elapsedtime)
        # total_seconds = int(elapsedtime.total_seconds())
        # hours = total_seconds // 3600
        # minutes = (total_seconds % 3600) // 60
        # seconds = total_seconds % 60               
        grid_and_optim_outcome["elapsedtime"] = f"{hours}:{minutes:02d}:{seconds:02d}"       
        grid_and_optim_outcome["elapsedtime_hours"] = hours
        grid_and_optim_outcome["retry"] = retry
        grid_and_optim_outcome["parameter names"] = parameter_names
        grid_and_optim_outcome["local theta"] = [mle.tolist()]
        grid_and_optim_outcome["X"] = params_hat["X"]
        grid_and_optim_outcome["Z"] = params_hat["Z"]   
        if "Phi" in params_hat.keys():
            grid_and_optim_outcome["Phi"] = params_hat["Phi"]
        grid_and_optim_outcome["alpha"] = params_hat["alpha"]
        grid_and_optim_outcome["beta"] = params_hat["beta"]
        grid_and_optim_outcome["gamma"] = params_hat["gamma"][0]
        if "delta" in params_hat.keys():
            grid_and_optim_outcome["delta"] = params_hat["delta"][0]        
        grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"][0]
        grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
        if "Phi" in params_hat.keys():
            grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
        grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
        grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
        if "delta" in params_hat.keys():
            grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]        
        grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
        
        grid_and_optim_outcome["param_positions_dict"] = param_positions_dict_theta
        grid_and_optim_outcome["mle_estimation_status"] = estimation_success
        grid_and_optim_outcome["variance_estimation_status"] = var_estimation_success
        estimated_thetas.append((mle, variance_local_vec, current_pid, grid_and_optim_outcome["timestamp"], 
                                grid_and_optim_outcome["elapsedtime"], hours, retry, estimation_success, var_estimation_success))
    else:
        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = current_pid
        grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsedtime = str(timedelta(seconds=time.time()-t0))            
        time_obj, hours, minutes, seconds, microsec = parse_timedelta_string(elapsedtime)
        # total_seconds = int(elapsedtime.total_seconds())
        # hours = total_seconds // 3600
        # minutes = (total_seconds % 3600) // 60
        # seconds = total_seconds % 60                   
        grid_and_optim_outcome["elapsedtime"] = f"{hours}:{minutes:02d}:{seconds:02d}"       
        grid_and_optim_outcome["elapsedtime_hours"] = hours
        grid_and_optim_outcome["retry"] = retry
        grid_and_optim_outcome["parameter names"] = parameter_names
        grid_and_optim_outcome["local theta"] = None
        grid_and_optim_outcome["X"] = None
        grid_and_optim_outcome["Z"] = None
        if "Phi" in parameter_names:
            grid_and_optim_outcome["Phi"] = None
        grid_and_optim_outcome["alpha"] = None
        grid_and_optim_outcome["beta"] = None
        grid_and_optim_outcome["gamma"] = None
        if "delta" in parameter_names:
            grid_and_optim_outcome["delta"] = None        
        grid_and_optim_outcome["sigma_e"] = None

        grid_and_optim_outcome["variance_Z"] = None
        if "Phi" in parameter_names:
            grid_and_optim_outcome["variance_Phi"] = None
        grid_and_optim_outcome["variance_alpha"] = None
        grid_and_optim_outcome["variance_gamma"] = None
        if "delta" in parameter_names:
            grid_and_optim_outcome["variance_delta"] = None
        grid_and_optim_outcome["variance_mu_e"] = None
        grid_and_optim_outcome["variance_sigma_e"] = None
        
        grid_and_optim_outcome["param_positions_dict"] = param_positions_dict_theta
        grid_and_optim_outcome["mle_estimation_status"] = result.success
        grid_and_optim_outcome["variance_estimation_status"] = None


    out_file = "{}/estimationallresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)
    with open(out_file, 'a') as f:         
        writer = jsonlines.Writer(f)
        writer.write(grid_and_optim_outcome)
    
    # keep best estimate for weighted combining
    best_theta, best_theta_var, current_pid, timestamp, eltime, hours, retry, success, varstatus =\
                                            rank_and_return_best_theta(estimated_thetas, Y, J, N, d, parameter_names, 
                                                            dst_func, param_positions_dict_theta, DIR_out, args_theta)
    params_hat = optimisation_dict2paramvectors(best_theta, param_positions_dict_theta, J, N, d, parameter_names)
    variance_hat = optimisation_dict2paramvectors(best_theta_var, param_positions_dict_theta, J, N, d, parameter_names) 
    grid_and_optim_outcome = dict()
    grid_and_optim_outcome["PID"] = current_pid
    grid_and_optim_outcome["timestamp"] = timestamp
    elapsedtime = str(timedelta(seconds=time.time()-t0))
    time_obj, hours, minutes, seconds, microsec = parse_timedelta_string(elapsedtime)
    # total_seconds = int(elapsedtime.total_seconds())
    # hours = total_seconds // 3600
    # minutes = (total_seconds % 3600) // 60
    # seconds = total_seconds % 60                      
    grid_and_optim_outcome["elapsedtime_besttheta"] = eltime
    grid_and_optim_outcome["elapsedtime_full"] = f"{hours}:{minutes:02d}:{seconds:02d}"       
    grid_and_optim_outcome["elapsedtime_full_hours"] = hours
    grid_and_optim_outcome["retry"] = retry
    grid_and_optim_outcome["parameter names"] = parameter_names
    grid_and_optim_outcome["local theta"] = [best_theta.tolist()]
    grid_and_optim_outcome["X"] = params_hat["X"]
    grid_and_optim_outcome["Z"] = params_hat["Z"]   
    if "Phi" in params_hat.keys():
        grid_and_optim_outcome["Phi"] = params_hat["Phi"]
    grid_and_optim_outcome["alpha"] = params_hat["alpha"]
    grid_and_optim_outcome["beta"] = params_hat["beta"]
    grid_and_optim_outcome["gamma"] = params_hat["gamma"][0]
    if "delta" in params_hat.keys():
        grid_and_optim_outcome["delta"] = params_hat["delta"][0]        
    grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"][0]
    grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
    if "Phi" in params_hat.keys():
        grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
    grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
    grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
    if "delta" in params_hat.keys():
        grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]        
    grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
    
    grid_and_optim_outcome["param_positions_dict"] = param_positions_dict_theta
    grid_and_optim_outcome["mle_estimation_status"] = success
    grid_and_optim_outcome["variance_estimation_status"] = varstatus
    out_file = "{}/estimationresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)
    with open(out_file, 'a') as f:         
        writer = jsonlines.Writer(f)
        writer.write(grid_and_optim_outcome)

    

            
class ProcessManagerSynthetic(ProcessManager):
    def __init__(self, max_processes):
        super().__init__(max_processes)
    
    def worker_process(self, args):

        current_pid = os.getpid()
        with self.execution_counter.get_lock():
            self.execution_counter.value += 1
            self.shared_dict[current_pid] = self.execution_counter.value        
        estimate_mle(args)         

def main(J=2, K=2, d=1, N=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        niter=None, parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, 
        constant_Z=0.0, retries=10, min_sigma_e=None, prior_loc_x=None, prior_scale_x=None, 
        prior_loc_z=None, prior_scale_z=None, prior_loc_phi=None, prior_scale_phi=None,
        prior_loc_beta=None, prior_scale_beta=None, prior_loc_alpha=None, prior_scale_alpha=None,
        prior_loc_gamma=None, prior_scale_gamma=None, prior_loc_delta=None, prior_scale_delta=None, 
        prior_loc_sigmae=None, prior_scale_sigmae=None, param_positions_dict=None, rng=None, batchsize=None):

    if parallel:
        manager = ProcessManagerSynthetic(total_running_processes)       
    else:
        manager = None 
    DIR_top = data_location      

    try:    
        if parallel:  
            manager.create_results_dict(optim_target="all")    
            while True:
                for m in range(trialsmin, trialsmax, 1):

                    theta_true = np.zeros((parameter_space_dim,))    
                    with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
                        for result in f.iter(type=dict, skip_invalid=True):
                            for param in parameter_names:
                                theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 

                    path = pathlib.Path("{}/{}/{}/".format(data_location, m, batchsize)) 

                    subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]                    
                    for dataset_index in range(len(subdatasets_names)):                    
                        subdataset_name = subdatasets_names[dataset_index]  

                        DIR_out = "{}/{}/{}/{}/estimation/".format(DIR_top, m, batchsize, subdataset_name) 

                        from_row = int(subdataset_name.split("_")[1])
                        to_row = int(subdataset_name.split("_")[2])
                        # if all have completed, following never exits the while loop
                        if pathlib.Path(DIR_out).is_dir():
                            estimationfiles = [file.name for file in pathlib.Path(DIR_out).iterdir() if file.is_file() and ".png" in file.name]
                        if pathlib.Path(DIR_out).is_dir() and \
                            pathlib.Path("{}/estimationresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)).exists() and\
                                len(estimationfiles) > 0:      
                                continue
                        else:
                            print(DIR_out)
                            pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True) 
                            args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, 
                                    parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z, 
                                    constant_Z, retries, parallel, min_sigma_e, prior_loc_x, prior_scale_x, prior_loc_z, 
                                    prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, 
                                    prior_scale_alpha, prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, 
                                    prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng, batchsize, theta_true)    
                                                
                            #####  parallelisation with Parallel Manager #####
                            manager.cleanup_finished_processes()
                            current_count = manager.current_process_count()                                
                            print(f"Currently running processes: {current_count}")
                            manager.print_shared_dict() 
                            while current_count == total_running_processes:
                                manager.cleanup_finished_processes()
                                current_count = manager.current_process_count()                                                                                                                                                   
                            if current_count < total_running_processes:
                                manager.spawn_process(args=(args,))                                                  
                            # Wait before next iteration
                            time.sleep(1)  
                            ################################################## 
                if manager.all_processes_complete.is_set():
                    break       
        else:
            for m in range(trialsmin, trialsmax, 1):
                
                theta_true = np.zeros((parameter_space_dim,))    
                with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
                    for result in f.iter(type=dict, skip_invalid=True):
                        for param in parameter_names:
                            theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 

                path = pathlib.Path("{}/{}/{}/".format(data_location, m, batchsize))
                subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]               
                for dataset_index in range(len(subdatasets_names)):               
                    subdataset_name = subdatasets_names[dataset_index]
                    DIR_out = "{}/{}/{}/{}/estimation/".format(DIR_top, m, batchsize, subdataset_name)
                    from_row = int(subdataset_name.split("_")[1])
                    to_row = int(subdataset_name.split("_")[2])
                    if pathlib.Path(DIR_out).is_dir():
                        estimationfiles = [file.name for file in pathlib.Path(DIR_out).iterdir() if file.is_file() and ".png" in file.name]
                    if pathlib.Path(DIR_out).is_dir() and \
                            pathlib.Path("{}/estimationresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)).exists() and\
                                len(estimationfiles) > 0:      
                                continue
                    else:
                        print(DIR_out)
                        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True) 
                        args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, 
                                parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, min_sigma_e,
                                prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,
                                prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng, batchsize, theta_true)
                        estimate_mle(args)                  

    except KeyboardInterrupt:
        # On Ctrl-C stop all processes
        print("\nShutting down gracefully...")
        if parallel:
            manager.cleanup_all_processes()
            manager.print_shared_dict()  # Final print of shared dictionary
            with jsonlines.open("{}/spawnedprocesses.jsonl".format(DIR_out), "w") as f:
                f.write(manager.shared_dict)
        sys.exit(0)
    if parallel:
        with jsonlines.open("{}/spawnedprocesses.jsonl".format(DIR_out), "w") as f:
            f.write(dict(manager.shared_dict))      
    
    return manager, args


if __name__ == "__main__":

    # plot_loglik_runtimes("/mnt/hdd2/ioannischalkiadakis/idealdata/timings_parallellikelihood.jsonl", "/mnt/hdd2/ioannischalkiadakis/idealdata/")
    # import sys
    # sys.exit(0)
    # python idealpestimation/src/mle.py  --trials 1 --K 30 --J 10 --sigmae 05 --parallel --total_running_processes 5

    seed_value = 9125
    random.seed(seed_value)
    np.random.seed(seed_value)
    rng = np.random.default_rng()

    args = parse_input_arguments()
    
    if args.trials is None or args.K is None or args.J is None or args.sigmae is None:
        parallel = False
        Mmin = 0
        M = 1
        K = 10000
        J = 1000
        sigma_e_true = 0.01
        total_running_processes = 20   
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

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes)
    # if not parallel:
    #     try:
    #         jax.default_device = jax.devices("gpu")[0]
    #     except:
    #         print("Using cpu")
    #         jax.default_device = jax.devices("cpu")[0]
    #     jax.config.update("jax_traceback_filtering", "off")

    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = 200
    penalty_weight_Z = 0.0
    constant_Z = 0.0
    retries = 30
    batchsize = 13
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
    data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata_plotstest/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e_true).replace(".", ""))
    # data_location = "/mnt/hdd2/ioannischalkiadakis/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))           
    # with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_location), mode="r") as f:
    #     for result in f.iter(type=dict, skip_invalid=True):                              
    #         J = result["J"]
    #         K = result["K"]
    #         d = result["d"]
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
    monitor = Monitor(interval=0.01)
    monitor.start()   
    t_start = time.time()  
    par_manager, main_run_args = main(J=J, K=K, d=d, N=N, total_running_processes=total_running_processes, 
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
                                                rng=rng, batchsize=batchsize)

    data_topdir = data_location
    if par_manager is not None:
        while not par_manager.all_processes_complete.is_set():
            print("Waiting for parallel processing to complete all batches...")
            continue
    t_end = time.time()
    monitor.stop()
    wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
        avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
    efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                avg_threads, max_threads, avg_processes, max_processes)
    elapsedtime = str(timedelta(seconds=t_end - t_start))   
    collect_mle_results(efficiency_measures, data_topdir, M, K, J, sigma_e_true, d, parameter_names, param_positions_dict, batchsize, main_run_args, seedint=seed_value)
    # collect_mle_results_batchsize_analysis(data_topdir, [64, 128, 192, 256, 320], M, K, J, sigma_e_true, d, parameter_names, param_positions_dict, seedint=seed_value)    
    