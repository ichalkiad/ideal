import os 
import sys
import ipdb
import pathlib
import jsonlines
import pickle
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from jax import hessian
import math
import random
from idealpestimation.src.parallel_manager import ProcessManager, \
                                                    jsonlines
from idealpestimation.src.utils import params2optimisation_dict, \
                                            optimisation_dict2params, \
                                                initialise_optimisation_vector_sobol, \
                                                    visualise_hessian, fix_plot_layout_and_save, \
                                                        get_hessian_diag_jax, get_jacobian, \
                                                            combine_estimate_variance_rule, optimisation_dict2paramvectors,\
                                                            create_constraint_functions, jax, jnp, \
                                                                time, datetime, timedelta, parse_input_arguments, \
                                                                    negative_loglik, negative_loglik_jax, collect_mle_results, \
                                                                        collect_mle_results_batchsize_analysis, sample_theta_curr_init

from idealpestimation.src.icm_annealing_posteriorpower import get_evaluation_grid

def variance_estimation(estimation_result, loglikelihood=None, loglikelihood_per_data_point=None, 
                        data=None, full_hessian=True, diag_hessian_only=True, nloglik_jax=None, parallel=False):

       
    params = estimation_result.x        
    try:                        
        if params.shape[0] == 1:
            # scalar parameters
            scores = []
            if len(data.shape)==1:
                data = data.reshape((data.shape[0], 1))            
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    loglik_xp = lambda params: loglikelihood_per_data_point(i, j, params, data)
                    score_function_xp = lambda params: get_jacobian(params, likelihood_function=loglik_xp)        
                    scores.append(score_function_xp(params))            
            scores = np.asarray(scores)
            sigma_sq_inv = np.mean(np.diag(scores @ scores.T))                
            # Compute variance as the inverse of the Fisher Information
            # Add small regularization to prevent singularity
            variance = np.linalg.inv(sigma_sq_inv + 1e-8 * np.eye(len(params)))
            if parallel:
                hess = np.linalg.inv(estimation_result.hess_inv * np.ones(len(params)) + 1e-8 * np.eye(len(params)))
            else:
                # due to JAX incompatibility with Python's multiprocessing
                hess = hessian(loglikelihood)(params)  
            if np.isnan(variance):
                # error status
                return sigma_sq_inv, hess, variance, False
            else:
                # success status
                return sigma_sq_inv, hess, variance, True
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
                    ipdb.set_trace()         
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
    subdataset_name=None, param_positions_dict=None, parallel=False, min_sigma_e=1e-6, args=None):
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
    bounds, _ = create_constraint_functions(len(initial_guess), min_sigma_e, args=None)  #######################    
    if niter is not None:
        result = minimize(likelihood_function, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxiter":niter, "maxfun":1000000}) #2000000
    else:
        result = minimize(likelihood_function, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxfun":1000000}) #2000000
    
    mle = result.x          

    if result.success:
        try:        
            variance_noninv, hessian_mat, variance_diag, variance_status = variance_estimation(estimation_result=result, loglikelihood=likelihood_function,
                                        data=data, full_hessian=full_hessian, diag_hessian_only=diag_hessian_only,
                                        loglikelihood_per_data_point=loglikelihood_per_data_point, nloglik_jax=negloglik_jax, parallel=parallel)
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

    current_pid = os.getpid()    
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, min_sigma_e, \
        prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi,\
        prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, prior_loc_gamma,\
        prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng = args

    # load data    
    batchsize = 304
    with open("{}/{}/{}/{}/{}.pickle".format(data_location, m, batchsize, subdataset_name, subdataset_name), "rb") as f: ############
        Y = pickle.load(f)


    from_row = int(subdataset_name.split("_")[1])
    to_row = int(subdataset_name.split("_")[2])
    # since each batch has N rows    
    N = Y.shape[0]
    Y = Y.astype(np.int8).reshape((N, J), order="F")         
    
    theta_samples_list = None
    base2exponent = 10
    idx_all = None
    parameter_space_dim_theta = (N+J)*d + J + N + 2
    param_positions_dict_theta = dict()            
    k = 0
    for param in parameter_names:
        if param == "X":
            param_positions_dict[param] = (k, k + N*d)                       
            k += N*d    
        elif param in ["Z"]:
            param_positions_dict[param] = (k, k + J*d)                                
            k += J*d
        elif param in ["Phi"]:            
            param_positions_dict[param] = (k, k + J*d)                                
            k += J*d
        elif param == "beta":
            param_positions_dict[param] = (k, k + N)                                   
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
    theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim_theta, base2exponent, param_positions_dict_theta, 
                                                                args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)

    """gridpoints_num = 200
    args = (None, None, None, None, None, J, N, d, dst_func, None, None, 
            parameter_space_dim, None, None, None, None, None, None, None, 
            prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi,
            prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
            gridpoints_num, None, None, min_sigma_e, None)
     

    X_list, _ = get_evaluation_grid("X", None, args)
    X_list = [xx for xx in X_list]
    Z_list, _ = get_evaluation_grid("Z", None, args)
    Z_list = [xx for xx in Z_list]
    alpha_list, _ = get_evaluation_grid("alpha", None, args)
    beta_list, _ = get_evaluation_grid("beta", None, args)
    gamma_list, _ = get_evaluation_grid("gamma", None, args)    
    sigma_e_list, _ = get_evaluation_grid("sigma_e", None, args)    
    Phi_list = None
    phiidx_all = None
    delta_list = None
    deltaidx_all = None
    # init parameter vector x0 - ensure 2**m > retries
    # m_sobol = 12
    # if 2**m_sobol < retries*J*N:
    #     raise AttributeError("Generate more Sobol points")
    # X_list, Z_list, Phi_list, alpha_list, beta_list, gamma_list, delta_list, mu_e_list, sigma_e_list = initialise_optimisation_vector_sobol(m=m_sobol, J=J, K=N, d=d, min_sigma_e=min_sigma_e)
    xidx_all = np.arange(0, len(X_list), 1).tolist()
    zidx_all = np.arange(0, len(Z_list), 1).tolist()
    # if "Phi" in parameter_names:
    #     phiidx_all = np.arange(0, len(Phi_list), 1).tolist()
    alphaidx_all = np.arange(0, len(alpha_list), 1).tolist()    
    betaidx_all = np.arange(0, len(beta_list), 1).tolist()    
    gammaidx_all = np.arange(0, len(gamma_list), 1).tolist()
    # if "delta" in parameter_names:
    #     deltaidx_all = np.arange(0, len(delta_list), 1).tolist()        
    sigmaeidx_all = np.arange(0, len(sigma_e_list), 1).tolist()   """

    retry = 0
    t0 = time.time()
    while retry < retries:   
        print("Retry: {}".format(retry))

        """xidx = np.random.choice(xidx_all, size=N, replace=False)
        Xrem = [X_list[ii] for ii in xidx]
        X = np.asarray(Xrem).reshape((d, N), order="F")
        zidx = np.random.choice(zidx_all, size=J, replace=False)
        Zrem = [Z_list[ii] for ii in zidx]
        Z = np.asarray(Zrem).reshape((d, J), order="F")
        if "Phi" in parameter_names:
            phiidx = np.random.choice(phiidx_all, size=J, replace=False)
            Phirem = [Phi_list[ii] for ii in phiidx]
            Phi = np.asarray(Phirem).reshape((d, J), order="F")
        else:
            Phi = None
        # alphaidx = np.random.choice(alphaidx_all, size=1, replace=False)
        # alpha = np.asarray(alpha_list[alphaidx[0]])
        alphaidx = np.random.choice(alphaidx_all, size=J, replace=False)
        alpha = np.asarray(alpha_list)[np.asarray(alphaidx)]
        # betaidx = np.random.choice(betaidx_all, size=1, replace=False)
        # beta = np.asarray(beta_list[betaidx[0]])
        betaidx = np.random.choice(betaidx_all, size=N, replace=False)
        beta = np.asarray(beta_list)[np.asarray(betaidx)]     
        gammaidx = np.random.choice(gammaidx_all, size=1, replace=False)
        gamma = gamma_list[gammaidx[0]]
        if "delta" in parameter_names:
            deltaidx = np.random.choice(deltaidx_all, size=1, replace=False)
            delta = delta_list[deltaidx[0]]
        else:
            delta = None                
        sigmaeidx = np.random.choice(sigmaeidx_all, size=1, replace=False)
        sigma_e = sigma_e_list[sigmaeidx[0]]"""
        
        # x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, sigma_e)
        x0 = theta_curr.copy()
        # print(x0)
        nloglik = lambda x: negative_loglik(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z)
        if parallel:
            nloglik_jax = None
        else:    
            nloglik_jax = lambda x: negative_loglik_jax(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z)
        mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                                variance_method='jacobian', disp=True, 
                                                optimization_method=optimisation_method, 
                                                data=Y, full_hessian=False, diag_hessian_only=True, plot_hessian=False,   
                                                loglikelihood_per_data_point=None, niter=niter, negloglik_jax=nloglik_jax, 
                                                output_dir=DIR_out, subdataset_name=subdataset_name, 
                                                param_positions_dict=param_positions_dict, parallel=parallel, min_sigma_e=min_sigma_e, args=args)          
        
        if result.success and result["variance_status"]:
            break
        else:    
            theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                            args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)          
            # for xr in xidx:
            #     xidx_all.remove(xr)                
            # for zr in zidx:
            #     zidx_all.remove(zr)                
            # if "Phi" in parameter_names:
            #     for phir in phiidx:
            #         phiidx_all.remove(phir)
            # alphaidx_all.remove(alphaidx[0])
            # betaidx_all.remove(betaidx[0])
            # gammaidx_all.remove(gammaidx[0])
            # if "delta" in parameter_names:
            #     deltaidx_all.remove(deltaidx[0])            
            # sigmaeidx_all.remove(sigmaeidx[0])
            retry += 1

    if result.success and result["variance_status"]:
        params_hat = optimisation_dict2params(mle, param_positions_dict, J, N, d, parameter_names)
        variance_hat = optimisation_dict2paramvectors(result["variance_diag"], param_positions_dict, J, K, d, parameter_names) 

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
        grid_and_optim_outcome["retry"] = retry
        grid_and_optim_outcome["parameter names"] = parameter_names
        grid_and_optim_outcome["local theta"] = [mle.tolist()]
        grid_and_optim_outcome["X"] = params_hat["X"].reshape((d*N,), order="F").tolist()  
        grid_and_optim_outcome["Z"] = params_hat["Z"].reshape((d*J,), order="F").tolist()     
        if "Phi" in params_hat.keys():
            grid_and_optim_outcome["Phi"] = params_hat["Phi"].reshape((d*J,), order="F").tolist() 
        grid_and_optim_outcome["alpha"] = params_hat["alpha"].tolist() 
        grid_and_optim_outcome["beta"] = params_hat["beta"].tolist() 
        grid_and_optim_outcome["gamma"] = params_hat["gamma"].tolist()[0]
        if "delta" in params_hat.keys():
            grid_and_optim_outcome["delta"] = params_hat["delta"].tolist()[0]        
        grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"].tolist()[0]
        grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
        if "Phi" in params_hat.keys():
            grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
        grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
        grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
        if "delta" in params_hat.keys():
            grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]        
        grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
        
        grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
        grid_and_optim_outcome["mle_estimation_status"] = result.success
        grid_and_optim_outcome["variance_estimation_status"] = result["variance_status"]
    else:
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
        
        grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
        grid_and_optim_outcome["mle_estimation_status"] = result.success
        grid_and_optim_outcome["variance_estimation_status"] = None


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
        
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method,\
            parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z,\
                constant_Z, retries, parallel, min_sigma_e, prior_loc_x, prior_scale_x, prior_loc_z,\
                    prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha,\
                        prior_scale_alpha, prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta,\
                            prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng = args
        
        # load data  
        batchsize = 304
        with open("{}/{}/{}/{}/{}.pickle".format(data_location, m, batchsize, subdataset_name, subdataset_name), "rb") as f: ############
            Y = pickle.load(f)

        from_row = int(subdataset_name.split("_")[1])
        to_row = int(subdataset_name.split("_")[2])
        # since each batch has N rows
        N = Y.shape[0]
        Y = Y.astype(np.int8).reshape((N, J), order="F")         
        
        theta_samples_list = None 
        base2exponent = 10
        idx_all = None
        parameter_space_dim_theta = (N+J)*d + J + N + 2
        param_positions_dict_theta = dict()            
        k = 0
        for param in parameter_names:
            if param == "X":
                param_positions_dict[param] = (k, k + N*d)                       
                k += N*d    
            elif param in ["Z"]:
                param_positions_dict[param] = (k, k + J*d)                                
                k += J*d
            elif param in ["Phi"]:            
                param_positions_dict[param] = (k, k + J*d)                                
                k += J*d
            elif param == "beta":
                param_positions_dict[param] = (k, k + N)                                   
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
        theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim_theta, base2exponent, param_positions_dict_theta, 
                                                                            args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)  
        """gridpoints_num = 200
        # args = (None, None, None, None, None, J, N, d, dst_func, None, None, 
        #     parameter_space_dim, None, None, None, None, None, None, None, 
        #     np.zeros((d,)), np.eye(d), np.zeros((d,)), np.eye(d), None, None, 0, 1, 0, 1, 0, 1, None, None, 0, 1, gridpoints_num, None, None, min_sigma_e, None)

        args = (None, None, None, None, None, J, N, d, dst_func, None, None, 
            parameter_space_dim, None, None, None, None, None, None, None, 
            prior_loc_x, prior_scale_x, prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi,
            prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
            gridpoints_num, None, None, min_sigma_e, None)
        
        X_list, _ = get_evaluation_grid("X", None, args)
        X_list = [xx for xx in X_list]
        Z_list, _ = get_evaluation_grid("Z", None, args)
        Z_list = [xx for xx in Z_list]
        alpha_list, _ = get_evaluation_grid("alpha", None, args)
        beta_list, _ = get_evaluation_grid("beta", None, args)
        gamma_list, _ = get_evaluation_grid("gamma", None, args)        
        sigma_e_list, _ = get_evaluation_grid("sigma_e", None, args)    
        Phi_list = None
        phiidx_all = None
        delta_list = None
        deltaidx_all = None
        # init parameter vector x0 - ensure 2**m > retries
        # m_sobol = 12
        # if 2**m_sobol < retries*J*N:
        #     raise AttributeError("Generate more Sobol points")
        # X_list, Z_list, Phi_list, alpha_list, beta_list, gamma_list, delta_list, mu_e_list, sigma_e_list = initialise_optimisation_vector_sobol(m=m_sobol, J=J, K=N, d=d, min_sigma_e=min_sigma_e)
        xidx_all = np.arange(0, len(X_list), 1).tolist()
        zidx_all = np.arange(0, len(Z_list), 1).tolist()
        # if "Phi" in parameter_names:
        #     phiidx_all = np.arange(0, len(Phi_list), 1).tolist()
        alphaidx_all = np.arange(0, len(alpha_list), 1).tolist()    
        betaidx_all = np.arange(0, len(beta_list), 1).tolist()    
        gammaidx_all = np.arange(0, len(gamma_list), 1).tolist()
        # if "delta" in parameter_names:
        #     deltaidx_all = np.arange(0, len(delta_list), 1).tolist()            
        sigmaeidx_all = np.arange(0, len(sigma_e_list), 1).tolist()     """

        retry = 0
        t0 = time.time()
        while retry < retries:   
            print("Retry: {}".format(retry))

            """xidx = np.random.choice(xidx_all, size=N, replace=False)
            Xrem = [X_list[ii] for ii in xidx]
            X = np.asarray(Xrem).reshape((d, N), order="F")
            zidx = np.random.choice(zidx_all, size=J, replace=False)
            Zrem = [Z_list[ii] for ii in zidx]
            Z = np.asarray(Zrem).reshape((d, J), order="F")
            if "Phi" in parameter_names:
                phiidx = np.random.choice(phiidx_all, size=J, replace=False)
                Phirem = [Phi_list[ii] for ii in phiidx]
                Phi = np.asarray(Phirem).reshape((d, J), order="F")
            else:
                Phi = None
            # alphaidx = np.random.choice(alphaidx_all, size=1, replace=False)
            # alpha = np.asarray(alpha_list[alphaidx[0]])
            alphaidx = np.random.choice(alphaidx_all, size=J, replace=False)
            alpha = np.asarray(alpha_list)[np.asarray(alphaidx)]
            # betaidx = np.random.choice(betaidx_all, size=1, replace=False)
            # beta = np.asarray(beta_list[betaidx[0]])
            betaidx = np.random.choice(betaidx_all, size=N, replace=False)
            beta = np.asarray(beta_list)[np.asarray(betaidx)]     
            gammaidx = np.random.choice(gammaidx_all, size=1, replace=False)
            gamma = gamma_list[gammaidx[0]]
            if "delta" in parameter_names:
                deltaidx = np.random.choice(deltaidx_all, size=1, replace=False)
                delta = delta_list[deltaidx[0]]
            else:
                delta = None                    
            sigmaeidx = np.random.choice(sigmaeidx_all, size=1, replace=False)
            sigma_e = sigma_e_list[sigmaeidx[0]]
            
            x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, sigma_e)"""
            
            x0 = theta_curr.copy()
            # print(x0)
            nloglik = lambda x: negative_loglik(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z)
            if parallel:
                nloglik_jax = None
            else:                
                nloglik_jax = lambda x: negative_loglik_jax(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z)
            mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                                    variance_method='jacobian', disp=True, 
                                                    optimization_method=optimisation_method, 
                                                    data=Y, full_hessian=True, diag_hessian_only=False, plot_hessian=True,   
                                                    loglikelihood_per_data_point=None, niter=niter, negloglik_jax=nloglik_jax, 
                                                    output_dir=DIR_out, subdataset_name=subdataset_name, 
                                                    param_positions_dict=param_positions_dict, parallel=parallel, min_sigma_e=min_sigma_e, args=args)          
            if result.success and result["variance_status"]:
                break
            else:
                theta_curr, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict,
                                                                            args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)                      
                # for xr in xidx:
                #     xidx_all.remove(xr)                
                # for zr in zidx:
                #     zidx_all.remove(zr)                
                # if "Phi" in parameter_names:
                #     for phir in phiidx:
                #         phiidx_all.remove(phir)
                # alphaidx_all.remove(alphaidx[0])
                # betaidx_all.remove(betaidx[0])
                # gammaidx_all.remove(gammaidx[0])
                # if "delta" in parameter_names:
                #     deltaidx_all.remove(deltaidx[0])                
                # sigmaeidx_all.remove(sigmaeidx[0])    
                retry += 1
        
        if result.success and result["variance_status"]:
            params_hat = optimisation_dict2params(mle, param_positions_dict, J, N, d, parameter_names)
            variance_hat = optimisation_dict2paramvectors(result["variance_diag"], param_positions_dict, J, K, d, parameter_names) 

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
            grid_and_optim_outcome["retry"] = retry
            grid_and_optim_outcome["parameter names"] = parameter_names
            grid_and_optim_outcome["local theta"] = [mle.tolist()]
            grid_and_optim_outcome["X"] = params_hat["X"].reshape((d*N,), order="F").tolist()  
            grid_and_optim_outcome["Z"] = params_hat["Z"].reshape((d*J,), order="F").tolist()     
            if "Phi" in params_hat.keys():
                grid_and_optim_outcome["Phi"] = params_hat["Phi"].reshape((d*J,), order="F").tolist() 
            grid_and_optim_outcome["alpha"] = params_hat["alpha"].tolist() 
            grid_and_optim_outcome["beta"] = params_hat["beta"].tolist() 
            grid_and_optim_outcome["gamma"] = params_hat["gamma"].tolist()[0]
            if "delta" in params_hat.keys():
                grid_and_optim_outcome["delta"] = params_hat["delta"].tolist()[0]            
            grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"].tolist()[0]
            grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
            if "Phi" in params_hat.keys():
                grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
            grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
            grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
            if "delta" in params_hat.keys():
                grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]            
            grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
            
            grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
            grid_and_optim_outcome["mle_estimation_status"] = result.success
            grid_and_optim_outcome["variance_estimation_status"] = result["variance_status"]
        else:
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
            grid_and_optim_outcome["variance_sigma_e"] = None
            
            grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
            grid_and_optim_outcome["mle_estimation_status"] = result.success
            grid_and_optim_outcome["variance_estimation_status"] = None
                
        out_file = "{}/estimationresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)
        self.append_to_json_file(grid_and_optim_outcome, output_file=out_file)

         


def main(J=2, K=2, d=1, N=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        niter=None, parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, 
        constant_Z=0.0, retries=10, min_sigma_e=None, prior_loc_x=None, prior_scale_x=None, 
        prior_loc_z=None, prior_scale_z=None, prior_loc_phi=None, prior_scale_phi=None,
        prior_loc_beta=None, prior_scale_beta=None, prior_loc_alpha=None, prior_scale_alpha=None,
        prior_loc_gamma=None, prior_scale_gamma=None, prior_loc_delta=None, prior_scale_delta=None, 
        prior_loc_sigmae=None, prior_scale_sigmae=None, param_positions_dict=None, rng=None):

    if parallel:
        manager = ProcessManagerSynthetic(total_running_processes)        
    DIR_top = data_location      
    try:    
        if parallel:  
            manager.create_results_dict(optim_target="all")    
            while True:
                for m in range(trialsmin, trialsmax, 1):

                    batchsize = 304
                    path = pathlib.Path("{}/{}/{}/".format(data_location, m, batchsize))  #########

                    subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]                    
                    for dataset_index in range(len(subdatasets_names)):                    
                        subdataset_name = subdatasets_names[dataset_index]  

                        DIR_out = "{}/{}/{}/{}/estimation/".format(DIR_top, m, batchsize, subdataset_name) #########

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
                                    prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng)    
                                                
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
                batchsize = 304
                path = pathlib.Path("{}/{}/{}/".format(data_location, m, batchsize))  #########

                subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]               
                for dataset_index in range(len(subdatasets_names)):               
                    subdataset_name = subdatasets_names[dataset_index]            

                    DIR_out = "{}/{}/{}/{}/estimation/".format(DIR_top, m, batchsize, subdataset_name) #########

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
                                prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng)
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


if __name__ == "__main__":

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
    if not parallel:
        try:
            jax.default_device = jax.devices("gpu")[0]
        except:
            print("Using cpu")
            jax.default_device = jax.devices("cpu")[0]
        jax.config.update("jax_traceback_filtering", "off")

    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = None
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
    data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e_true).replace(".", ""))
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
    main(J=J, K=K, d=d, N=N, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, niter=niter, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, 
        trialsmax=M, penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=retries, min_sigma_e=min_sigma_e,
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, 
        prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, prior_loc_phi=prior_loc_phi, 
        prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, prior_loc_gamma=prior_loc_gamma, 
        prior_scale_gamma=prior_scale_gamma, prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta, 
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae, param_positions_dict=param_positions_dict, rng=rng)


    # data_topdir = data_location
    # collect_mle_results(data_topdir, M, K, J, sigma_e_true, d, parameter_names, param_positions_dict)
    # collect_mle_results_batchsize_analysis(data_topdir, [64, 128, 192, 256, 320], M, K, J, sigma_e_true, d, parameter_names, param_positions_dict)    
    
