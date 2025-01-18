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
                                                            create_constraint_functions, p_ij_arg, jax, jnp, log_complement_from_log_cdf, \
                                                                time, datetime, timedelta, log_complement_from_log_cdf_vec
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
    subdataset_name=None, param_positions_dict=None, parallel=False):
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
    bounds, _ = create_constraint_functions(len(initial_guess))    
    if niter is not None:
        result = minimize(likelihood_function, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxiter":niter, "maxfun":250000})
    else:
        result = minimize(likelihood_function, **optimize_kwargs, bounds=bounds, options={"disp":disp, "maxfun":250000})
    
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

def negative_loglik(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    errscale = sigma_e
    errloc = mu_e          
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")    
    _nll = 0
    if debug:
        for i in range(K):
            for j in range(J):
                pij_arg = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
                philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, variance=errscale)
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf

    pij_arg = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
    philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
    log_one_minus_cdf = log_complement_from_log_cdf_vec(philogcdf, pij_arg, mean=errloc, variance=errscale)
    nll = np.sum(Y*philogcdf + (1-Y)*log_one_minus_cdf)

    if debug:
        assert(np.allclose(nll, _nll))

    sum_Z_J_vectors = np.sum(Z, axis=1)    
    return -nll + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)

def negative_loglik_jax(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = jnp.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    Z = jnp.asarray(params_hat["Z"]).reshape((d, J), order="F")   
    Y = jnp.asarray(Y)    
    if "Phi" in params_hat.keys():
        Phi = jnp.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
        delta = jnp.asarray(params_hat["delta"])
    else:
        Phi = jnp.zeros(Z.shape)
        delta = 0
    alpha = jnp.asarray(params_hat["alpha"])
    beta = jnp.asarray(params_hat["beta"])
    # c = params_hat["c"]
    gamma = jnp.asarray(params_hat["gamma"])    
    mu_e = jnp.asarray(params_hat["mu_e"])
    sigma_e = jnp.asarray(params_hat["sigma_e"])
    errscale = sigma_e
    errloc = mu_e 
    _nll = 0
    dst_func = lambda x, y: jnp.sum((x-y)**2)

    pij_argJ = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict, use_jax=True)  
    philogcdfJ = jax.scipy.stats.norm.logcdf(pij_argJ, loc=errloc, scale=errscale)
    # log_one_minus_cdfJ = log_complement_from_log_cdf_vec(philogcdfJ, pij_argJ, mean=errloc, variance=errscale, use_jax=True) - probably numerical errors vs iterative
    log_one_minus_cdfJ = jnp.zeros(philogcdfJ.shape)
    nlltest = 0
    for i in range(K):
        for j in range(J):
            if debug:
                pij_arg = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]    
                philogcdf = jax.scipy.stats.norm.logcdf(pij_arg, loc=errloc, scale=errscale)
            else:
                pij_arg = pij_argJ[i, j]
                philogcdf = philogcdfJ[i, j]
            
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, 
                                                            variance=errscale, use_jax=True)
            if debug:
                log_one_minus_cdfJ.at[i,j].set(log_one_minus_cdf[0])
                nlltest += (1-Y[i, j])*log_one_minus_cdfJ[i,j]
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
            else:
                nlltest += (1-Y[i, j])*log_one_minus_cdf
                
    nll = jnp.sum(Y*philogcdfJ) + nlltest
    if debug:
        assert(jnp.allclose(nll, _nll))
        
    sum_Z_J_vectors = jnp.sum(Z, axis=1)
    return -nll[0] + jnp.asarray(penalty_weight_Z) * jnp.sum((sum_Z_J_vectors-jnp.asarray([constant_Z]*d))**2)    


def estimate_mle(args):

    current_pid = os.getpid()    
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter, \
                                                                            parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel = args

    # load data    
    with open("{}/{}/{}/{}.pickle".format(data_location, m, subdataset_name, subdataset_name), "rb") as f:
        Y = pickle.load(f)
    from_row = int(subdataset_name.split("_")[1])
    to_row = int(subdataset_name.split("_")[2])
    # since each batch has N rows    
    N = Y.shape[0]
    Y = Y.astype(np.int8).reshape((N, J), order="F")         
    
    gridpoints_num = 100
    args = (None, None, None, None, None, J, N, d, dst_func, None, None, 
            parameter_space_dim, None, None, None, None, None, None, None, 
            np.zeros((d,)), np.eye(d), np.zeros((d,)), np.eye(d), None, None, 0, 1, 0, 1, 0, 1, None, None, 0, 1, 0, 1, gridpoints_num, None, None)
    
    X_list = get_evaluation_grid("X", None, args)
    X_list = [xx for xx in X_list]
    Z_list = get_evaluation_grid("Z", None, args)
    Z_list = [xx for xx in Z_list]
    alpha_list = get_evaluation_grid("alpha", None, args)
    beta_list = get_evaluation_grid("beta", None, args)
    gamma_list = get_evaluation_grid("gamma", None, args)
    mu_e_list = get_evaluation_grid("mu_e", None, args)
    sigma_e_list = get_evaluation_grid("sigma_e", None, args)    
    Phi_list = None
    phiidx_all = None
    delta_list = None
    deltaidx_all = None
    # init parameter vector x0 - ensure 2**m > retries
    # m_sobol = 12
    # if 2**m_sobol < retries or 2**m_sobol < J or 2**m_sobol < N:
    #     raise AttributeError("Generate more Sobol points")
    # X_list, Z_list, Phi_list, alpha_list, beta_list, gamma_list, delta_list, mu_e_list, sigma_e_list = initialise_optimisation_vector_sobol(m=m_sobol, J=J, K=N, d=d)
    xidx_all = np.arange(0, len(X_list), 1).tolist()
    zidx_all = np.arange(0, len(Z_list), 1).tolist()
    # if "Phi" in parameter_names:
    #     phiidx_all = np.arange(0, len(Phi_list), 1).tolist()
    alphaidx_all = np.arange(0, len(alpha_list), 1).tolist()    
    betaidx_all = np.arange(0, len(beta_list), 1).tolist()    
    gammaidx_all = np.arange(0, len(gamma_list), 1).tolist()
    # if "delta" in parameter_names:
    #     deltaidx_all = np.arange(0, len(delta_list), 1).tolist()    
    mueidx_all = np.arange(0, len(mu_e_list), 1).tolist()    
    sigmaeidx_all = np.arange(0, len(sigma_e_list), 1).tolist()   

    retry = 0
    t0 = time.time()
    while retry < retries:   
        print("Retry: {}".format(retry))

        xidx = np.random.choice(xidx_all, size=N, replace=False)
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
        mueidx = np.random.choice(mueidx_all, size=1, replace=False)
        mu_e = mu_e_list[mueidx[0]]
        sigmaeidx = np.random.choice(sigmaeidx_all, size=1, replace=False)
        sigma_e = sigma_e_list[sigmaeidx[0]]
        
        x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
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
                                                output_dir=DIR_out, subdataset_name=subdataset_name, param_positions_dict=param_positions_dict, parallel=parallel)          
        
        if result.success:
            break
        else:              
            for xr in xidx:
                xidx_all.remove(xr)                
            for zr in zidx:
                zidx_all.remove(zr)                
            if "Phi" in parameter_names:
                for phir in phiidx:
                    phiidx_all.remove(phir)
            alphaidx_all.remove(alphaidx[0])
            betaidx_all.remove(betaidx[0])
            gammaidx_all.remove(gammaidx[0])
            if "delta" in parameter_names:
                deltaidx_all.remove(deltaidx[0])
            mueidx_all.remove(mueidx[0])
            sigmaeidx_all.remove(sigmaeidx[0])
            retry += 1

    if result.success:
        params_hat = optimisation_dict2params(mle, param_positions_dict, J, N, d, parameter_names)
        variance_hat = optimisation_dict2paramvectors(result["variance_diag"], param_positions_dict, J, K, d, parameter_names) 

        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = current_pid
        grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        grid_and_optim_outcome["elapsedtime"] = str(timedelta(seconds=time.time()-t0))   
        time_obj = datetime.strptime(grid_and_optim_outcome["elapsedtime"], '%H:%M:%S.%f')
        hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
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
        grid_and_optim_outcome["mu_e"] = params_hat["mu_e"].tolist()[0]
        grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"].tolist()[0]
        grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
        if "Phi" in params_hat.keys():
            grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
        grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
        grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
        if "delta" in params_hat.keys():
            grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]
        grid_and_optim_outcome["variance_mu_e"] = variance_hat["mu_e"]
        grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
        
        grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
        grid_and_optim_outcome["mle_estimation_status"] = result.success
        grid_and_optim_outcome["variance_estimation_status"] = result["variance_status"]
    else:
        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = current_pid
        grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        grid_and_optim_outcome["elapsedtime"] = str(timedelta(seconds=time.time()-t0))   
        time_obj = datetime.strptime(grid_and_optim_outcome["elapsedtime"], '%H:%M:%S.%f')
        hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
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
        grid_and_optim_outcome["mu_e"] = None
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
        
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter, \
                                                                            parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel = args

        # load data    
        with open("{}/{}/{}/{}.pickle".format(data_location, m, subdataset_name, subdataset_name), "rb") as f:
            Y = pickle.load(f)
        from_row = int(subdataset_name.split("_")[1])
        to_row = int(subdataset_name.split("_")[2])
        # since each batch has N rows
        N = Y.shape[0]
        Y = Y.astype(np.int8).reshape((N, J), order="F")         
                
        # init parameter vector x0 - ensure 2**m > retries
        m_sobol = 15
        if 2**m_sobol < retries or 2**m_sobol < J*d*retries or 2**m_sobol < N*d*retries:
            raise AttributeError("Generate more Sobol points")
        X_list, Z_list, Phi_list, alpha_list, beta_list, gamma_list, delta_list, mu_e_list, sigma_e_list = initialise_optimisation_vector_sobol(m=m_sobol, J=J, K=N, d=d)
        xidx_all = np.arange(0, len(X_list), 1).tolist()
        zidx_all = np.arange(0, len(Z_list), 1).tolist()
        if "Phi" in parameter_names:
            phiidx_all = np.arange(0, len(Phi_list), 1).tolist()
        alphaidx_all = np.arange(0, len(alpha_list), 1).tolist()    
        betaidx_all = np.arange(0, len(beta_list), 1).tolist()    
        gammaidx_all = np.arange(0, len(gamma_list), 1).tolist()
        if "delta" in parameter_names:
            deltaidx_all = np.arange(0, len(delta_list), 1).tolist()    
        mueidx_all = np.arange(0, len(mu_e_list), 1).tolist()    
        sigmaeidx_all = np.arange(0, len(sigma_e_list), 1).tolist()   

        retry = 0
        t0 = time.time()
        while retry < retries:   
            print("Retry: {}".format(retry))

            xidx = np.random.choice(xidx_all, size=N, replace=False)
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
            alphaidx = np.random.choice(alphaidx_all, size=1, replace=False)
            alpha = np.asarray(alpha_list[alphaidx[0]])
            betaidx = np.random.choice(betaidx_all, size=1, replace=False)
            beta = np.asarray(beta_list[betaidx[0]])
            gammaidx = np.random.choice(gammaidx_all, size=1, replace=False)
            gamma = gamma_list[gammaidx[0]]
            if "delta" in parameter_names:
                deltaidx = np.random.choice(deltaidx_all, size=1, replace=False)
                delta = delta_list[deltaidx[0]]
            else:
                delta = None        
            mueidx = np.random.choice(mueidx_all, size=1, replace=False)
            mu_e = mu_e_list[mueidx[0]]
            sigmaeidx = np.random.choice(sigmaeidx_all, size=1, replace=False)
            sigma_e = sigma_e_list[sigmaeidx[0]]
            
            x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
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
                                                    output_dir=DIR_out, subdataset_name=subdataset_name, param_positions_dict=param_positions_dict, parallel=parallel)          
            if result.success:
                break
            else:                     
                for xr in xidx:
                    xidx_all.remove(xr)                
                for zr in zidx:
                    zidx_all.remove(zr)                
                if "Phi" in parameter_names:
                    for phir in phiidx:
                        phiidx_all.remove(phir)
                alphaidx_all.remove(alphaidx[0])
                betaidx_all.remove(betaidx[0])
                gammaidx_all.remove(gammaidx[0])
                if "delta" in parameter_names:
                    deltaidx_all.remove(deltaidx[0])
                mueidx_all.remove(mueidx[0])
                sigmaeidx_all.remove(sigmaeidx[0])    
                retry += 1
        
        if result.success:
            params_hat = optimisation_dict2params(mle, param_positions_dict, J, N, d, parameter_names)
            variance_hat = optimisation_dict2paramvectors(result["variance_diag"], param_positions_dict, J, K, d, parameter_names) 

            grid_and_optim_outcome = dict()
            grid_and_optim_outcome["PID"] = current_pid
            grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            grid_and_optim_outcome["elapsedtime"] = str(timedelta(seconds=time.time()-t0))   
            time_obj = datetime.strptime(grid_and_optim_outcome["elapsedtime"], '%H:%M:%S.%f')
            hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
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
            grid_and_optim_outcome["mu_e"] = params_hat["mu_e"].tolist()[0]
            grid_and_optim_outcome["sigma_e"] = params_hat["sigma_e"].tolist()[0]
            grid_and_optim_outcome["variance_Z"] = variance_hat["Z"]
            if "Phi" in params_hat.keys():
                grid_and_optim_outcome["variance_Phi"] = variance_hat["Phi"]
            grid_and_optim_outcome["variance_alpha"] = variance_hat["alpha"]    
            grid_and_optim_outcome["variance_gamma"] = variance_hat["gamma"]
            if "delta" in params_hat.keys():
                grid_and_optim_outcome["variance_delta"] = variance_hat["delta"]
            grid_and_optim_outcome["variance_mu_e"] = variance_hat["mu_e"]
            grid_and_optim_outcome["variance_sigma_e"] = variance_hat["sigma_e"]
            
            grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
            grid_and_optim_outcome["mle_estimation_status"] = result.success
            grid_and_optim_outcome["variance_estimation_status"] = result["variance_status"]
        else:
            grid_and_optim_outcome = dict()
            grid_and_optim_outcome["PID"] = current_pid
            grid_and_optim_outcome["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            grid_and_optim_outcome["elapsedtime"] = str(timedelta(seconds=time.time()-t0))   
            time_obj = datetime.strptime(grid_and_optim_outcome["elapsedtime"], '%H:%M:%S.%f')
            hours = (time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000)
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
            grid_and_optim_outcome["mu_e"] = None
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
        self.append_to_json_file(grid_and_optim_outcome, output_file=out_file)

         


def main(J=2, K=2, d=1, N=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        niter=None, parameter_space_dim=None, trials=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10):

    if parallel:
        manager = ProcessManagerSynthetic(total_running_processes)        
    DIR_top = data_location      
    try:    
        if parallel:  
            manager.create_results_dict(optim_target="all")    
            while True:
                for m in range(trials):
                    path = pathlib.Path("{}/{}".format(data_location, m))  
                    subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]                    
                    for dataset_index in range(len(subdatasets_names)):                    
                        subdataset_name = subdatasets_names[dataset_index]                        
                        DIR_out = "{}/{}/{}/estimation/".format(DIR_top, m, subdataset_name)
                        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True) 
                        args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, 
                                parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel)    
                                            
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
            for m in range(trials):
                path = pathlib.Path("{}/{}".format(data_location, m))  
                subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]               
                for dataset_index in range(len(subdatasets_names)):               
                    subdataset_name = subdatasets_names[dataset_index]                    
                    DIR_out = "{}/{}/{}/estimation/".format(DIR_top, m, subdataset_name)
                    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True) 
                    args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, 
                            parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel)
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

    seed_value = 9125
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    parallel = True
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
    retries = 20
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "mu_e", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "mu_e", "sigma_e"]
    M = 1
    K = 10000
    J = 100
    sigma_e_true = 0.5
    d = 2    
    data_location = "./idealpestimation/data_K{}_J{}_sigmae{}_nopareto/".format(K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 10              
    # with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_location), mode="r") as f:
    #     for result in f.iter(type=dict, skip_invalid=True):                              
    #         J = result["J"]
    #         K = result["K"]
    #         d = result["d"]
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 4
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 3
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    # for distributing per N rows
    N = math.ceil(parameter_space_dim/J)
    print("Observed data points per data split: {}".format(N*J))        
    main(J=J, K=K, d=d, N=N, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, niter=niter, parameter_space_dim=parameter_space_dim, trials=M, 
        penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=10)
    
    # params_out_jsonl = dict()
    # for m in range(M):
    #     data_location = "./idealpestimation/data_K{}_J{}_sigmae{}_nopareto/{}/".format(K, J, str(sigma_e_true).replace(".", ""), m)
    #     # data_location = "/home/ioannischalkiadakis/ideal/idealpestimation/data_K{}_J{}_sigmae{}_nopareto/{}/".format(K, J, str(sigma_e_true).replace(".", ""), m)
    #     params_out = combine_estimate_variance_rule(data_location, J, K, d, parameter_names)    
    #     for param in parameter_names:
    #         if param == "X":                
    #             params_out_jsonl[param] = params_out[param].reshape((d*K,), order="F").tolist()                        
    #         elif param == "Z":
    #             params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()                         
    #         elif param == "Phi":            
    #             params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()                         
    #         elif param in ["beta", "alpha"]:
    #             params_out_jsonl[param] = params_out[param].tolist()            
    #         else:
    #             params_out_jsonl[param] = params_out[param]
    #     out_file = "{}/params_out_global_theta_hat.jsonl".format(data_location)
    #     with open(out_file, 'a') as f:         
    #         writer = jsonlines.Writer(f)
    #         writer.write(params_out_jsonl)