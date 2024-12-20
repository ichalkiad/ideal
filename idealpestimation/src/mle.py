import os 
import sys
import time 
import ipdb
import pathlib
import jsonlines
import pickle
import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize
from jax import hessian
import jax
import jax.numpy as jnp
import math
from idealpestimation.src.parallel_manager import ProcessManager, \
                                                    jsonlines
from idealpestimation.src.utils import params2optimisation_dict, \
                                            optimisation_dict2params, \
                                                initialise_optimisation_vector_sobol, \
                                                    visualise_hessian, fix_plot_layout_and_save, \
                                                        get_hessian_diag_jax, get_jacobian, \
                                                            combine_estimate_variance_rule, get_global_theta

def variance_estimation(estimation_result, loglikelihood=None, loglikelihood_per_data_point=None, 
                        data=None, full_hessian=True, diag_hessian_only=True, nloglik_jax=None):

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
            hess = hessian(loglikelihood)(params)  
            if np.isnan(variance):
                raise ArithmeticError
            else:
                return sigma_sq_inv, hess, variance
        else:
            if full_hessian:
                # Use Hessian approximation to compute Fisher Information as the sample Hessian                                                  
                params_jax = jnp.asarray(params)                  
                hess = np.asarray(hessian(nloglik_jax)(params_jax)                                     )
                # Add small regularization to prevent singularity
                variance = -np.linalg.inv(hess + 1e-8 * np.eye(len(params)))            
                if np.any(np.isnan(variance)):                                
                    raise ArithmeticError
                else:
                    return variance, hess, np.diag(variance)
            if diag_hessian_only:                
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
                        raise ArithmeticError
                else:
                    return None, None, variance
    except Exception as e:
        raise ArithmeticError

def maximum_likelihood_estimator(
    likelihood_function, 
    initial_guess=None, 
    variance_method='jacobian', 
    optimization_method='L-BFGS-B',
    data=None, full_hessian=True, diag_hessian_only=True,
    loglikelihood_per_data_point=None, disp=False, niter=None, 
    jac=None, output_dir="/tmp/", plot_hessian=False, negloglik_jax=None, subdataset_name=None):
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
    # Set default initial guess if not provided
    if initial_guess is None:
        initial_guess = [0]
    optimize_kwargs = {
        'method': optimization_method,
        'x0': initial_guess,                
        'jac': '3-point'#,
        # 'hess': '2-point'
    }                   
    # Perform maximum likelihood estimation    
    if niter is not None:
        result = minimize(likelihood_function, **optimize_kwargs, options={"disp":disp, "maxiter":niter, "maxls":20})
    else:
        result = minimize(likelihood_function, **optimize_kwargs, options={"disp":disp, "maxls":20})
    
    mle = result.x          

    try:        
        variance_noninv, hessian, variance_diag = variance_estimation(estimation_result=result, loglikelihood=likelihood_function,
                                       data=data, full_hessian=full_hessian, diag_hessian_only=diag_hessian_only,
                                       loglikelihood_per_data_point=loglikelihood_per_data_point, nloglik_jax=negloglik_jax)
        result["variance_method"] = variance_method
        result["variance"] = variance_diag
        if full_hessian:
            result["full_hessian"] = hessian
            if plot_hessian:
                fig = visualise_hessian(hessian)
                fix_plot_layout_and_save(fig, "{}/hessian_{}_{}.html".format(output_dir, subdataset_name, datetime.now().strftime("%Y-%m-%d")),
                                        xaxis_title="", yaxis_title="", title="Full Hessian matrix estimate", showgrid=False, showlegend=False,
                                        print_png=True, print_html=True, print_pdf=False)
                
    except ArithmeticError as e:            
        # Fallback to zero variance if computation fails
        print(f"Variance estimation failed: {e}")
        variance = np.zeros((mle.shape[0], mle.shape[0]))
        result["variance_method"] = "{}-failed".format(variance_method)
        result["variance"] = variance
        
    return mle, result

def negative_loglik(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")      
    if "Phi" in params_hat.keys(): 
        Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
        delta = params_hat["delta"]
    else:
        Phi = np.zeros(Z.shape)
        delta = 0
    alpha = params_hat["alpha"]
    beta = params_hat["beta"]
    # c = params_hat["c"]
    gamma = params_hat["gamma"]    
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    nll = 0
    for i in range(K):
        for j in range(J):
            phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]
            errscale = sigma_e
            errloc = mu_e
            nll += Y[i, j]*norm.logcdf(1-phi, loc=errloc, scale=errscale) + (1-Y[i, j])*norm.logcdf(1-phi, loc=errloc, scale=errscale)

    return -nll

def negative_loglik_jax(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = jnp.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    Z = jnp.asarray(params_hat["Z"]).reshape((d, J), order="F")       
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
    nll = 0
    dst_func = lambda x, y: jnp.sum((x-y)**2)
    for i in range(K):
        for j in range(J):
            phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]
            errscale = sigma_e
            errloc = mu_e
            nll += Y[i, j]*jax.scipy.stats.norm.logcdf(1-phi, loc=errloc, scale=errscale) + (1-Y[i, j])*jax.scipy.stats.norm.logcdf(1-phi, loc=errloc, scale=errscale)

    return -nll[0]


def estimate_mle(args):

    current_pid = os.getpid()    
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m = args

    # load data    
    with open("{}/{}/{}".format(data_location, m, subdataset_name), "rb") as f:
        Y = pickle.load(f)
    from_row = int(subdataset_name.split("_")[1])
    to_row = int(subdataset_name.split("_")[2][:-7])
    # since each batch has N rows
    N = Y.shape[0]
    Y = Y.astype(np.int8).reshape((N, J), order="F")         
    nloglik = lambda x: negative_loglik(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict)
    nloglik_jax = lambda x: negative_loglik_jax(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict)
    
    # init parameter vector x0
    X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e = initialise_optimisation_vector_sobol(m=16, J=J, K=N, d=d)
    x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
    mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                            variance_method='jacobian', disp=True, 
                                            optimization_method=optimisation_method, 
                                            data=Y, full_hessian=False, diag_hessian_only=True,   
                                            loglikelihood_per_data_point=None, niter=niter, negloglik_jax=nloglik_jax, subdataset_name=subdataset_name)          
    params_hat = optimisation_dict2params(mle, param_positions_dict, J, N, d, parameter_names)
        
    # Place estimates and their variance in the correct positions in the global Theta parameter vector
    theta_global = np.zeros((parameter_space_dim,))
    theta_global_variance = np.zeros((parameter_space_dim,))
    if "Phi" not in params_hat.keys():
        params_hat["Phi"] = None
        params_hat["delta"] = None
    theta_global, theta_global_variance, param_positions_dict_global  = get_global_theta(from_row, to_row, parameter_space_dim, J, N, d, parameter_names, 
                                                                            params_hat["X"], params_hat["Z"], params_hat["Phi"], params_hat["alpha"], 
                                                                            params_hat["beta"], params_hat["gamma"], params_hat["delta"], 
                                                                            params_hat["mu_e"], params_hat["sigma_e"], result["variance"], total_K=K)

    grid_and_optim_outcome = dict()
    grid_and_optim_outcome["PID"] = [current_pid]        
    grid_and_optim_outcome["timestamp"] = [time.strftime("%Y-%m-%d %H:%M:%S")]    
    grid_and_optim_outcome["parameter names"] = parameter_names
    grid_and_optim_outcome["local theta"] = [mle.tolist()]
    grid_and_optim_outcome["Theta"] = [theta_global.tolist()]
    grid_and_optim_outcome["Theta Variance"] = [theta_global_variance.tolist()] 
    # in optimisation vector, not the global
    grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
    grid_and_optim_outcome["param_positions_dict_global"] = param_positions_dict_global

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
        
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m = args

        # load data    
        with open("{}/{}/{}".format(data_location, m, subdataset_name), "rb") as f:
            Y = pickle.load(f)
        from_row = int(subdataset_name.split("_")[1])
        to_row = int(subdataset_name.split("_")[2][:-7])
        # since each batch has N rows
        N = Y.shape[0]
        Y = Y.astype(np.int8).reshape((N, J), order="F")         
        nloglik = lambda x: negative_loglik(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict)
        nloglik_jax = lambda x: negative_loglik_jax(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict)
        
        # init parameter vector x0
        X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e = initialise_optimisation_vector_sobol(m=16, J=J, K=N, d=d)
        x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
        mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                                variance_method='jacobian', disp=True, 
                                                optimization_method=optimisation_method, 
                                                data=Y, full_hessian=True, diag_hessian_only=False, plot_hessian=True,   
                                                loglikelihood_per_data_point=None, niter=niter, negloglik_jax=nloglik_jax, output_dir=DIR_out)          
        params_hat = optimisation_dict2params(mle, param_positions_dict, J, N, d, parameter_names)
        
        print(subdataset_name)                        
        # Place estimates and their variance in the correct positions in the global Theta parameter vector
        theta_global = np.zeros((parameter_space_dim,))
        theta_global_variance = np.zeros((parameter_space_dim,))
        if "Phi" not in params_hat.keys():
            params_hat["Phi"] = None
            params_hat["delta"] = None
        theta_global, theta_global_variance, param_positions_dict_global  = get_global_theta(from_row, to_row, parameter_space_dim, J, N, d, parameter_names, 
                                                                            params_hat["X"], params_hat["Z"], params_hat["Phi"], params_hat["alpha"], 
                                                                            params_hat["beta"], params_hat["gamma"], params_hat["delta"], 
                                                                            params_hat["mu_e"], params_hat["sigma_e"], result["variance"], total_K=K)
                   
        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = [current_pid]        
        grid_and_optim_outcome["timestamp"] = [time.strftime("%Y-%m-%d %H:%M:%S")]    
        grid_and_optim_outcome["parameter names"] = parameter_names
        grid_and_optim_outcome["local theta"] = [mle.tolist()]
        grid_and_optim_outcome["Theta"] = [theta_global.tolist()]
        grid_and_optim_outcome["Theta Variance"] = [theta_global_variance.tolist()] 
        # in optimisation vector, not the global
        grid_and_optim_outcome["param_positions_dict"] = param_positions_dict
        grid_and_optim_outcome["param_positions_dict_global"] = param_positions_dict_global
        
        out_file = "{}/estimationresult_dataset_{}_{}.jsonl".format(DIR_out, from_row, to_row)
        self.append_to_json_file(grid_and_optim_outcome, output_file=out_file)

         


def main(J=2, K=2, d=1, N=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        niter=None, parameter_space_dim=None, trials=None):

    # ipdb.set_trace()
    if parallel:
        manager = ProcessManagerSynthetic(total_running_processes)        
    DIR_top = data_location      
    try:    
        if parallel:  
            manager.create_results_dict(optim_target="all")    
            while True:
                for m in range(trials):
                    path = pathlib.Path("{}/{}".format(data_location, m))  
                    subdatasets_names = [file.name for file in path.iterdir() if file.is_file() and "dataset_" in file.name]
                    DIR_out = "{}/{}/estimation/".format(DIR_top, m)
                    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
                    for dataset_index in range(len(subdatasets_names)):                    
                        subdataset_name = subdatasets_names[dataset_index]
                        args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, 
                                parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m)    
                                            
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
                subdatasets_names = [file.name for file in path.iterdir() if file.is_file() and "dataset_" in file.name]
                DIR_out = "{}/{}/estimation/".format(DIR_top, m)
                pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True) 
                for dataset_index in range(len(subdatasets_names)):               
                    subdataset_name = subdatasets_names[dataset_index]
                    args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, 
                            parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m)
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
    
    jax.config.update("jax_traceback_filtering", "off")
    parallel = True
    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = None
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "mu_e", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "mu_e", "sigma_e"]
    M = 2
    K = 1000
    J = 100
    sigma_e = 0.5
    d = 2    
    data_location = "/home/ioannischalkiadakis/ideal/idealpestimation/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e).replace(".", ""))
    total_running_processes = 30              
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
        dst_func=dst_func, niter=niter, parameter_space_dim=parameter_space_dim, trials=M)
    
    # for m in range(M):
    #     data_location = "/home/ioannischalkiadakis/ideal/idealpestimation/data_K{}_J{}_sigmae{}/{}/".format(K, J, str(sigma_e).replace(".", ""), m)
    #     params_out = combine_estimate_variance_rule("{}/estimation/".format(data_location), J, K, d, parameter_names)    
    #     out_file = "{}/params_out_global_theta_hat.jsonl".format(data_location)
    #     with open(out_file, 'a') as f:         
    #         writer = jsonlines.Writer(f)
    #         writer.write(params_out)