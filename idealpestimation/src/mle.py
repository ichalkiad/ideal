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
from idealpestimation.src.parallel_manager import ProcessManager, \
                                                    jsonlines
from idealpestimation.src.utils import params2optimisation_dict, \
                                            optimisation_dict2params, \
                                                initialise_optimisation_vector_sobol, \
                                                    visualise_hessian, fix_plot_layout_and_save, \
                                                        get_hessian_diag_jax, get_jacobian, combine_estimate_variance_rule

def variance_estimation(estimation_result, loglikelihood=None, loglikelihood_per_data_point=None, 
                        data=None, full_hessian=True, diag_hessian_only=True):

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
                hess = hessian(loglikelihood)(params)                                     
                # Add small regularization to prevent singularity
                variance = -np.linalg.inv(hess + 1e-8 * np.eye(len(params)))            
                if np.any(np.isnan(variance)):                                
                    raise ArithmeticError
                else:
                    return variance, hess, np.diag(variance)
            if diag_hessian_only:
                hess_jax = get_hessian_diag_jax(loglikelihood, params)                                     
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
    jac=None, output_dir="/tmp/", plot_hessian=False):
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
                                       loglikelihood_per_data_point=loglikelihood_per_data_point)
        result["variance_method"] = variance_method
        result["variance"] = variance_diag
        if full_hessian:
            result["full_hessian"] = hessian
            if plot_hessian:
                fig = visualise_hessian(hessian)
                fix_plot_layout_and_save(fig, "{}/hessian_{}.html".format(output_dir, datetime.now().strftime("%Y-%m-%d")),
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
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
    alpha = params_hat["alpha"]
    beta = params_hat["beta"]
    # c = params_hat["c"]
    gamma = params_hat["gamma"]
    delta = params_hat["delta"]
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

def negative_loglik_at_data_point(i, j, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")       
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
    alpha = params_hat["alpha"]
    beta = params_hat["beta"]
    # c = params_hat["c"]
    gamma = params_hat["gamma"]
    delta = params_hat["delta"]
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]    
    phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]
    errscale = sigma_e
    errloc = mu_e
    nll = Y[i, j]*norm.logcdf(1-phi, loc=errloc, scale=errscale) + (1-Y[i, j])*norm.logcdf(1-phi, loc=errloc, scale=errscale)

    return nll


def estimate_mle(args):

    ipdb.set_trace()

    current_pid = os.getpid()    
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter = args

    # load data    
    with open("{}/{}".format(data_location, subdataset_name), "rb") as f:
        Y = pickle.load(f)
    # since each batch has N rows
    Y = Y.astype(np.int8).reshape((N, J), order="F")     
    nloglik = lambda x: negative_loglik(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict)
    # nloglik_at_data_point = lambda i, j, theta, Y: negative_loglik_at_data_point(i, j, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict)

    # init parameter vector x0
    X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e = initialise_optimisation_vector_sobol(m=16, J=J, K=N, d=d)
    x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
    mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                            variance_method='jacobian', 
                                            optimization_method=optimisation_method, 
                                            data=Y, full_hessian=True, diag_hessian_only=True,   ####################################
                                            loglikelihood_per_data_point=None, niter=niter)          
    params_hat = optimisation_dict2params(mle.x, param_positions_dict, J, K, d, parameter_names)
    
    ipdb.set_trace()                           
    
    grid_and_optim_outcome = dict()
    grid_and_optim_outcome["PID"] = [current_pid]        
    grid_and_optim_outcome["timestamp"] = [time.strftime("%Y-%m-%d %H:%M:%S")]    
    grid_and_optim_outcome["Theta"] = [mle.x]
    grid_and_optim_outcome["Theta Variance"] = [result["variance"]]
    grid_and_optim_outcome["param_positions_dict"] = param_positions_dict

    out_file = "{}/estimationresult_dataset{}.jsonl".format(DIR_out, dataset_index)
    with open(out_file, 'a') as f:         
        writer = jsonlines.Writer(f)
        writer.write(grid_and_optim_outcome)
            
class ProcessManagerSynthetic(ProcessManager):
    def __init__(self, max_processes):
        super().__init__(max_processes)
    
    # estimate_with_retry
    def worker_process(self, args):

        current_pid = os.getpid()
        with self.execution_counter.get_lock():
            self.execution_counter.value += 1
            self.shared_dict[current_pid] = self.execution_counter.value
        
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, dst_func = args

        # load data    
        with open("{}/{}".format(data_location, subdataset_name), "rb") as f:
            Y = pickle.load(f)
        
        nloglik = lambda x: negative_loglik(Y, J, K, d, parameter_names, x, dst_func)
        
        # init parameter vector x0
        X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e = initialise_optimisation_vector_sobol(m=16, J=J, K=K, d=d)
        x0, param_positions_dict = params2optimisation_dict(J, K, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
        mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                                            variance_method='jacobian', 
                                                            optimization_method=optimisation_method)          
        params_hat = optimisation_dict2params(mle.x, param_positions_dict, J, K, d, parameter_names)
                   
        grid_and_optim_outcome = dict()
        grid_and_optim_outcome["PID"] = [current_pid]        
        grid_and_optim_outcome["timestamp"] = [time.strftime("%Y-%m-%d %H:%M:%S")]    
        grid_and_optim_outcome["Theta"] = [mle.x]
        grid_and_optim_outcome["Theta Variance"] = [result["variance"]]
        
        out_file = "{}/estimationresult_dataset{}.jsonl".format(DIR_out, dataset_index)
        self.append_to_json_file(grid_and_optim_outcome, output_file=out_file)


def main(J=2, K=2, d=1, N=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, niter=None):

    if parallel:
        manager = ProcessManagerSynthetic(total_running_processes)        
    DIR_top = data_location
    path = pathlib.Path(data_location)
    subdatasets_names = [file.name for file in path.iterdir() if file.is_file() and "dataset_" in file.name]
    DIR_out = "{}/estimation/".format(DIR_top)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     

    try:    
        if parallel:  
            manager.create_results_dict(optim_target="all")              
        for dataset_index in range(len(subdatasets_names)):
            subdataset_name = subdatasets_names[dataset_index]
            args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func, niter)    
            if parallel:                          
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
            else:
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

    data_location = "./idealpestimation/data"    
    total_running_processes = 1
    parallel = False
    optimisation_method = "L-BFGS-B"
    parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "mu_e", "sigma_e"]
    with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_location), mode="r") as f:
        for result in f.iter(type=dict, skip_invalid=True):                              
            J = result["J"]
            K = result["K"]
            d = result["d"]

    parameter_space_dim = (K+2*J)*d + J + K + 4
    # for distributing per N rows
    N = round(parameter_space_dim/J)
    dst_func = lambda x, y: np.sum((x-y)**2)
    niter = 2
    main(J=J, K=K, d=d, N=N, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, niter=niter)
    combine_estimate_variance_rule("{}/estimation/".format(data_location), J, K, d, parameter_names)

    # read outputs and combine