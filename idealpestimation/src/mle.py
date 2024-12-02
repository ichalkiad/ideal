import os 
import sys
import time 
import ipdb
import jax
import jax.numpy as jnp
from jax import jvp, grad, hessian
import pathlib
import jsonlines
import pickle
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from idealpestimation.src.parallel_manager import ProcessManager, \
                                                    jsonlines
from idealpestimation.src.utils import params2optimisation_dict, \
                                            optimisation_dict2params, \
                                                initialise_optimisation_vector_sobol

def uni_gaussian_hessian(mu, sigma, x):
    
    x = np.asarray(x)    
    z = (x - mu) / sigma
    
    # Hessian matrix elements for log-likelihood
    # First parameter (mu): [d²/d2mu]
    d2l_dmu2 = -len(x) / (sigma**2)
    # print(d2l_dmu2)
    # Second parameter (sigma): [d²/d2σ]
    d2l_d2sigma = len(x) / (sigma**2) - 3*np.sum(z**2) / (sigma**2)
    # print(d2l_d2sigma)
    # Mixed partial derivative [d²/dμdσ]
    d2l_dmu_dsigma2 = -2*np.sum(z) / (sigma**2)
    # print(d2l_dmu_dsigma2)
        
    # Construct the Hessian matrix
    hessian = np.array([
        [d2l_dmu2, d2l_dmu_dsigma2],
        [d2l_dmu_dsigma2, d2l_d2sigma]
    ])
    
    return hessian


def aux_test_parameter_estimation(distribution_type='poisson', sample_size=1000, num_trials=100):
    """
    Test maximum likelihood estimation for Poisson and Normal distributions.
    
    Parameters:
    -----------
    distribution_type : str, optional
        Type of distribution to test. Options: 'poisson' or 'normal'
    sample_size : int, optional
        Number of samples to generate in each trial
    num_trials : int, optional
        Number of Monte Carlo trials to run
    
    Returns:
    --------
    dict
        Dictionary containing mean squared errors for parameter estimates
    """
    # Set random seed for reproducibility
    np.random.seed(2611)    
    # Containers for results
    mse_results = {}

    if distribution_type == 'poisson':
        # True parameter (lambda)
        true_lambda = 4.5

        def poisson_likelihood(theta):
            """Negative log-likelihood for Poisson"""
            try:
                nll = -np.sum(stats.poisson.logpmf(data, theta))
            except:
                nll = -jnp.sum(jax.scipy.stats.poisson.logpmf(data, theta))
            return nll

        def poisson_likelihood_at_data_point(i, j, theta, data):
            xp = data[i, j]            
            return stats.poisson.logpmf(xp, theta)
        
        # Monte Carlo trials
        lambdas_estimated = []
        for _ in range(num_trials):
            # Generate data from Poisson distribution
            data = np.random.poisson(true_lambda, sample_size)            
            # Estimate parameters
            mle, results = maximum_likelihood_estimator(poisson_likelihood, initial_guess=[np.mean(data)], 
                                                        variance_method='jacobian', data=data, 
                                                        loglikelihood_per_data_point=poisson_likelihood_at_data_point)                     
            lambdas_estimated.append(mle[0])        
        # Compute Mean Squared Error
        mse_results['poisson'] = np.mean((np.array(lambdas_estimated) - true_lambda)**2)
        mse_results["all_results"] = results
    
    elif distribution_type == 'normal':
        # True parameters (mu, sigma)
        true_mu = 2.0
        true_sigma = 1.5
        
        def normal_likelihood(params, data):
            """Negative log-likelihood for Normal distribution"""
            mu, sigma = params
            try:
                nll = -np.sum(stats.norm.logpdf(data, mu, sigma))
            except:
                nll = -jnp.sum(jax.scipy.stats.norm.logpdf(data, mu, sigma))
            return nll

        def normal_likelihood_at_data_point(i, j, params, data):            
            mu, sigma = params
            xp = data[i, j]
            return stats.norm.logpdf(xp, mu, sigma)
    
        # Monte Carlo trials
        mus_estimated = []
        sigmas_estimated = []
        hess_estimated = []
        full_hessian = False 
        diag_hessian_only = True
        for _ in range(num_trials):
            # Generate data from Normal distribution
            data = np.random.normal(true_mu, true_sigma, sample_size)       
            nloglik = lambda params: normal_likelihood(params, data)    
            # Estimate parameters
            mle, results = maximum_likelihood_estimator(nloglik, 
                                                        initial_guess=[np.mean(data), np.std(data)],
                                                        variance_method='jacobian',
                                                        data=data, full_hessian=full_hessian, diag_hessian_only=diag_hessian_only,
                                                        loglikelihood_per_data_point=normal_likelihood_at_data_point)            
            mus_estimated.append(mle[0])
            sigmas_estimated.append(mle[1])
            if full_hessian:
                true_hess = uni_gaussian_hessian(true_mu, true_sigma, data)
                hess_estimated.append(np.mean((true_hess-np.array(results.full_hessian))**2))            

        # Compute Mean Squared Errors
        mse_results['normal_mu'] = np.mean((np.array(mus_estimated) - true_mu)**2)
        mse_results['normal_sigma'] = np.mean((np.array(sigmas_estimated) - true_sigma)**2)    ##################################################
        if full_hessian:
            mse_results["hessian vs jax"] = np.mean(np.asarray(hess_estimated))
        mse_results["all_results"] = results
    else:
        raise ValueError("Invalid distribution type. Choose 'poisson' or 'normal'.")
    
    return mse_results

def get_jacobian(params, likelihood_function):
    """Numerical approximation of Jacobian"""
    fprime = approx_fprime(params, likelihood_function)    
    return fprime

# def hessian(f):
#     # f: function w.r.t to parameter vector
#     return jax.jacfwd(jax.jacrev(f))

def hvp(f, x, v):
  return jvp(grad(f), (x,), (v,))[1]

def get_hessian_diag_jax(f, x):
    # f: function w.r.t to parameter vector x
    print(jnp.diag(hessian(f)(x)))
    print(hvp(f, x, jnp.ones_like(x)))
    return hvp(f, x, jnp.ones_like(x))

def get_hessian_diag(likelihood_function, params, eps=1e-8):
    """
    Compute Hessian's diagonal using numerical second-order central differences
    """
    n = len(params)
    hessian = np.zeros((n,))
    
    # Compute mixed partial derivatives
    for i in range(n):
        # Central difference approximation for second derivatives
        x_ij_plus = params.copy()
        x_ij_minus = params.copy()
        x_ij_plus[i] += eps
        x_ij_minus[i] -= eps
        
        x_i_plus = params.copy()
        x_i_minus = params.copy()
        x_i_plus[i] += eps
        x_i_minus[i] -= eps
        
        # Compute mixed second partial derivative
        hessian[i, i] = (
            likelihood_function(x_ij_plus) - likelihood_function(x_ij_minus) - 
            likelihood_function(x_i_plus) + likelihood_function(x_i_minus)
        ) / (4 * eps**2)
    
    return hessian

  
def variance_estimation(estimation_result, loglikelihood=None, loglikelihood_per_data_point=None, data=None, full_hessian=True, diag_hessian_only=True):

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
            sigma_sq_inv = np.mean(np.diag(scores@scores.T))                
            # Compute variance as the inverse of the Fisher Information
            # Add small regularization to prevent singularity
            variance = np.linalg.inv(sigma_sq_inv + 1e-8 * np.eye(len(params)))
            if np.isnan(variance):
                raise ArithmeticError
            else:
                return sigma_sq_inv, None, variance
        else:
            if full_hessian:
                # use Hessian approximation            
                hess = hessian(loglikelihood)(params)
                # Compute Fisher Information as the sample Hessian         
                sigma_sq_inv = -hess                            
                # Add small regularization to prevent singularity
                variance = np.linalg.inv(sigma_sq_inv + 1e-8 * np.eye(len(params)))            
                if np.any(np.isnan(variance)):                                
                    raise ArithmeticError
                else:
                    return variance, hess, np.diag(variance)
            if diag_hessian_only:
                # use Hessian approximation            
                hess = get_hessian_diag(loglikelihood, params)  ###############################################
                ipdb.set_trace()
                hess_jax = get_hessian_diag_jax(loglikelihood, params)         
                sigma_sq_inv = -hess_jax                            
                # Add small regularization to prevent singularity
                variance = 1/sigma_sq_inv            
                if np.any(np.isnan(variance)):                                
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
    loglikelihood_per_data_point=None, disp=False, niter=None, jac=None):
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
        'jac': '2-point',
        'hess': '2-point'
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
    DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func = args

    # load data    
    with open("{}/{}".format(data_location, subdataset_name), "rb") as f:
        Y = pickle.load(f)
    # since each batch has N rows
    Y = Y.astype(np.int8).reshape((N, J), order="F")     
    nloglik = lambda x: negative_loglik(x, Y, J, N, d, parameter_names, dst_func, param_positions_dict)
    nloglik_at_data_point = lambda i, j, theta, Y: negative_loglik_at_data_point(i, j, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict)

    # init parameter vector x0
    X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e = initialise_optimisation_vector_sobol(m=16, J=J, K=N, d=d)
    x0, param_positions_dict = params2optimisation_dict(J, N, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e)
    mle, result = maximum_likelihood_estimator(nloglik, initial_guess=x0, 
                                            variance_method='jacobian', 
                                            optimization_method=optimisation_method, 
                                            data=Y, full_hessian=True, diag_hessian_only=False,   ####################################
                                            loglikelihood_per_data_point=nloglik_at_data_point)          
    params_hat = optimisation_dict2params(mle.x, param_positions_dict, J, K, d, parameter_names)
    
    ipdb.set_trace()                           
    
    grid_and_optim_outcome = dict()
    grid_and_optim_outcome["PID"] = [current_pid]        
    grid_and_optim_outcome["timestamp"] = [time.strftime("%Y-%m-%d %H:%M:%S")]    
    grid_and_optim_outcome["Theta"] = [mle.x]
    grid_and_optim_outcome["Theta Variance"] = [result["variance"]]
            
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
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2):

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
            args = (DIR_out, data_location, subdataset_name, dataset_index, optimisation_method, parameter_names, J, K, d, N, dst_func)    
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
    main(J=J, K=K, d=d, N=N, total_running_processes=total_running_processes, 
         data_location=data_location, parallel=parallel, 
         parameter_names=parameter_names, optimisation_method=optimisation_method, dst_func=dst_func)


    # read outputs and combine