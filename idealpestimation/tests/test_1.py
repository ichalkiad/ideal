from idealpestimation.src.mle import maximum_likelihood_estimator
from idealpestimation.src.utils import uni_gaussian_hessian
from scipy import stats
from scipy import stats
import jax
import jax.numpy as jnp
import numpy as np
import ipdb

def mle_test_parameter_estimation(distribution_type='poisson', sample_size=1000, num_trials=100):
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
    full_hessian = True 
    diag_hessian_only = True
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
                                                        variance_method='jacobian', data=data, full_hessian=full_hessian, 
                                                        diag_hessian_only=diag_hessian_only, 
                                                        loglikelihood_per_data_point=poisson_likelihood_at_data_point, 
                                                        plot_hessian=True)                     
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
        hess_diag_estimated = []
        for _ in range(num_trials):
            # Generate data from Normal distribution
            data = np.random.normal(true_mu, true_sigma, sample_size)       
            nloglik = lambda params: normal_likelihood(params, data)    
            # Estimate parameters
            mle, results = maximum_likelihood_estimator(nloglik, 
                                                        initial_guess=[np.mean(data), np.std(data)],
                                                        variance_method='jacobian',
                                                        data=data, full_hessian=full_hessian, diag_hessian_only=diag_hessian_only,
                                                        loglikelihood_per_data_point=normal_likelihood_at_data_point, plot_hessian=True)            
            mus_estimated.append(mle[0])
            sigmas_estimated.append(mle[1])
            if full_hessian:
                # using estimated parameters in Hessian formula
                true_hess = uni_gaussian_hessian(mle[0], mle[1], data)
                hess_diag_estimated.append(np.mean((np.diag(true_hess)-np.diag(np.array(results.full_hessian)))**2))            

        # Compute Mean Squared Errors
        mse_results['normal_mu'] = np.mean((np.array(mus_estimated) - true_mu)**2)
        mse_results['normal_sigma'] = np.mean((np.array(sigmas_estimated) - true_sigma)**2)  
        if full_hessian:
            mse_results["hessian diag vs jax"] = np.mean(np.asarray(hess_diag_estimated))
        mse_results["all_results"] = results
    else:
        raise ValueError("Invalid distribution type. Choose 'poisson' or 'normal'.")
    
    return mse_results


def test_1():
    """
    Run tests for different distributions and sample sizes
    """
    print("Maximum Likelihood Estimation Tests:")
    
    # Test Poisson distribution
    print("\nPoisson Distribution Test:")
    poisson_results = mle_test_parameter_estimation(
        distribution_type='poisson', 
        sample_size=5000, 
        num_trials=200
    )
    print("Poisson Lambda MSE:", poisson_results.get('poisson', 'N/A'))
    print("Hessian Hessian diag MSE:", poisson_results.get('hessian diag vs jax', 'N/A'))
    assert(poisson_results["poisson"] < 0.02)
    print(poisson_results)
    
    # Test Normal distribution
    print("\nNormal Distribution Test:")
    normal_results = mle_test_parameter_estimation(
        distribution_type='normal', 
        sample_size=5000, 
        num_trials=200
    )
    print("Normal Mu MSE:", normal_results.get('normal_mu', 'N/A'))
    print("Normal Sigma MSE:", normal_results.get('normal_sigma', 'N/A'))
    print("Normal Hessian diag MSE:", normal_results.get('hessian diag vs jax', 'N/A'))
    assert(normal_results["normal_mu"] < 0.005)
    assert(normal_results["normal_sigma"] < 0.005)

    print(normal_results)

if __name__ == "__main__":
    test_1()