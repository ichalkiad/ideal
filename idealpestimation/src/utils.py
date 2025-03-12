from scipy.stats import qmc
import ipdb
import numpy as np
import math
import time
from datetime import timedelta, datetime
import pathlib
from scipy.stats import norm, multivariate_normal, invgamma
import plotly.graph_objs as go
import plotly.io as pio
import jax
import jax.numpy as jnp
from jax import jvp, grad
import jsonlines
from scipy.optimize import approx_fprime
import argparse
from scipy.special import gammaincc, gammaln
import itertools
from sklearn.decomposition import PCA
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots
from itertools import product


def fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=False,
                            print_png=True, print_html=True, print_pdf=True):
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(title=title, plot_bgcolor='rgb(255,255,255)',
                        yaxis=dict(
                            title=yaxis_title,
                            titlefont_size=20,
                            tickfont_size=20,
                            showgrid=showgrid,
                        ),
                        xaxis=dict(
                            title=xaxis_title,
                            titlefont_size=20,
                            tickfont_size=20,
                            showgrid=showgrid
                        ),
                        font=dict(
                            size=20
                        ),
                        showlegend=showlegend)
        if showlegend:
            fig.update_layout(legend=dict(
                yanchor="top",
                y=1.1,  # 0.01
                xanchor="right",  # "left", #  "right"
                x=1,    #0.01,  # 0.99
                bordercolor="Black",
                borderwidth=0.3,
                font=dict(
                    size=18,
        )))

        if print_html:
            pio.write_html(fig, savename, auto_open=False)
        if print_pdf:
            pio.write_image(fig, savename.replace(".html", ".pdf"), engine="kaleido")
        if print_png:
            pio.write_image(fig, savename.replace("html", "png"), width=1540, height=871, scale=1)
            

####################### MLE #############################

def params2optimisation_dict(J, K, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, sigma_e):
    
    param_positions_dict = dict()
    optim_vector = []
    k = 0
    for param in parameter_names:
        if param == "X":
            param_positions_dict[param] = (k, k + K*d)   
            Xvec = X.reshape((d*K,), order="F").tolist()        
            optim_vector.extend(Xvec)            
            k += K*d    
        elif param == "Z":
            param_positions_dict[param] = (k, k + J*d)            
            Zvec = Z.reshape((d*J,), order="F").tolist()         
            optim_vector.extend(Zvec)            
            k += J*d
        elif param == "Phi":            
            param_positions_dict[param] = (k, k + J*d)            
            Phivec = Phi.reshape((d*J,), order="F").tolist()         
            optim_vector.extend(Phivec)            
            k += J*d
        elif param == "beta":
            param_positions_dict[param] = (k, k + K)               
            optim_vector.extend(beta.tolist())            
            k += K    
        elif param == "alpha":
            param_positions_dict[param] = (k, k + J)               
            optim_vector.extend(alpha.tolist())                        
            k += J    
        elif param == "gamma":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(gamma)
            k += 1
        elif param == "delta":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(delta)
            k += 1
        elif param == "sigma_e":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(sigma_e)
            k += 1

    return optim_vector, param_positions_dict

def optimisation_dict2params(optim_vector, param_positions_dict, J, K, d, parameter_names):
    
    params_out = dict()
    for param in parameter_names:
        param_out = optim_vector[param_positions_dict[param][0]:param_positions_dict[param][1]]
        if param == "X":            
            param_out = param_out.reshape((d, K), order="F")                     
        elif param in ["Z"]:            
            param_out = param_out.reshape((d, J), order="F")                      
        elif param in ["Phi"]:            
            param_out = param_out.reshape((d, J), order="F")                                
        params_out[param] = param_out
        
    return params_out

def optimisation_dict2paramvectors(optim_vector, param_positions_dict, J, K, d, parameter_names):
    
    params_out = dict()
    for param in parameter_names:        
        param_out = optim_vector[param_positions_dict[param][0]:param_positions_dict[param][1]]                
        params_out[param] = param_out.tolist()
        
    return params_out


def create_constraint_functions(n, param_positions_dict=None, sum_z_constant=0, min_sigma_e=1e-6, args=None):
    
    if args is not None:
        DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,\
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true = args

        grid_width_std = 5  
        bounds = []
        for param in parameter_names:
            if param == "X":
                bounds.extend([(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0])]*(K*d)) 
            elif param == "Z":
                bounds.extend([(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0])]*(J*d)) 
            elif param == "Phi":
                bounds.extend([(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0])]*(J*d)) 
            elif param == "alpha":
                bounds.extend([(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha)]*J) 
            elif param == "beta":
                bounds.extend([(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta)]*K) 
            elif param == "gamma":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma)) 
            elif param == "delta":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta)) 
    else:
        # def sum_zero_constraint(x):
        #     """Constraint: Sum of Z's should be a constant - default 0, i.e. balanced politician set"""
        #     return np.sum(x[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]])
        bounds = [(None, None)]*(n-1)
    bounds.append((min_sigma_e, None))
    
    return bounds, None


def initialise_optimisation_vector_sobol(m=16, J=2, K=2, d=1, min_sigma_e=0.1):

    sobol_generators = dict()
    # sampler = qmc.Sobol(d=1, scramble=False)   
    # sample = sampler.random_base2(m=m)
    # gamma = sample[:int(len(sample)/2)]
    # delta = sample[int(len(sample)/2):]
    # sobol_generators["gammadelta"] = [sampler]
    
    # gamma-delta: unidimensional parameters, generate uniform grid
    gamma = np.linspace(-2, 2, 100).tolist()
    if 0 in gamma:
        gamma.remove(0)
    delta = np.linspace(-2, 2, 100).tolist()
    if 0 in delta:
        delta.remove(0)        
    
    # sampler = qmc.Sobol(d=d, scramble=False)   
    # sample = sampler.random_base2(m=2)                               
    # c = sample[0]    
    # sobol_generators["c"] = [sampler]

    # K will be quite high dimensional so better use Sobol sequence
    sampler = qmc.Sobol(d=K, scramble=False)   
    sample = sampler.random_base2(m=m)                               
    beta = sample.tolist()
    sobol_generators["beta"] = [sampler]

    # J will be quite high dimensional so better use Sobol sequence
    sampler = qmc.Sobol(d=J, scramble=False)   
    sample = sampler.random_base2(m=m)                               
    alpha = sample.tolist()
    sobol_generators["alpha"] = [sampler]

    # Could also use uniform grid, given that d=2
    sampler = qmc.Sobol(d=d, scramble=False)   
    sample = sampler.random_base2(m=m)
    Phi = sample[:int(len(sample)/3), :].tolist()    
    Z = sample[int(len(sample)/3):2*int(len(sample)/3), :].tolist()    
    X = sample[2*int(len(sample)/3):, :].tolist()
    sobol_generators["XZPhi"] = [sampler]
            
    sigma_e = np.linspace(min_sigma_e, 5, 100).tolist()

    return X, Z, Phi, alpha, beta, gamma, delta, sigma_e

def visualise_hessian(hessian, title='Hessian matrix'):
    
    hessian = np.asarray(hessian)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=hessian,
        colorscale='RdBu_r',  
        zmin=np.min(np.diag(hessian)), 
        zmax=np.max(np.diag(hessian)),
        colorbar=dict(title='Hessian Value')
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Parameter Index',
        yaxis_title='Parameter Index'        
    )
    
    return fig

def uni_gaussian_hessian(mu, sigma, x):
    
    x = np.asarray(x)    
    z = (x - mu) / sigma
    
    # First parameter (mu): [d²/d2mu]
    d2l_dmu2 = len(x) / (sigma**2)
    # print(d2l_dmu2)
    # Second parameter (sigma): [d²/d2σ]
    d2l_d2sigma = -len(x) / (sigma**2) + 3*np.sum(z**2) / (sigma**2)
    # print(d2l_d2sigma)
    # Mixed partial derivative [d²/dμdσ]
    d2l_dmu_dsigma2 = 2*np.sum(z) / (sigma**2)
    # print(d2l_dmu_dsigma2)
        
    hessian = np.array([
        [d2l_dmu2, d2l_dmu_dsigma2],
        [d2l_dmu_dsigma2, d2l_d2sigma]
    ])
    
    return hessian

def get_jacobian(params, likelihood_function):
    """Numerical approximation of Jacobian"""
    fprime = approx_fprime(params, likelihood_function)    
    return fprime

def hvp(f, x, v):
  return jvp(grad(f), (x,), (v,))[1]

def get_hessian_diag_jax(f, x):
    # f: function w.r.t to parameter vector x
    return hvp(f, x, jnp.ones_like(x))
    
def combine_estimate_variance_rule(DIR_out, J, K, d, parameter_names, error_dict, theta_true, param_positions_dict):

    params_out = dict()
    params_out["X"] = np.zeros((d*K,))
    params_out["beta"] = np.zeros((K,))    
    for param in parameter_names:             
        weighted_estimate = None
        weight = None
        theta = None
        all_weights = []
        all_estimates = []
        path = pathlib.Path(DIR_out)  
        subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]                    
        for dataset_index in range(len(subdatasets_names)):                    
            subdataset_name = subdatasets_names[dataset_index]                        
            DIR_read = "{}/{}/estimation/".format(DIR_out, subdataset_name)
            path = pathlib.Path(DIR_read)  
            estimates_names = [file.name for file in path.iterdir() if file.is_file() and "estimationresult_dataset" in file.name]
            if len(estimates_names) > 1:
                raise AttributeError("Should have 1 output estimation file.")
            for estim in estimates_names:
                with jsonlines.open("{}/{}".format(DIR_read, estim), mode="r") as f: 
                    for result in f.iter(type=dict, skip_invalid=True):                                                              
                        if param in ["X", "beta"]:
                            # single estimate per data split
                            theta = result[param]
                            namesplit = estim.split("_")
                            start = int(namesplit[2])
                            end   = int(namesplit[3].replace(".jsonl", ""))
                            if param == "X":
                                params_out[param][start*d:end*d] = theta
                            else:
                                params_out[param][start:end] = theta
                            if param == "beta":
                                mse_trial_m_batch_index = np.mean((theta - theta_true[param_positions_dict[param][0]+start:param_positions_dict[param][0]+end])**2)
                            else:
                                X_true = np.asarray(theta_true[param_positions_dict[param][0]+start*d:param_positions_dict[param][0]+end*d]).reshape((d, end-start), order="F")
                                X_hat = np.asarray(theta).reshape((d, end-start), order="F")
                                Rx, tx, mse_trial_m_batch_index, _ = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat)
                        else:                            
                            weight = result["variance_{}".format(param)]                            
                            theta = result[param]
                            all_weights.append(weight)
                            all_estimates.append(theta)
                            if param == "Z":
                                Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")  
                                Z_hat = np.asarray(theta).reshape((d, J), order="F")
                                Rz, tz, mse_trial_m_batch_index, _ = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat)
                            else:
                                mse_trial_m_batch_index = np.mean((theta - theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]])**2)
            
            if error_dict is not None:
                error_dict[param].append(mse_trial_m_batch_index)

        if param in ["X", "beta"]:
            params_out[param] = params_out[param]        
        else:                
            all_weights = np.stack(all_weights)
            if param not in ["Z", "Phi", "alpha"]:
                all_weights = all_weights.flatten()
            all_estimates = np.stack(all_estimates)
            # sum acrocs each coordinate's weight
            all_weights_sum = np.sum(all_weights, axis=0)
            all_weights_norm = all_weights/all_weights_sum
            assert np.allclose(np.sum(all_weights_norm, axis=0), np.ones(all_weights_sum.shape))
            # element-wise multiplication
            weighted_estimate = np.sum(all_weights_norm*all_estimates, axis=0)
            params_out[param] = weighted_estimate
    
    return params_out, error_dict

def parse_input_arguments():

    parser = argparse.ArgumentParser(
        description='Input values: trials, either integer or range of values, e.g. 0-5, K, J, sigmae as a string without a decimal point, e.g. 0.5 -> 05.\
            Pass all arguments otherwise those hardcoded will be used.'
    )
    
    # Add optional arguments of different types with default values
    parser.add_argument(
        '--trials',
        type=str,
        default=None,
        help='Trial folders to run.'
    )
    
    parser.add_argument(
        '--K',
        type=int,
        default=None,
        help='K users'
    )
    
    parser.add_argument(
        '--J',
        type=int,
        default=None,
        help='J users'
    )

    parser.add_argument(
        '--sigmae',
        type=str,
        default=None,
        help='sigma_e'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='parallel execution flag - omit for single thread execution'
    )

    parser.add_argument(
        '--elementwise',
        action='store_true',
        help='for ICM: elementwise posterior optimisation'
    )

    parser.add_argument(
        '--evaluate_posterior',
        action='store_true',
        help='for ICM: evaluate posterior on a grid (True), differentiate (False)'
    )

    parser.add_argument(
        '--total_running_processes',
        type=int,
        default=10,
        help='total number of running processes in parallel'
    )
    
    return parser.parse_args()

def negative_loglik(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = 0
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
    mu_e = 0
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


def collect_mle_results(data_topdir, M, K, J, sigma_e_true, d, parameter_names, param_positions_dict):
    
    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))
    with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_topdir), "r") as f:
        for result in f.iter(type=dict, skip_invalid=True):
            for param in parameter_names:
                theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 

    params_out_jsonl = dict()
    estimation_error_per_trial_per_batch = dict()
    estimation_error_per_trial = dict()
    for param in parameter_names:
        estimation_error_per_trial[param] = []
    for m in range(M):
        fig_m_over_databatches = go.Figure()
        estimation_error_per_trial_per_batch[m] = dict()
        for param in parameter_names:            
            estimation_error_per_trial_per_batch[m][param] = []
        data_location = "{}/{}/".format(data_topdir, m)
        params_out, estimation_error_per_trial_per_batch[m] = combine_estimate_variance_rule(data_location, J, K, d, parameter_names, 
                                                                estimation_error_per_trial_per_batch[m], theta_true, param_positions_dict)    
        for param in parameter_names:
            if param == "X":                
                params_out_jsonl[param] = params_out[param].reshape((d*K,), order="F").tolist()     
                X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                X_hat = np.asarray(params_out_jsonl[param]).reshape((d, K), order="F")
                Rx, tx, mse_x, mse_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat)
                estimation_error_per_trial[param].append(mse_x)
            elif param == "Z":
                params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                Z_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                Rz, tz, mse_z, mse_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat)
                estimation_error_per_trial[param].append(mse_z)                      
            elif param == "Phi":            
                params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                Phi_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                Phi_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                Rphi, tphi, mse_phi, mse_phi_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Phi_true, param_hat=Phi_hat)
                estimation_error_per_trial[param].append(mse_phi)                        
            elif param in ["beta", "alpha"]:
                params_out_jsonl[param] = params_out[param].tolist()     
                mse = np.mean((theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])**2)  
                estimation_error_per_trial[param].append(mse)        
            else:
                params_out_jsonl[param] = params_out[param]
                mse = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])**2
                estimation_error_per_trial[param].append(mse[0]) 
            
            fig_m_over_databatches.add_trace(go.Box(
                    y=estimation_error_per_trial_per_batch[m][param], 
                    x=[param]*len(estimation_error_per_trial_per_batch[m][param]),
                    showlegend=False,  
                    boxpoints='outliers'
                    ))
            
        savename = "{}/mle_estimation_plots/mse_trial{}_perparam_unweighted_boxplot.html".format(data_topdir, m)
        pathlib.Path("{}/mle_estimation_plots/".format(data_topdir)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig_m_over_databatches, savename, xaxis_title="Parameter", yaxis_title="MSE Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, print_pdf=False)
            
        out_file = "{}/params_out_global_theta_hat.jsonl".format(data_location)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write(params_out_jsonl)
    
    # box plot - mse relativised per parameter over trials
    fig = go.Figure()
    for param in parameter_names:
        fig.add_trace(go.Box(
                        y=np.asarray(estimation_error_per_trial[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_error_per_trial[param]), boxpoints='outliers'                                
                    ))
    savename = "{}/mle_estimation_plots/mse_overAllTrials_perparam_weighted_boxplot.html".format(data_topdir)
    fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="MSE Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)



def collect_mle_results_batchsize_analysis(data_topdir, batchsizes, M, K, J, sigma_e_true, d, parameter_names, param_positions_dict):
    
    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))
    with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(data_topdir), "r") as f:
        for result in f.iter(type=dict, skip_invalid=True):
            for param in parameter_names:
                theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 

    params_out_jsonl = dict()
    estimation_error_per_trial = dict()
    for param in parameter_names:
        estimation_error_per_trial[param] = dict()
    for batchsize in batchsizes:            
        estimation_error_per_trial[param][batchsize] = []
        for m in range(M):            
            data_location = "{}/{}/{}/".format(data_topdir, m, batchsize)
            params_out, _ = combine_estimate_variance_rule(data_location, J, K, d, parameter_names, 
                                                            None, theta_true, param_positions_dict)    
            for param in parameter_names:
                if param == "X":                
                    params_out_jsonl[param] = params_out[param].reshape((d*K,), order="F").tolist()     
                    X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                    X_hat = np.asarray(params_out_jsonl[param]).reshape((d, K), order="F")
                    Rx, tx, mse_x, mse_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat)
                    estimation_error_per_trial[param][batchsize].append(float(mse_x))
                elif param == "Z":
                    params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                    Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                    Z_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                    Rz, tz, mse_z, mse_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat)
                    estimation_error_per_trial[param][batchsize].append(float(mse_z))              
                elif param == "Phi":            
                    params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                    Phi_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                    Phi_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                    Rphi, tphi, mse_phi, mse_phi_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Phi_true, param_hat=Phi_hat)
                    estimation_error_per_trial[param][batchsize].append(float(mse_phi))
                elif param in ["beta", "alpha"]:
                    params_out_jsonl[param] = params_out[param].tolist()     
                    mse = np.sum(((theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]])**2)/len(params_out_jsonl[param])
                    estimation_error_per_trial[param][batchsize].append(float(mse))
                else:
                    params_out_jsonl[param] = params_out[param]
                    mse = ((theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]])**2
                    estimation_error_per_trial[param][batchsize].append(float(mse[0])) 
                                  
        out_file = "{}/params_out_global_theta_hat.jsonl".format(data_location)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write(params_out_jsonl)
    
    # box plot - mse relativised per parameter over trials    
    for param in parameter_names:
        fig = go.Figure()
        for batchsize in batchsizes:
            fig.add_trace(go.Box(
                            y=np.asarray(estimation_error_per_trial[param][batchsize]).tolist(), showlegend=True, name=param,
                            x=[batchsize]*len(estimation_error_per_trial[param][batchsize]), boxpoints='outliers'                                
                        ))
        savename = "{}/mle_estimation_plots/mse_overAllTrials_{}_weighted_boxplot.html".format(data_topdir, param)
        fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="MSE Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, 
                                print_pdf=False)

####################### MLE #############################



####################### ICM #############################

def create_constraint_functions_icm(n, vector_coordinate=None, param=None, param_positions_dict=None, args=None):
    
    bounds = []
    grid_width_std = 5
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    if param == "alpha":
        bounds.append((-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha))
    elif param == "beta":
        bounds.append((-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta))
    elif param == "gamma":            
        bounds.append((None, None))  ########## MODIFY TO SET NON_ZERO CONSTRAINT?
    elif param == "delta":                    
        bounds.append((None, None))  ########## MODIFY TO SET NON_ZERO CONSTRAINT?        
    elif param == "sigma_e":
        bounds.append((min_sigma_e, grid_width_std*np.sqrt(prior_scale_sigmae)))
    else:
        if d == 1 or elementwise:
            if param == "Phi":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate]))
            elif param == "Z":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate]))
            elif param == "X":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate]))        
        else:
            raise NotImplementedError("At the moment, we have not implemented diffentiation for multivariate parameter vector.")
    
    return bounds


def log_complement_from_log_cdf_vec(log_cdfx, x, mean, variance, use_jax=False):
    """
    Computes log(1-CDF(x)) given log(CDF(x)) in a numerically stable way.
    """    
    if use_jax:
        if len(log_cdfx.shape)==0 or (isinstance(log_cdfx, jnp.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1): 
            if log_cdfx < -0.693:  #log(0.5)
                # If CDF(x) < 0.5, direct computation is stable  
                ret = jnp.log1p(-jnp.exp(log_cdfx))
            else: 
                # If CDF(x) ≥ 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
                ret = jax.scipy.stats.norm.logcdf(-x, loc=mean, scale=variance)       
        else:
            ret = jnp.zeros(log_cdfx.shape)    
            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case1 = jnp.argwhere(log_cdfx < -0.693)
                if idx_case1.size > 0:
                    ret.at[log_cdfx < -0.693].set(jnp.log1p(-jnp.exp(log_cdfx[log_cdfx < -0.693])))                  
            else:
                idx_case1 = jnp.argwhere(log_cdfx < -0.693).flatten()       
                if idx_case1.size > 0:
                    ret.at[idx_case1].set(jnp.log1p(-np.exp(log_cdfx[idx_case1])))          

            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case2 = jnp.argwhere(log_cdfx >= -0.693)
                if idx_case2.size > 0:
                    ret.at[log_cdfx >= -0.693].set(jax.scipy.stats.norm.logcdf(-x[log_cdfx >= -0.693], loc=mean, scale=variance))
            else:
                idx_case2 = jnp.argwhere(log_cdfx >= -0.693).flatten()    
                if idx_case2.size > 0:                
                    ret.at[idx_case2].set(jax.scipy.stats.norm.logcdf(-x[idx_case2], loc=mean, scale=variance))       
    else:
        if isinstance(log_cdfx, float) or (isinstance(log_cdfx, np.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1): 
            if log_cdfx < -0.693:  #log(0.5)
                # If CDF(x) < 0.5, direct computation is stable  
                ret = np.log1p(-np.exp(log_cdfx))
            else: 
                # If CDF(x) ≥ 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
                ret = norm.logcdf(-x, loc=mean, scale=variance)       
        else:                          
            ret = np.zeros(log_cdfx.shape)                
            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case1 = np.argwhere(log_cdfx < -0.693)
                if idx_case1.size > 0:
                    ret[log_cdfx < -0.693] = np.log1p(-np.exp(log_cdfx[log_cdfx < -0.693]))                     
            else:
                idx_case1 = np.argwhere(log_cdfx < -0.693).flatten()       
                if idx_case1.size > 0:
                    ret[idx_case1] = np.log1p(-np.exp(log_cdfx[idx_case1]))             

            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case2 = np.argwhere(log_cdfx >= -0.693)
                if idx_case2.size > 0:
                    ret[log_cdfx >= -0.693] = norm.logcdf(-x[log_cdfx >= -0.693], loc=mean, scale=variance)
            else:
                idx_case2 = np.argwhere(log_cdfx >= -0.693).flatten()    
                if idx_case2.size > 0:                
                    ret[idx_case2] = norm.logcdf(-x[idx_case2], loc=mean, scale=variance)   
              
    return ret


def log_complement_from_log_cdf(log_cdfx, x, mean, variance, use_jax=False):
    """
    Computes log(1-CDF(x)) given log(CDF(x)) in a numerically stable way.
    """    
    def get_one_minus_logcdf(logcdfxx):
        logcdfx, xx = logcdfxx
        if logcdfx < -0.693:  #log(0.5)
            # If CDF(x) < 0.5, direct computation is stable  
            if use_jax:
                ret = jnp.log1p(-jnp.exp(logcdfx))
                # ret = jnp.asarray(list(ret))
            else:                
                ret = np.log1p(-np.exp(logcdfx))
                # ret = np.asarray(list(ret))                
        else: 
            # If CDF(x) ≥ 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
            if use_jax:
                ret = jax.scipy.stats.norm.logcdf(-xx, loc=mean, scale=variance)
            else:
                ret = norm.logcdf(-xx, loc=mean, scale=variance)             
        
        return ret

    if use_jax:
        if len(log_cdfx.shape)==0 or (isinstance(log_cdfx, jnp.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1):        
            return get_one_minus_logcdf((log_cdfx, x))
        else:                
            retvallist = list(map(get_one_minus_logcdf, zip(log_cdfx, x)))        
            return jnp.array(retvallist)
    else:
        if isinstance(log_cdfx, float) or (isinstance(log_cdfx, np.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1):        
            return get_one_minus_logcdf((log_cdfx, x))
        else:                
            retvallist = list(map(get_one_minus_logcdf, zip(log_cdfx, x)))        
            return np.array(retvallist)



def p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict, use_jax=False):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    if use_jax:
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
        if isinstance(i, int) and isinstance(j, int):            
            phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]            
        else:
            def pairwise_dst_fast(xi_betai):
                xi, betai = xi_betai            
                x_broadcast = xi[:, jnp.newaxis]
                diff_xz = x_broadcast - Z
                dst_xz = jnp.sum(diff_xz * diff_xz, axis=0)
                if "Phi" in params_hat.keys():
                    diff_xphi = x_broadcast - Phi
                    dst_xphi = jnp.sum(diff_xphi * diff_xphi, axis=1)
                else:
                    dst_xphi = 0
                phi = gamma*dst_xz - delta*dst_xphi + alpha + betai     
                return phi
            if isinstance(i, int) and j is None:        
                phi = pairwise_dst_fast((X[:, i], beta[i]))            
            elif i is None and j is None:                          
                arr_list = list(map(pairwise_dst_fast, zip(X.transpose(), beta)))
                phi = jnp.vstack(arr_list)
    else:
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

        if isinstance(i, int) and isinstance(j, int):
            phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]
        else:
            def pairwise_dst_fast(xi_betai):
                xi, betai = xi_betai  
                x_broadcast = xi[:, np.newaxis]
                diff_xz = x_broadcast - Z
                dst_xz = np.sum(diff_xz * diff_xz, axis=0)
                if "Phi" in params_hat.keys():
                    diff_xphi = x_broadcast - Phi
                    dst_xphi = np.sum(diff_xphi * diff_xphi, axis=1)
                else:
                    dst_xphi = 0
                phi = gamma*dst_xz - delta*dst_xphi + alpha + betai     
                return phi
            if isinstance(i, int) and j is None:        
                phi = pairwise_dst_fast((X[:, i], beta[i]))            
            elif i is None and j is None:                          
                arr_list = list(map(pairwise_dst_fast, zip(X.transpose(), beta)))
                phi = np.vstack(arr_list)
    
    return phi

def halve_annealing_rate_upd_schedule(N, gamma, delta_n, temperature_rate, temperature_steps, 
                                    all_gammas=None, testparam=None):
       
    tr_idx = temperature_rate.index(delta_n)            
    temperature_rate_upd = temperature_rate[:tr_idx]
    temperature_rate_upd.append(temperature_rate[tr_idx]/2)
    temperature_rate_upd.extend(temperature_rate[(tr_idx+1):])        
    all_gammas = []             
    gamma = 0.01        
    for gidx in range(len(temperature_steps[1:])):
        upperlim = temperature_steps[1+gidx]        
        start = gamma if gidx==0 else all_gammas[-1]        
        all_gammas.extend(np.arange(start, upperlim, temperature_rate_upd[gidx]))            
    N = len(all_gammas)
    print("Annealing schedule: {}".format(N))                
    gamma = all_gammas[0] 
    delta_n = temperature_rate_upd[0] 

    if testparam is not None:
        all_gammas = [1]*N
        gamma = 1

    return gamma, delta_n, temperature_rate_upd, all_gammas, N


def check_convergence(elementwise, theta_curr, theta_prev, param_positions_dict, iteration, parameter_space_dim, d=None, testparam=None, testidx=None, p=0.2, tol=1e-6):
    
    converged = False
    if testparam is None:
        # check full vector for convergence
        delta_theta_se = (theta_curr - theta_prev)**2
        if not np.allclose(np.sum(delta_theta_se), 1e-14):                                   
            delta_theta = delta_theta_se/np.sum(delta_theta_se)
        else:
            delta_theta = delta_theta_se
    else:
        if not elementwise and testidx is not None:
            # check vector portion
            delta_theta_se = (theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d] - theta_prev[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d])**2
            if not np.allclose(np.sum(delta_theta_se), 1e-14):                                   
                delta_theta = delta_theta_se/np.sum(delta_theta_se)
            else:
                delta_theta = delta_theta_se
        elif elementwise:
            delta_theta = np.abs(theta_curr[param_positions_dict[testparam][0]+testidx] - theta_prev[param_positions_dict[testparam][0]+testidx])
    if np.all(delta_theta <= tol):
        converged = True

    random_restart = False
    if testparam is None:
        # check all vector, after a full iteration - add check for iteration to make sure
        if iteration >= parameter_space_dim and (np.sum(delta_theta < tol) > int(p*len(delta_theta))):        
            random_restart = True
    else:                
        # check tested param only, ensure all coordinates have been updated once - delta_theta is already the sub-vector
        if (not elementwise) and (testidx is not None) and (iteration >= param_positions_dict[testparam][1]):
            if (np.sum(delta_theta < tol) > int(p*len(delta_theta))):        
                random_restart = True
        else:
            # no point to check when looking at specific coordinate
            random_restart = False
        
    return converged, delta_theta, random_restart


def rank_and_plot_solutions(estimated_thetas, elapsedtime, Y, J, K, d, parameter_names, dst_func, param_positions_dict, DIR_out, args):

    computed_loglik = []
    for theta in estimated_thetas:
        loglik = log_full_likelihood(Y, theta.copy(), param_positions_dict, args)
        computed_loglik.append(loglik[0])
    # sort in increasing order, i.e. from best to worst solution
    sorted_idx = np.argsort(np.asarray(computed_loglik))
    sorted_idx_lst = sorted_idx.tolist()    
    for i in sorted_idx_lst:
        theta = estimated_thetas[i]
        loglik = computed_loglik[i]
        params_out = dict()
        params_out["loglik"] = loglik
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
    computed_loglik = np.array(computed_loglik)[sorted_idx]
    if theta_matrix.shape[0] > 1:
        theta_matrix = theta_matrix[sorted_idx, :]
        pca = PCA(n_components=2)
        components = pca.fit_transform(theta_matrix)
        fig = go.Figure()
        for i in range(components.shape[0]):
            fig.add_trace(go.Scatter(x=[components[i, 0]], y=[components[i, 1]], marker_symbol=raw_symbols[i], text=computed_loglik[i]))
        fig.update(layout_yaxis_range = [np.min(components[:,1])-1,np.max(components[:,1])+1])
        fix_plot_layout_and_save(fig, "{}/solution_plots/project_solutions_2D.html".format(DIR_out, sorted_idx_lst.index(i)), 
                                    xaxis_title="PC1", yaxis_title="PC2", title="", 
                                    showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
        # fig.show()
 

def sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict, args, samples_list=None, idx_all=None):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    if samples_list is None and parameter_space_dim <= 21201:
        sampler = qmc.Sobol(d=parameter_space_dim, scramble=False)   
        samples_list = list(sampler.random_base2(m=base2exponent))       
    elif samples_list is None:
        if d > 2:
            raise NotImplementedError("In {}-dimensional space for the ideal points, find a way to generate random initial solutions.")
        sampler_alpha_sigma = qmc.Sobol(d=J+2, scramble=False)   
        samples_list_alpha_sigma = sampler_alpha_sigma.random_base2(m=base2exponent)
        samples_list = np.zeros((2**base2exponent, parameter_space_dim))
        param = "alpha"
        samples_list[:, param_positions_dict[param][0]:param_positions_dict[param][1]] = samples_list_alpha_sigma[:, :J]
        param = "gamma"
        samples_list[:, param_positions_dict[param][0]:param_positions_dict[param][1]] = \
                            samples_list_alpha_sigma[:, J].reshape(samples_list[:, param_positions_dict[param][0]:param_positions_dict[param][1]].shape)
        param = "sigma_e"
        samples_list[:, param_positions_dict[param][0]:param_positions_dict[param][1]] = \
                            samples_list_alpha_sigma[:, J+1].reshape(samples_list[:, param_positions_dict[param][0]:param_positions_dict[param][1]].shape)
        x = np.linspace(0, 1, math.ceil(np.sqrt((K+J)*d+K)))
        y = np.linspace(0, 1, math.ceil(np.sqrt((K+J)*d+K)))    
        grid_points = list(product(x, y))
        idxgrid = np.arange(0, len(grid_points), 1)
        for itmrp in range(2**base2exponent):
            samples_list[itmrp, :(K+J)*d] = np.asarray([grid_points[igp] for igp in np.random.choice(idxgrid, size=(K+J), 
                                                                            replace=True).tolist()]).reshape(samples_list[itmrp, :(K+J)*d].shape)
            param = "beta"
            samples_list[itmrp, param_positions_dict[param][0]:param_positions_dict[param][1]] = \
                np.asarray([grid_points[igp] for igp in np.random.choice(idxgrid, size=int(K/2), replace=True).tolist()]).reshape(samples_list[itmrp, 
                                                                                            param_positions_dict[param][0]:param_positions_dict[param][1]].shape)
        samples_list = list(samples_list)
        

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
        elif param == "sigma_e":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = min_sigma_e
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_sigmae) #+prior_loc_gamma
    
    theta_curr = qmc.scale(theta_curr, lbounds, ubounds).reshape((parameter_space_dim,))

    return theta_curr, samples_list, idx_all


def update_annealing_temperature(gamma_prev, total_iter, temperature_rate, temperature_steps, all_gammas=None):

    delta_n = None
    if all_gammas is None:        
        if (gamma_prev >= temperature_steps[0] and gamma_prev <= temperature_steps[1]):
            delta_n = temperature_rate[0]
        elif (gamma_prev > temperature_steps[1] and gamma_prev <= temperature_steps[2]):
            delta_n = temperature_rate[1]
        elif (gamma_prev > temperature_steps[2] and gamma_prev <= temperature_steps[3]):
            delta_n = temperature_rate[2]
        elif (gamma_prev > temperature_steps[3]):
            delta_n = temperature_rate[3]        
        gamma = gamma_prev + delta_n        
        # print("Delta_{} = {}".format(total_iter, delta_n))            
    elif isinstance(all_gammas, list):  
        idx = total_iter % len(all_gammas)
        gamma = all_gammas[idx]                                    
        if (gamma >= temperature_steps[0] and gamma <= temperature_steps[1]):
            tr_idx = 0            
        elif (gamma > temperature_steps[1] and gamma <= temperature_steps[2]):
            tr_idx = 1            
        elif (gamma > temperature_steps[2] and gamma <= temperature_steps[3]):
            tr_idx = 2            
        elif (gamma > temperature_steps[3]):
            tr_idx = 3     
        delta_n = temperature_rate[tr_idx]             

    return gamma, delta_n

def get_min_achievable_mse_under_rotation_trnsl(param_true, param_hat):

    """
    Minimizes ||Y - (XR + t)||_F where ||.||_F is the Frobenius norm and R a rotation matrix
    
    """
    if param_true.shape != param_hat.shape:
        raise ValueError("Input matrices X and Y must have the same shape")
    
    X_mean = np.mean(param_hat, axis=0)
    Y_mean = np.mean(param_true, axis=0)
    
    X_centered = param_hat - X_mean
    Y_centered = param_true - Y_mean
    
    # Compute the covariance
    H = X_centered.T @ Y_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Construct the rotation matrix
    # Handle reflection by ensuring proper rotation, i.e. det(R) = 1
    d = np.linalg.det(Vt.T @ U.T)
    M = np.eye(U.shape[1])
    M[-1, -1] = d
    R = Vt.T @ M @ U.T
    
    # Step 5: Compute the optimal translation
    t = Y_mean - X_mean @ R
    
    # Compute the residual error
    # error = np.linalg.norm(param_true - (param_hat @ R + t), 'fro')
    # error = np.sum((param_true - (param_hat @ R + t))**2)/(param_true.shape[0]*param_true.shape[1])
    # relative error
    error = np.sum(((param_true - (param_hat @ R + t))/param_true)**2)/(param_true.shape[0]*param_true.shape[1])
    # non-rotated/non-translated relative error
    error_nonRT = np.sum(((param_true - param_hat)/param_true)**2)/(param_true.shape[0]*param_true.shape[1])
    
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(R.shape[0]))    
    det_is_one = np.abs(np.linalg.det(R) - 1.0) < 1e-10    
    t_shape_correct = t.shape == (param_hat.shape[1],)
    if not (orthogonality_error < 1e-10 and det_is_one and t_shape_correct):
        raise AttributeError("Error in solving projection problem?")
    
    return R, t, error, error_nonRT


    
def compute_and_plot_mse(theta_true, theta_hat, fullscan, iteration, args, param_positions_dict,
                        plot_online=True, mse_theta_full=[], fig_xz=None, mse_x_list=[], mse_z_list=[],
                        mse_x_nonRT_list=[], mse_z_nonRT_list=[], per_param_ers=dict(), 
                        per_param_heats=dict(), xbox=[], plot_restarts=[]):


    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true = args
  
    # compute with full theta vector - relative
    sse = ((theta_true - theta_hat)/theta_true)**2
    rel_se = sse/len(sse)
    per_param_heats["theta"].append(rel_se)
    # total relative error
    mse_theta_full.append(float(np.sum(rel_se)))
    if plot_online:        
        fig = go.Figure(data=go.Heatmap(z=per_param_heats["theta"], colorscale = 'Viridis'))
        savename = "{}/theta_heatmap/theta_full_relativised_squarederror.html".format(DIR_out)
        pathlib.Path("{}/theta_heatmap/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig, savename, xaxis_title="Coordinate", yaxis_title="Iteration", title="", 
                                showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                print_pdf=False)        

    # compute min achievable mse for X, Z under rotation and scaling
    params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
    X_true = np.asarray(params_true["X"]).reshape((d, K), order="F")       
    Z_true = np.asarray(params_true["Z"]).reshape((d, J), order="F")                         
    params_hat = optimisation_dict2params(theta_hat, param_positions_dict, J, K, d, parameter_names)

    for param in parameter_names:        
        if param in ["gamma", "delta", "sigma_e"]:
            rel_se = np.sum((((params_true[param] - params_hat[param])/params_true[param])**2))/len(params_true[param])
            # time series plots
            per_param_ers[param].append(float(rel_se))
            if plot_online:
                fig = make_subplots(specs=[[{"secondary_y": True}]])   
                fig.add_trace(go.Scatter(
                                        y=per_param_ers[param], showlegend=False,
                                        x=np.arange(iteration)                                    
                                    ), secondary_y=False)
                fig.add_trace(go.Scatter(
                                        y=mse_theta_full, showlegend=True,
                                        x=np.arange(iteration), line_color="red", name="Θ MSE"                                
                                    ), secondary_y=True)
                for itm in plot_restarts:
                    scanrep, totaliterations, halvedgammas, restarted = itm
                    if halvedgammas:
                        vcolor = "red"
                    else:
                        vcolor = "green"
                    if restarted=="fullrestart":
                        fig.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                    else:
                        # partial restart
                        fig.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                savename = "{}/timeseries_plots/{}_squarederror.html".format(DIR_out, param)
                pathlib.Path("{}/timeseries_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                fix_plot_layout_and_save(fig, savename, xaxis_title="Annealing iterations", yaxis_title="Squared error", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                        print_pdf=False)
        else:
            if param == "X":
                X_hat = np.asarray(params_hat[param]).reshape((d, K), order="F")       
                X_hat_vec = np.asarray(params_hat[param]).reshape((d*K,), order="F")       
                X_true_vec = np.asarray(params_true[param]).reshape((d*K,), order="F")       
                se = (((X_true_vec - X_hat_vec)/X_true_vec)**2)/len(X_true_vec)                
            elif param == "Z":
                Z_hat = np.asarray(params_hat[param]).reshape((d, J), order="F")          
                Z_hat_vec = np.asarray(params_hat[param]).reshape((d*J,), order="F")         
                Z_true_vec = np.asarray(params_true[param]).reshape((d*J,), order="F")       
                se = (((Z_true_vec - Z_hat_vec)/Z_true_vec)**2)/len(Z_true_vec) 
            else:
                se = (((params_true[param] - params_hat[param])/params_true[param])**2)/len(params_true[param])
            
            per_param_heats[param].append(se)   
            rel_se = float(np.sum(se))            
            if plot_online:  
                fig = go.Figure(data=go.Heatmap(z=per_param_heats[param], colorscale = 'Viridis'))
                savename = "{}/params_heatmap/{}_relativised_squarederror.html".format(DIR_out, param)
                pathlib.Path("{}/params_heatmap/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                fix_plot_layout_and_save(fig, savename, xaxis_title="Coordinate", yaxis_title="Iteration", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                        print_pdf=False)            
                parray = np.stack(per_param_heats[param])
                # timeseries plots
                for pidx in range(parray.shape[1]):
                    fig = make_subplots(specs=[[{"secondary_y": True}]]) 
                    fig.add_trace(go.Scatter(
                                    y=parray[:, pidx], showlegend=False,
                                    x=np.arange(iteration)                                    
                                ), secondary_y=False)
                    fig.add_trace(go.Scatter(
                                    y=mse_theta_full, showlegend=True,
                                    x=np.arange(iteration), line_color="red", name="Θ MSE"                                
                                ), secondary_y=True)
                    for itm in plot_restarts:
                        scanrep, totaliterations, halvedgammas, restarted = itm
                        if halvedgammas:
                            vcolor = "red"
                        else:
                            vcolor = "green"
                        if restarted=="fullrestart":
                            fig.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                        label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                        font=dict(size=16, family="Times New Roman"),),)
                        else:
                            # partial restart
                            fig.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                        label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                        font=dict(size=16, family="Times New Roman"),),)
                    savename = "{}/timeseries_plots/{}_idx_{}_relativised_squarederror.html".format(DIR_out, param, pidx)
                    pathlib.Path("{}/timeseries_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                    fix_plot_layout_and_save(fig, savename, xaxis_title="Annealing iterations", yaxis_title="Relative squared error", title="", 
                                            showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                            print_pdf=False)     

    if fig_xz is None:
        fig_xz = go.Figure()  
    # mean error over all elements of the matrices  
    Rx, tx, mse_x, mse_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat)
    mse_x_list.append(mse_x)
    mse_x_nonRT_list.append(mse_x_nonRT)
    Rz, tz, mse_z, mse_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat)
    mse_z_list.append(mse_z)   
    mse_z_nonRT_list.append(mse_z_nonRT)
    # following does not reset when a new full scan starts
    per_param_ers["X_rot_translated_mseOverMatrix"].append(mse_x)
    per_param_ers["Z_rot_translated_mseOverMatrix"].append(mse_z)
    per_param_ers["X_mseOverMatrix"].append(mse_x_nonRT)
    per_param_ers["Z_mseOverMatrix"].append(mse_z_nonRT)
    xbox.append(fullscan)
    if plot_online:
        fig_xz.add_trace(go.Box(
                            y=np.asarray(mse_x_list).tolist(), 
                            x=xbox,
                            name="X - total iter. {}".format(iteration),
                            boxpoints='outliers', line=dict(color="blue")
                            ))
        fig_xz.add_trace(go.Box(
                            y=np.asarray(mse_x_nonRT_list).tolist(), 
                            x=xbox, opacity=0.5,
                            name="X (nonRT) - total iter. {}".format(iteration),
                            boxpoints='outliers', line=dict(color="blue")
                            ))
        fig_xz.add_trace(go.Box(
                            y=np.asarray(mse_z_list).tolist(), 
                            x=xbox,
                            name="Z - total iter. {}".format(iteration),
                            boxpoints='outliers', line=dict(color="green")
                            ))
        fig_xz.add_trace(go.Box(
                            y=np.asarray(mse_z_nonRT_list).tolist(), 
                            x=xbox, opacity=0.5,
                            name="Z (nonRT) - total iter. {}".format(iteration),
                            boxpoints='outliers', line=dict(color="green")
                            ))
        fig_xz.update_layout(boxmode="group")
        savename = "{}/xz_boxplots/relative_mse.html".format(DIR_out)
        pathlib.Path("{}/xz_boxplots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig_xz, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=True,
                            print_png=True, print_html=True, print_pdf=False)
        figX = make_subplots(specs=[[{"secondary_y": True}]]) 
        figX.add_trace(go.Scatter(
                                y=per_param_ers["X_rot_translated_mseOverMatrix"], 
                                x=np.arange(iteration),
                                name="X - min MSE<br>(under rot/transl)"
                            ), secondary_y=False)
        figX.add_trace(go.Scatter(
                                y=mse_theta_full, 
                                x=np.arange(iteration), line_color="red", name="Θ MSE"                                
                            ), secondary_y=True)
        
        figZ = make_subplots(specs=[[{"secondary_y": True}]]) 
        figZ.add_trace(go.Scatter(
                                y=per_param_ers["Z_rot_translated_mseOverMatrix"], 
                                x=np.arange(iteration),
                                name="Z - min MSE<br>(under rot/trl)"
                            ), secondary_y=False)
        figZ.add_trace(go.Scatter(
                                y=mse_theta_full, 
                                x=np.arange(iteration), line_color="red", name="Θ MSE"                                
                            ), secondary_y=True)
        for itm in plot_restarts:
            scanrep, totaliterations, halvedgammas, restarted = itm
            if halvedgammas:
                vcolor = "red"
            else:
                vcolor = "green"
            if restarted=="fullrestart":
                figX.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                            label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                            font=dict(size=16, family="Times New Roman"),),)
                figZ.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                            label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                            font=dict(size=16, family="Times New Roman"),),)
            else:
                # partial restart
                figX.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                            label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                            font=dict(size=16, family="Times New Roman"),),)
                figZ.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                            label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                            font=dict(size=16, family="Times New Roman"),),)
        savenameX = "{}/timeseries_plots/X_rot_translated_relative_mse.html".format(DIR_out)
        pathlib.Path("{}/timeseries_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(figX, savenameX, xaxis_title="", yaxis_title="MSE", title="", 
                                showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                print_pdf=False)
        savenameZ = "{}/timeseries_plots/Z_rot_translated_relative_mse.html".format(DIR_out)
        pathlib.Path("{}/timeseries_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(figZ, savenameZ, xaxis_title="", yaxis_title="MSE", title="", 
                                showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                print_pdf=False)

    return mse_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, fig_xz, per_param_ers, per_param_heats, xbox


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


def get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta, gamma, param_positions_dict, args):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

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
    elif param == "sigma_e":
        post2optim = lambda x: log_conditional_posterior_sigma_e(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma, prior_loc_sigmae, prior_scale_sigmae, min_sigma_e) 
        
    return post2optim


def get_evaluation_grid(param, vector_coordinate, args, gridpoints_num_plot=None):

    grid_width_std = 5    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    if gridpoints_num_plot is not None:
        gridpoints_num = gridpoints_num_plot
    
    gridpoints_num_alpha_beta = gridpoints_num**2
    xx_ = None

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
    elif param == "sigma_e":
        grid = np.linspace(min_sigma_e, grid_width_std*np.sqrt(prior_scale_sigmae)+prior_loc_sigmae, gridpoints_num_alpha_beta).tolist()
    else:           
        if d == 1 or elementwise and vector_coordinate is not None:
            if param == "Phi":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], gridpoints_num).tolist()                
            elif param == "Z":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], gridpoints_num).tolist()                
            elif param == "X":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], gridpoints_num).tolist()                            
        elif (d > 1 and d <= 5) and not elementwise:
            if param == "Phi":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
                xx_ = np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], gridpoints_num).tolist()
            elif param == "Z":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
                xx_ = np.linspace(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], gridpoints_num).tolist()
            elif param == "X":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
                xx_ = np.linspace(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], gridpoints_num).tolist()
        else:
            raise NotImplementedError("Use a Sobol sequence to generate a grid in such high dimensional space.")
    if xx_ is None:
        xx_ = grid.copy()

    return grid, xx_

def plot_posterior_elementwise(outdir, param, Y, idx, vector_coordinate, theta_curr, gamma, param_positions_dict, args, 
                            true_param=None, hat_param=None, iteration=None, fig_in=None, plot_arrows=False, all_theta=None):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    f = get_posterior_for_optimisation_vec(param=param, Y=Y, idx=idx, vector_index_in_param_matrix=idx, vector_coordinate=vector_coordinate, theta=theta_true.copy(), 
                                           gamma=gamma, param_positions_dict=param_positions_dict, args=args)
    
    gridpoints_num_plot = 20

    xx, xx_ = get_evaluation_grid(param, vector_coordinate, args, gridpoints_num_plot=gridpoints_num_plot)      
    if vector_coordinate is None:        
        xxlist = [ix for ix in xx]                
    else:        
        xxlist = xx_    
    yy = np.asarray(list(map(f, xxlist)))

    # plot : all_theta[param][idx] for X, Phi, Z or all_theta[param][vector_coordinate] for alpha, beta, or all_theta[param] for scalars
    if isinstance(idx, int) and vector_coordinate is None:
        fig = go.Figure()
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])   
    # density plot
    if vector_coordinate is not None:
        fig.add_trace(go.Scatter(
                        x=xx_,
                        y=yy,
                        mode='lines',
                        name="{}: cond. posterior".format(param),
                        line=dict(
                            color='royalblue',
                            width=2
                        ),
                        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ), secondary_y=False,)
    else:        
        fig.add_trace(go.Contour(
                        x=xx_,
                        y=xx_,
                        z=yy,
                        colorscale='Hot',
                        contours=dict(showlabels=True),
                        colorbar=dict(
                            title='Cond. posterior',
                            titleside='right'
                        )
                    )
                )
        fig.update_layout(xaxis_title='x1', yaxis_title='x2')

    if (idx is None and vector_coordinate==0 and true_param is not None) or (isinstance(idx, int) and isinstance(vector_coordinate, int)):
        # scalar param        
        fig.add_vline(x=true_param, line_width=3, line_dash="dash", line_color="green", name="True θ", showlegend=True)
    elif isinstance(idx, int) and vector_coordinate is None:
        # surface plot
        fig.add_trace(go.Scatter(x=[true_param[0]], y=[true_param[1]], name="True θ", mode="markers", marker_symbol="star", marker_color="green"))

    # if only plotting density
    if all_theta is None:
        fig.update_layout(hovermode='x unified', legend=dict(orientation="h"))         
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)    
        savename = "{}/{}_idx_{}_vector_coord_{}.html".format(outdir, param, idx, vector_coordinate)
        if param in ["X", "Z", "Phi"] and vector_coordinate is None:
            yaxistitle = ""
        else:
            yaxistitle = "Conditional posterior"
        fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title=yaxistitle, title="", 
                                showgrid=False, showlegend=True, print_png=True, print_html=False, print_pdf=False)
        return fig

    if param in ["X", "Z", "Phi"] and vector_coordinate is None:
        # 2d plot
        if d != 2:
            raise NotImplementedError("Only implemented for 2D ideal points space.")
        try:
            y1 = np.asarray([itm[2] for itm in all_theta[param][idx][0]])
            y2 = np.asarray([itm[2] for itm in all_theta[param][idx][1]])
        except:
            y1 = []
            y2 = []
            print("Not enough data for both coordinates yet.")            
            pass
        if len(y1) > 0 and len(y2) > 0:
            if len(y1) > len(y2):
                all_theta[param][idx][1].append(all_theta[param][idx][1][-1])
                y2 = np.asarray([itm[2] for itm in all_theta[param][idx][1]])
            elif len(y2) > len(y1):
                all_theta[param][idx][0].append(all_theta[param][idx][0][-1])
                y1 = np.asarray([itm[2] for itm in all_theta[param][idx][0]])
            opacitylevels = np.linspace(0.1, 1, len(y1))
            text_i = np.asarray(["i: {}, total i: {}, γ = {:.3f}".format(itm[0], itm[1], gamma) for itm in all_theta[param][idx][0]])
            for iii in range(len(y1)):
                fig.add_trace(
                    go.Scatter(
                        x=[y1[iii]],
                        y=[y2[iii]],
                        mode="markers", marker_symbol="square", marker_color="blue",            
                        marker_size=8,
                        showlegend=False,
                        text=text_i,
                        hoverinfo="text"
                    )
                )   
            for iii in range(len(y1)-1):            
                fig.add_annotation(
                    x=y1[iii+1],
                    y=y2[iii+1],
                    ax=y1[iii],
                    ay=y2[iii],
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    opacity=opacitylevels[iii], 
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='blue'
                )        
    elif param in ["X", "Z", "Phi"] and (isinstance(idx, int) and isinstance(vector_coordinate, int)):
        # per coord plot of vector param
        opacitylevels = np.linspace(0.1, 1, len(all_theta[param][idx][vector_coordinate]))
        opc = 0
        for itm in all_theta[param][idx][vector_coordinate]:
            fig.add_vline(x=itm[2], opacity=opacitylevels[opc], line_width=2, line_dash="dash", line_color="red", showlegend=False) 
                        # , label=dict(text="{}, γ = {:.3f}".format(itm[1], gamma), textposition="top left",
                        # font=dict(size=16, family="Times New Roman"),),)
            fig.add_trace(go.Scatter(x=[itm[2]], y=[itm[3][0]], text="{}, γ = {}".format(itm[1], gamma), showlegend=False, 
                    mode="markers+lines", marker_symbol="square", marker_color="red", 
                    name="step: {}".format(itm[1])), secondary_y=True,)
            opc += 1
        x = np.asarray([itm[2] for itm in all_theta[param][idx][vector_coordinate]])
        y = np.asarray([itm[3][0] for itm in all_theta[param][idx][vector_coordinate]])
        for iii in range(len(x)-1):        
            fig.add_annotation(
                x=x[iii+1],
                y=y[iii+1],
                ax=x[iii],
                ay=y[iii],
                xref='x',
                yref='y2',
                axref='x',
                ayref='y2',
                opacity=opacitylevels[iii],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )       
    elif param in ["alpha", "beta"]:
        # per coord plot of vector param
        opacitylevels = np.linspace(0.1, 1, len(all_theta[param][vector_coordinate]))
        opc = 0
        for itm in all_theta[param][vector_coordinate]:
            fig.add_vline(x=itm[2], opacity=opacitylevels[opc], line_width=2, line_dash="dash", line_color="red", showlegend=False) 
                        # , label=dict(text="{}, γ = {:.3f}".format(itm[1], gamma), textposition="top left",
                        # font=dict(size=16, family="Times New Roman"),),)
            fig.add_trace(go.Scatter(x=[itm[2]], y=[itm[3][0]], text="{}, γ = {}".format(itm[1], gamma),
                    mode="markers+lines", marker_symbol="square", marker_color="red", showlegend=False, 
                    name="step: {}".format(itm[1])), secondary_y=True,)
            opc += 1
        x = np.asarray([itm[2] for itm in all_theta[param][vector_coordinate]])
        y = np.asarray([itm[3][0] for itm in all_theta[param][vector_coordinate]])
        for iii in range(len(x)-1):       
            fig.add_annotation(
                x=x[iii+1],
                y=y[iii+1],
                ax=x[iii],
                ay=y[iii],
                xref='x',
                yref='y2',
                axref='x',
                ayref='y2',
                opacity=opacitylevels[iii],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )        
    else:
        # scalar plot
        opacitylevels = np.linspace(0.2, 1, len(all_theta[param]))
        opc = 0
        for itm in all_theta[param]:
            fig.add_vline(x=itm[2], opacity=opacitylevels[opc], line_width=2, line_dash="dash", line_color="red", showlegend=False), 
                        # , label=dict(text="{}, γ = {:.3f}".format(itm[1], gamma), textposition="top left",
                        # font=dict(size=16, family="Times New Roman"),),)
            fig.add_trace(go.Scatter(x=[itm[2]], y=[itm[3][0]], text="{}, γ = {}".format(itm[1], gamma),
                    mode="markers+lines", marker_symbol="square", marker_color="red", showlegend=False,
                    name="step: {}".format(itm[1])), secondary_y=True,)
            opc += 1
        x = np.asarray([itm[2] for itm in all_theta[param]])
        y = np.asarray([itm[3][0] for itm in all_theta[param]])
        for iii in range(len(x)-1):    
            fig.add_annotation(
                x=x[iii+1],
                y=y[iii+1],
                ax=x[iii],
                ay=y[iii],
                xref='x',
                yref='y2',
                axref='x',
                ayref='y2',
                opacity=opacitylevels[iii],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )       
    
    fig.update_layout(hovermode='x unified', legend=dict(orientation="h")) 
    if not (isinstance(idx, int) and vector_coordinate is None):
        fig.update_layout(yaxis2=dict(title="Data loglikelihood")) 
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)    
    savename = "{}/{}_idx_{}_vector_coord_{}.html".format(outdir, param, idx, vector_coordinate)
    if param in ["X", "Z", "Phi"] and vector_coordinate is None:
        yaxistitle = ""
    else:
        yaxistitle = "Conditional posterior"
    fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title=yaxistitle, title="", 
                            showgrid=False, showlegend=True, print_png=True, print_html=False, print_pdf=False)

    return fig

def plot_posteriors_during_estimation(Y, iteration, plotting_thetas, theta_curr, vect_iter, fig_posteriors, 
                                    fig_posteriors_annealed, gamma, param_positions_dict, args, plot_arrows=False, testparam=None, testidx=None, testvec=None):
    
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    data_loglik = log_full_likelihood(Y, theta_curr.copy(), param_positions_dict, args)

    # non-annealed posterior
    for theta_i in range(parameter_space_dim):       
        
        if testparam is not None and elementwise and testidx is not None and param_positions_dict[testparam][0] + testidx != theta_i:
            continue

        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=theta_i, d=d)   

        if testparam is not None and testparam != target_param:
            continue

        if testparam is not None and not elementwise and testvec != vector_index_in_param_matrix:
            continue

        if target_param in ["gamma", "delta", "sigma_e"]:
            if len(plotting_thetas[target_param]) == 0:                
                plotting_thetas[target_param].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
            elif len(plotting_thetas[target_param]) >= 1 and plotting_thetas[target_param][-1][2] != theta_curr[theta_i]:
                # plot if parameter value has moved
                plotting_thetas[target_param].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
        else:            
            if isinstance(plotting_thetas[target_param], list) and len(plotting_thetas[target_param]) == 0:                
                plotting_thetas[target_param] = dict()
                if target_param in ["X", "beta"]:                    
                    for kk in range(K):
                        if target_param == "X":
                            plotting_thetas[target_param][kk] = dict()
                            for dd in range(d):
                                plotting_thetas[target_param][kk][dd] = []                            
                        else:
                            plotting_thetas[target_param][kk] = []                            
                elif target_param in ["Phi", "Z", "alpha"]:
                    for jj in range(J):
                        if target_param in ["Z", "Phi"]:
                            plotting_thetas[target_param][jj] = dict()
                            for dd in range(d):
                                plotting_thetas[target_param][jj][dd] = []                              
                        else:
                            plotting_thetas[target_param][jj] = []                                  
                if target_param in ["X", "Z", "Phi"]:
                    plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
                else:
                    plotting_thetas[target_param][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
            else:
                if target_param in ["X", "Z", "Phi"]:                    
                    if len(plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate]) == 0:                        
                        plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
                    elif len(plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate]) >= 1 and \
                        plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate][-1][2] != theta_curr[theta_i]:
                        # plot if parameter value has moved
                        plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
                else:
                    if len(plotting_thetas[target_param][vector_coordinate]) == 0:                        
                        plotting_thetas[target_param][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))
                    elif len(plotting_thetas[target_param][vector_coordinate]) >= 1 and \
                        plotting_thetas[target_param][vector_coordinate][-1][2] != theta_curr[theta_i]:
                        # plot if parameter value has moved
                        plotting_thetas[target_param][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_loglik))

        if elementwise or (not elementwise and target_param in ["alpha", "beta", "gamma", "delta", "sigma_e"]):                                         
            fig_posteriors[target_param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=target_param, 
                        Y=Y, idx=vector_index_in_param_matrix, vector_coordinate=vector_coordinate, 
                        theta_curr=theta_curr.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[theta_i], 
                        hat_param=theta_curr[theta_i], iteration=iteration,
                        fig_in=fig_posteriors[target_param], plot_arrows=plot_arrows, all_theta=plotting_thetas)     
    # vector 2D plots
    if not elementwise:
        for param in parameter_names:
            if testparam is not None and (testparam != param):
                continue
            if param in ["X"]:
                for i in range(K):         
                    if testparam is not None and testvec != i:         
                        continue
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=i, vector_coordinate=None, 
                        theta_curr=theta_curr.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d].copy(), 
                        hat_param=theta_curr[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d].copy(), iteration=iteration,
                        fig_in=fig_posteriors[param], plot_arrows=plot_arrows, all_theta=plotting_thetas)                    
            elif param in ["Z", "Phi"]:
                for j in range(J):                                    
                    if testparam is not None and testvec != j:         
                        continue
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=j, vector_coordinate=None, 
                        theta_curr=theta_curr.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d].copy(), 
                        hat_param=theta_curr[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d].copy(), iteration=iteration, 
                        fig_in=fig_posteriors[param], plot_arrows=plot_arrows, all_theta=plotting_thetas)    
        
    if gamma != 1:
        # annealed posterior
        for theta_i in range(parameter_space_dim):     
            if testidx is not None and testparam is not None and param_positions_dict[testparam][0] + testidx != theta_i:
                continue  
            target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=theta_i, d=d)     
            if testparam is not None and testparam != target_param:
                continue
            if testparam is not None and not elementwise and testvec != vector_index_in_param_matrix:
                continue
            keyname = "param_{}_vindexParammatrix_{}_veccord_{}_gamma_{}".format(target_param, vector_index_in_param_matrix, vector_coordinate, gamma)   
            if keyname in fig_posteriors_annealed.keys():
                fig = fig_posteriors_annealed[keyname]             
            else:
                fig = go.Figure()   
            if elementwise or (not elementwise and target_param in ["alpha", "beta", "gamma", "delta", "sigma_e"]):     
                fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=target_param, 
                            Y=Y, idx=vector_index_in_param_matrix, vector_coordinate=vector_coordinate, 
                            theta_curr=theta_curr.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                            true_param=theta_true[theta_i], 
                            hat_param=theta_curr[theta_i], iteration=iteration,
                            fig_in=fig, plot_arrows=plot_arrows, all_theta=plotting_thetas)     
        # vector 2D plots
        if not elementwise:
            for param in parameter_names:
                if testparam is not None and (testparam != param): # or testidx is not None):
                    continue
                keyname = "param_{}_vindexParammatrix_{}_gamma_{}".format(param, vector_index_in_param_matrix, gamma)   
                if keyname in fig_posteriors_annealed.keys():
                    fig = fig_posteriors_annealed[keyname]             
                else:
                    fig = go.Figure()
                if param in ["X"]:
                    for i in range(K):    
                        if testparam is not None and testvec != i:         
                            continue                               
                        fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, idx=i, vector_coordinate=None, 
                            theta_curr=theta_curr.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                            true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], 
                            hat_param=theta_curr[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], iteration=iteration,
                            fig_in=fig, plot_arrows=plot_arrows, all_theta=plotting_thetas)                
                elif param in ["Z", "Phi"]:
                    for j in range(J):   
                        if testparam is not None and testvec != j:         
                            continue
                        fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}//".format(DIR_out, gamma), param=param, Y=Y, idx=j, vector_coordinate=None, 
                            theta_curr=theta_curr.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                            true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], 
                            hat_param=theta_curr[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], iteration=iteration, 
                            fig_in=fig, plot_arrows=plot_arrows, all_theta=plotting_thetas)               
                
    return fig_posteriors, fig_posteriors_annealed, plotting_thetas




class TruncatedInverseGamma:
    """
    Implementation of a truncated Inverse Gamma distribution.
    The distribution is truncated to the interval [lower, upper].
    
    Parameters:
    -----------
    alpha : float
        Shape parameter of the inverse gamma distribution
    beta : float
        Scale parameter of the inverse gamma distribution
    lower : float
        Lower bound of the truncation interval
    upper : float
        Upper bound of the truncation interval
    """
    
    def __init__(self, alpha, beta, lower, upper):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if lower >= upper:
            raise ValueError("lower bound must be less than upper bound")
        if lower <= 0:
            raise ValueError("lower bound must be positive")
            
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        
        # Calculate normalization constant
        self.norm_constant = self._get_normalization_constant()
        self.log_norm_constant = np.log(self.norm_constant)
        
    def _get_normalization_constant(self):
        """Calculate the normalization constant for truncation."""
        # Integrate the inverse gamma from lower to upper
        lower_cdf = invgamma.cdf(self.lower, a=self.alpha, loc=0, scale=self.beta)
        upper_cdf = invgamma.cdf(self.upper, a=self.alpha, loc=0, scale=self.beta)
        return upper_cdf - lower_cdf
    
    def pdf(self, x):
        """
        Probability density function of the truncated inverse gamma.
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the PDF
            
        Returns:
        --------
        array_like
            PDF values at x
        """
        x = np.asarray(x)
        # Return 0 for values outside the truncation interval
        pdf = np.zeros_like(x, dtype=float)
        mask = (x >= self.lower) & (x <= self.upper)
        
        # Calculate PDF for values within bounds
        pdf[mask] = invgamma.pdf(x[mask], a=self.alpha, loc=0, scale=self.beta) / self.norm_constant
        return pdf

    def logcdf(self, x):
        """
        Log of the cumulative distribution function of the truncated inverse gamma.
        Computed in a numerically stable way.
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the log CDF
            
        Returns:
        --------
        array_like
            Log CDF values at x
        """
        x = np.asarray(x)
        logcdf = np.full_like(x, -np.inf, dtype=float)
        
        # For x < lower, logcdf is -inf
        # For x > upper, logcdf is 0
        upper_mask = x > self.upper
        logcdf[upper_mask] = 0.0
        
        # For values in truncation interval
        mask = (x >= self.lower) & (x <= self.upper)
        if np.any(mask):
            # Compute unnormalized CDF using the complementary incomplete gamma function
            # P(X ≤ x) = 1 - P(X > x) = 1 - gammaincc(alpha, beta/x)
            z = self.beta / x[mask]
            unnorm_cdf = -np.log1p(-gammaincc(self.alpha, z))
            
            # Normalize by the truncation interval
            lower_cdf = invgamma.cdf(self.lower, a=self.alpha, loc=0, scale=self.beta)
            logcdf[mask] = unnorm_cdf - np.log(self.norm_constant)
        
        return logcdf
    
    def rvs(self, size=1, random_state=None):
        """
        Random variates from the truncated inverse gamma distribution.
        
        Parameters:
        -----------
        size : int
            Number of random variates to generate
        random_state : int or RandomState, optional
            Random state for reproducibility
            
        Returns:
        --------
        array_like
            Random variates from the truncated distribution
        """
        rng = np.random.RandomState(random_state)
        
        # Use rejection sampling
        samples = []
        while len(samples) < size:
            # Sample from the original inverse gamma
            candidate = invgamma.rvs(self.alpha, loc=0, scale=self.beta, 
                                   size=size, random_state=rng)
            
            # Accept samples that fall within bounds
            valid_samples = candidate[(candidate >= self.lower) & 
                                   (candidate <= self.upper)]
            samples.extend(valid_samples[:size - len(samples)])
            
        return np.array(samples[:size])
    
    def logpdf(self, x):
        """
        Log probability density function of the truncated inverse gamma.
        This method is numerically more stable than taking the log of pdf().
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the log PDF
            
        Returns:
        --------
        array_like
            Log PDF values at x
        """
        x = np.asarray(x)
        # Initialize with -inf for values outside the truncation interval
        logpdf = np.full_like(x, -np.inf, dtype=float)
        mask = (x >= self.lower) & (x <= self.upper)
        
        # Calculate logpdf for values within bounds
        # log(pdf) = log(beta^alpha) - log(Gamma(alpha)) - (alpha + 1)log(x) - beta/x - log(norm_constant)
        valid_x = x[mask]
        if len(valid_x) > 0:
            logpdf[mask] = (self.alpha * np.log(self.beta) - 
                            gammaln(self.alpha) -
                           (self.alpha + 1) * np.log(valid_x) -
                           self.beta / valid_x -
                           self.log_norm_constant)
        
        return logpdf

def log_full_likelihood(Y, theta_curr, param_positions_dict, args):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
    prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
    prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
    gridpoints_num, diff_iter, disp, min_sigma_e, theta_true = args

    ll_utility_part = -negative_loglik(theta_curr, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False)
    params_hat = optimisation_dict2params(theta_curr, param_positions_dict, J, K, d, parameter_names)

    loglik = ll_utility_part
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")
    loglik += multivariate_normal.logpdf(X.T, mean=prior_loc_x, cov=prior_scale_x).sum()
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F") 
    loglik += multivariate_normal.logpdf(Z.T, mean=prior_loc_z, cov=prior_scale_z).sum()
    if "Phi" in parameter_names:
        Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F") 
        loglik += multivariate_normal.logpdf(Phi.T, mean=prior_loc_phi, cov=prior_scale_phi).sum()
    alpha = params_hat["alpha"]
    loglik += norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha).sum()
    beta = params_hat["beta"]
    loglik += norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta).sum()
    gamma = params_hat["gamma"]
    loglik += norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma)
    if "delta" in parameter_names:
        delta = params_hat["delta"]
        loglik += norm.logpdf(delta, loc=prior_loc_delta, scale=prior_scale_delta)
    sigma_e = params_hat["sigma_e"]
    tig = TruncatedInverseGamma(alpha=prior_loc_sigmae, beta=prior_scale_sigmae, lower=min_sigma_e, upper=10*np.sqrt(prior_scale_sigmae)+prior_scale_sigmae) 
    loglik += tig.logpdf(sigma_e)

    return loglik

def log_conditional_posterior_x_vec(xi, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1, debug=False):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[:, i] = xi
    theta_test = theta.copy()
    theta_test[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpx_i = 0   
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                  
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)        
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)        
            _logpx_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(xi, mean=prior_loc_x, cov=prior_scale_x)
    
    pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpx_i = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(xi, mean=prior_loc_x, cov=prior_scale_x))
    if debug:
        assert(np.allclose(logpx_i, _logpx_i))
    
    return logpx_i*gamma

def log_conditional_posterior_x_il(x_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1, debug=False):
    # l denotes the coordinate of vector x_i

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[l, i] = x_il
    theta_test = theta.copy()
    theta_test[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpx_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpx_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(x_il, loc=prior_loc_x, scale=prior_scale_x)
    
    pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpx_il = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(x_il, mean=prior_loc_x, cov=prior_scale_x))
    if debug:
        assert(np.allclose(logpx_il, _logpx_il))
             
    return logpx_il*gamma


def log_conditional_posterior_phi_vec(phii, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi=0, prior_scale_phi=1, gamma=1, debug=False):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")                         
    Phi[:, i] = phii
    theta_test = theta.copy()
    theta_test[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpphi_i = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpphi_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(phii, mean=prior_loc_phi, cov=prior_scale_phi)

    pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpphi_i = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(phii, mean=prior_loc_phi, cov=prior_scale_phi))
    if debug:
        assert(np.allclose(logpphi_i, _logpphi_i))
             
    return logpphi_i*gamma


def log_conditional_posterior_phi_jl(phi_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi=0, prior_scale_phi=1, gamma=1, debug=False):
    # l denotes the coordinate of vector phi_i

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")                         
    Phi[l, i] = phi_il
    theta_test = theta.copy()
    theta_test[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpphi_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpphi_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(phi_il, loc=prior_loc_phi, scale=prior_scale_phi)
    
    pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpphi_il = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(phi_il, mean=prior_loc_phi, cov=prior_scale_phi))
    if debug:
        assert(np.allclose(logpphi_il, _logpphi_il))

    return logpphi_il*gamma

def log_conditional_posterior_z_vec(zi, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z=0, 
                                    prior_scale_z=1, gamma=1, constant_Z=0, penalty_weight_Z=100, debug=False):
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         
    Z[:, i] = zi
    theta_test = theta.copy()
    theta_test[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpz_i = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpz_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(zi, mean=prior_loc_z, cov=prior_scale_z)
    
    pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpz_i = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(zi, mean=prior_loc_z, cov=prior_scale_z))
    if debug:
        assert(np.allclose(logpz_i, _logpz_i))
    
    if abs(penalty_weight_Z) > 1e-10:
        sum_Z_J_vectors = np.sum(Z, axis=1)    
        obj = logpz_i + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)
    else:
        obj = logpz_i
    
    return obj*gamma


def log_conditional_posterior_z_jl(z_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z=0, 
                                   prior_scale_z=1, gamma=1, constant_Z=0, penalty_weight_Z=100, debug=False):
    # l denotes the coordinate of vector z_i
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         
    Z[l, i] = z_il
    theta_test = theta.copy()
    theta_test[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpz_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpz_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(z_il, loc=prior_loc_z, scale=prior_scale_z)
             
    pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpz_il = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(z_il, mean=prior_loc_z, cov=prior_scale_z))
    if debug:
        assert(np.allclose(logpz_il, _logpz_il))

    if abs(penalty_weight_Z) > 1e-10:
        sum_Z_J_vectors = np.sum(Z, axis=1)    
        obj = logpz_il + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)
    else:
        obj = logpz_il

    return obj*gamma


def log_conditional_posterior_alpha_j(alpha, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_alpha=0, prior_scale_alpha=1, gamma=1, debug=False):
    # Assuming independent, Gaussian alphas.
    # Hence, even when evaluating with vector parameters, we use the uni-dimensional posterior for alpha.

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()    
    theta_test[param_positions_dict["alpha"][0] + idx] = alpha     
    if debug:
        _logpalpha_j = 0
        for j in range(J):        
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpalpha_j += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha)
                
    pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpalpha_j = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha))
    if debug:
        assert(np.allclose(logpalpha_j, _logpalpha_j))
    
    return logpalpha_j*gamma

def log_conditional_posterior_beta_i(beta, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_beta=0, prior_scale_beta=1, gamma=1, debug=False):
    
    # print(param_positions_dict["beta"][0],param_positions_dict["beta"][0] + idx, idx)
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()
    theta_test[param_positions_dict["beta"][0] + idx] = beta
    if debug:
        _logpbeta_k = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpbeta_k += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta)

    pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpbeta_k = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta))
    if debug:
        assert(np.allclose(logpbeta_k, _logpbeta_k))
    
    return logpbeta_k*gamma

def log_conditional_posterior_gamma(gamma, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=1, prior_loc_gamma=0, 
                                    prior_scale_gamma=1, debug=False):    
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()
    theta_test[param_positions_dict["gamma"][0]] = gamma
    if debug:
        _logpgamma = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpgamma += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma)

    pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpgamma = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma))
    if debug:
        assert(np.allclose(logpgamma, _logpgamma))
                    
    return logpgamma*gamma_annealing

def log_conditional_posterior_delta(delta, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, prior_loc_delta=0, prior_scale_delta=1, debug=False):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()
    theta_test[param_positions_dict["delta"][0]] = delta
    if debug:
        _logpdelta = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)           
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpdelta += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(delta, loc=prior_loc_delta, scale=prior_scale_delta)

    pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpdelta = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(delta, loc=prior_loc_delta, scale=prior_scale_delta))   
    if debug:
        assert(np.allclose(logpdelta, _logpdelta))     
        
    return logpdelta**gamma

def log_conditional_posterior_sigma_e(sigma_e, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, 
                                    prior_loc_sigmae=0, prior_scale_sigmae=1, min_sigma_e=0.0001, debug=False):    
    
    tig = TruncatedInverseGamma(alpha=prior_loc_sigmae, beta=prior_scale_sigmae, lower=min_sigma_e, upper=10*np.sqrt(prior_scale_sigmae)+prior_scale_sigmae)    
    mu_e = 0
    theta_test = theta.copy()
    theta_test[param_positions_dict["sigma_e"][0]] = sigma_e
    if debug:
        _logpsigma_e = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpsigma_e += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf + tig.logpdf(sigma_e)
    
    pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpsigma_e = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + tig.logpdf(sigma_e))   
    if debug:
        assert(np.allclose(logpsigma_e, _logpsigma_e)) 
        
    return logpsigma_e*gamma

####################### ICM #############################
