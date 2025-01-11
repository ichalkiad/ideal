from scipy.stats import qmc
import ipdb
import math
import numpy as np
import math
import pathlib
from scipy.stats import norm, multivariate_normal
import plotly.graph_objs as go
import plotly.io as pio
import jax
import jax.numpy as jnp
from jax import jvp, grad
import jsonlines
from scipy.optimize import approx_fprime

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

def params2optimisation_dict(J, K, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e):
    
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
        elif param == "mu_e":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(mu_e)
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


def create_constraint_functions(n, param_positions_dict=None, sum_z_constant=0):
    
    # # for scale_e
    # def positive_constraints(x):
    #     """Constraint: scale_e should be positive - last entry in the optimisation vector"""
    #     return x[-1]    
    # def sum_zero_constraint(x):
    #     """Constraint: Sum of Z's should be a constant - default 0, i.e. balanced politician set"""
    #     return np.sum(x[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]])
    bounds = [(None, None)]*(n-1)
    bounds.append((0.0, None))
    
    return bounds, None


def initialise_optimisation_vector_sobol(m=16, J=2, K=2, d=1):

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
            
    mu_e = np.linspace(-1, 1, 100).tolist()
    sigma_e = np.linspace(0.1, 2, 100).tolist()

    return X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e

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
    
def combine_estimate_variance_rule(DIR_out, J, K, d, parameter_names):

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
                        else:                            
                            weight = result["variance_{}".format(param)]                            
                            theta = result[param]
                            all_weights.append(weight)
                            all_estimates.append(theta)
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
    
    return params_out

####################### MLE #############################



####################### ICM #############################

def create_constraint_functions_icm(n, vector_coordinate=None, param=None, param_positions_dict=None, args=None):
    
    bounds = []
    grid_width_std = 5
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, delta_n, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args
    
    if param == "alpha":
        bounds.append((-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha))
    elif param == "beta":
        bounds.append((-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta))
    elif param == "gamma":            
        bounds.append((None, None))  ########## MODIFY TO SET NON_ZERO CONSTRAINT?
    elif param == "delta":                    
        bounds.append((None, None))  ########## MODIFY TO SET NON_ZERO CONSTRAINT?        
    elif param == "mu_e":
        bounds.append((None, None))
    elif param == "sigma_e":
        bounds.append((0.000001, 5))
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

def log_complement_from_log_cdf(log_cdfx, x, mean, variance, use_jax=False):
    """
    Computes log(1-CDF(x)) given log(CDF(x)) in a numerically stable way.
    """    
    if log_cdfx < -0.693:  #log(0.5)
        # If CDF(x) < 0.5, direct computation is stable  
        if use_jax:
            ret = jnp.log1p(-jnp.exp(log_cdfx))
        else:
            ret = np.log1p(-np.exp(log_cdfx))
    else: 
        # If CDF(x) ≥ 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
        if use_jax:
            if isinstance(x, jnp.ndarray) and len(x.shape) > 1 and (x.shape[0] > 1 or x.shape[1] > 1):                        
                ret = jax.scipy.stats.multivariate_normal.logcdf(x, mean=mean, cov=variance)            
            else:            
                ret = jax.scipy.stats.norm.logcdf(-x, loc=mean, scale=variance)
        else:
            if isinstance(x, np.ndarray) and len(x.shape) > 1 and (x.shape[0] > 1 or x.shape[1] > 1):                        
                ret = multivariate_normal.logcdf(x, mean=mean, cov=variance)            
            else:            
                ret = norm.logcdf(-x, loc=mean, scale=variance)       
                    
    return ret


def p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict):

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
    # mu_e = params_hat["mu_e"]
    # sigma_e = params_hat["sigma_e"]
        
    phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]
    
    return phi


def log_conditional_posterior_x_vec(xi, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[:, i] = xi
    theta[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    logpx_i = 0
    for j in range(J):
        pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
        philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
        log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
        logpx_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(xi, mean=prior_loc_x, cov=prior_scale_x)
             
    return logpx_i**gamma

def log_conditional_posterior_x_il(x_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1):
    # l denotes the coordinate of vector x_i

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[l, i] = x_il
    theta[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    logpx_il = 0
    for j in range(J):
        pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
        philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
        log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
        logpx_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(x_il, loc=prior_loc_x, scale=prior_scale_x)
             
    return logpx_il**gamma


def log_conditional_posterior_phi_vec(phii, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi=0, prior_scale_phi=1, gamma=1):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")                         
    Phi[:, i] = phii
    theta[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    logpphi_i = 0
    for j in range(J):
        pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
        philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
        log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
        logpphi_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(phii, mean=prior_loc_phi, cov=prior_scale_phi)
             
    return logpphi_i**gamma


def log_conditional_posterior_phi_jl(phi_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi=0, prior_scale_phi=1, gamma=1):
    # l denotes the coordinate of vector phi_i

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")                         
    Phi[l, i] = phi_il
    theta[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    logpphi_il = 0
    for j in range(J):
        pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
        philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
        log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
        logpphi_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(phi_il, loc=prior_loc_phi, scale=prior_scale_phi)
             
    return logpphi_il**gamma

def log_conditional_posterior_z_vec(zi, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z=0, 
                                    prior_scale_z=1, gamma=1, constant_Z=0, penalty_weight_Z=100):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         
    Z[:, i] = zi
    theta[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    logpz_i = 0
    for j in range(J):
        pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
        philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
        log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
        logpz_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(zi, mean=prior_loc_z, cov=prior_scale_z)
    
    if abs(penalty_weight_Z) > 1e-10:
        sum_Z_J_vectors = np.sum(Z, axis=1)    
        obj = logpz_i + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)
    else:
        obj = logpz_i

    return obj**gamma


def log_conditional_posterior_z_jl(z_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z=0, 
                                   prior_scale_z=1, gamma=1, constant_Z=0, penalty_weight_Z=100):
    # l denotes the coordinate of vector z_i
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         
    Z[l, i] = z_il
    theta[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    logpz_il = 0
    for j in range(J):
        pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
        philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
        log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
        logpz_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(z_il, loc=prior_loc_z, scale=prior_scale_z)
             
    if abs(penalty_weight_Z) > 1e-10:
        sum_Z_J_vectors = np.sum(Z, axis=1)    
        obj = logpz_il + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)
    else:
        obj = logpz_il

    return obj**gamma


def log_conditional_posterior_alpha_j(alpha, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_alpha=0, prior_scale_alpha=1, gamma=1):
    # Assuming independent, Gaussian alphas.
    # Hence, even when evaluating with vector parameters, we use the uni-dimensional posterior for alpha.

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["alpha"][0]:param_positions_dict["alpha"][0] + idx] = alpha
    logpalpha_j = 0
    for j in range(J):
        for i in range(K):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            logpalpha_j += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha)
             
    return logpalpha_j**gamma

def log_conditional_posterior_beta_i(beta, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_beta=0, prior_scale_beta=1, gamma=1):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["beta"][0]:param_positions_dict["beta"][0] + idx] = beta
    logpbeta_k = 0
    for j in range(J):
        for i in range(K):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            logpbeta_k += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta)
             
    return logpbeta_k**gamma

def log_conditional_posterior_gamma(gamma, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=1):    
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["gamma"][0]:param_positions_dict["gamma"][1]] = gamma
    logpgamma = 0
    for j in range(J):
        for i in range(K):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            logpgamma += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
                    
    return logpgamma**gamma_annealing

def log_conditional_posterior_delta(delta, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["delta"][0]:param_positions_dict["delta"][1]] = delta
    logpdelta = 0
    for j in range(J):
        for i in range(K):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)           
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            logpdelta += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf
        
    return logpdelta**gamma

def log_conditional_posterior_mu_e(mu_e, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["mu_e"][0]:param_positions_dict["mu_e"][1]] = mu_e
    logpmu_e = 0
    for j in range(J):
        for i in range(K):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            logpmu_e += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf
        
    return logpmu_e**gamma

def log_conditional_posterior_sigma_e(sigma_e, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    theta[param_positions_dict["sigma_e"][0]:param_positions_dict["sigma_e"][1]] = sigma_e
    logpsigma_e = 0
    for j in range(J):
        for i in range(K):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            logpsigma_e += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf
        
    return logpsigma_e**gamma

####################### ICM #############################
