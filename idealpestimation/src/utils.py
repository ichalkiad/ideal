from scipy.stats import qmc
import ipdb
import math
import time
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
from datetime import datetime, timedelta

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
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, L, tol, \
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
        # mu_e = params_hat["mu_e"]
        # sigma_e = params_hat["sigma_e"]        
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
        # mu_e = params_hat["mu_e"]
        # sigma_e = params_hat["sigma_e"]        

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

def get_T0(Y, J, K, d, parameter_names, dst_func, param_positions_dict, args):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
                gridpoints_num, diff_iter, disp  = args

    sampler = qmc.Sobol(d=parameter_space_dim, scramble=False)   
    thetas = sampler.random_base2(m=10)
    logps = []
    for theta in thetas[1:]:      
        params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)    
        mu_e = params_hat["mu_e"]
        sigma_e = params_hat["sigma_e"]
        logp = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                logp += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(params_hat["X"][:, i], mean=prior_loc_x, cov=prior_scale_x) \
                                            + multivariate_normal.logpdf(params_hat["Z"][:, j], mean=prior_loc_z, cov=prior_scale_z) \
                                            + multivariate_normal.logpdf(params_hat["alpha"][j], mean=prior_loc_alpha, cov=prior_scale_alpha)\
                                            + multivariate_normal.logpdf(params_hat["beta"][i], mean=prior_loc_beta, cov=prior_scale_beta)
        logps.append(logp[0])
    
    diffs = np.diff(logps)
    T0 = 2*np.max(diffs)
    
    return T0

def update_annealing_temperature(gamma_prev, n, temperature_rate, temperature_steps):

    delta_n = None
    if (gamma_prev >= temperature_steps[0] and gamma_prev <= temperature_steps[1]):
        delta_n = temperature_rate[0]
    elif (gamma_prev > temperature_steps[1] and gamma_prev <= temperature_steps[2]):
        delta_n = temperature_rate[1]
    elif (gamma_prev > temperature_steps[2] and gamma_prev <= temperature_steps[3]):
        delta_n = temperature_rate[2]
    elif (gamma_prev > temperature_steps[3]):
        delta_n = temperature_rate[3]

    # print("Delta_{} = {}".format(n, delta_n))
            
    gamma = gamma_prev + delta_n

    return gamma, delta_n

def get_min_achievable_mse_under_rotation_scaling(param_true, param_hat):

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
    error = np.sum((param_true - (param_hat @ R + t))**2)
    
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(R.shape[0]))    
    det_is_one = np.abs(np.linalg.det(R) - 1.0) < 1e-10    
    t_shape_correct = t.shape == (param_hat.shape[1],)
    if not (orthogonality_error < 1e-10 and det_is_one and t_shape_correct):
        raise AttributeError("Error in solving projection probelm?")
    
    return R, t, error


def compute_and_plot_mse(theta_true, theta_hat, annealing_step, iteration, delta_rate, gamma_n, args, param_positions_dict,
                         plot_online=True, fig_theta_full=None, mse_theta_full=[], fig_xz=None, mse_x_list=[], mse_z_list=[]):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, N, L, tol, \
    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
            gridpoints_num, diff_iter, disp  = args

    if fig_theta_full is None:
        fig_theta_full = go.Figure()    
    # compute with full theta vector
    mse = np.sum((theta_true - theta_hat)**2)/len(theta_true)    
    mse_theta_full.append(mse)    
    if plot_online:
        fig_theta_full.add_trace(go.Box(
                                y=np.asarray(mse_theta_full).flatten(), 
                                x=[delta_rate] * len(mse_theta_full),
                                name="Annealing step = {}<br>Iteration = {}<br>gamma_{}={}".format(delta_rate, iteration, annealing_step, gamma_n),
                                boxpoints='outliers'
                            ))
        fig_theta_full.show()
        savename = "{}/mse_plots_theta/theta_full.html".format(DIR_out)
        pathlib.Path("{}/mse_plots_theta/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig_theta_full, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=False,
                            print_png=True, print_html=True, print_pdf=False)

    # compute min achievable mse for X, Z under rotation and scaling
    params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
    X_true = np.asarray(params_true["X"]).reshape((d, K), order="F")       
    Z_true = np.asarray(params_true["Z"]).reshape((d, J), order="F")                         

    params_hat = optimisation_dict2params(theta_hat, param_positions_dict, J, K, d, parameter_names)
    X_hat = np.asarray(params_hat["X"]).reshape((d, K), order="F")       
    Z_hat = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         

    if fig_xz is None:
        fig_xz = go.Figure()    
    # compute with full theta vector
    Rx, tx, mse_x = get_min_achievable_mse_under_rotation_scaling(param_true=X_true, param_hat=X_hat)
    mse_x_list.append(mse_x)
    Rz, tz, mse_z = get_min_achievable_mse_under_rotation_scaling(param_true=Z_true, param_hat=Z_hat)
    mse_z_list.append(mse_z)    
    if plot_online:
        fig_xz.add_trace(go.Box(
                            y=np.asarray(mse_x_list).tolist(), 
                            x=[delta_rate] * len(mse_x_list),
                            name="Users<br>Annealing step = {:.4f}<br>Iteration = {}<br>gamma_{}={}".format(delta_rate, iteration, annealing_step, gamma_n),
                            boxpoints='outliers', line=dict(color="red")
                            ))
        fig_xz.add_trace(go.Box(
                            y=np.asarray(mse_z_list).tolist(), 
                            x=[delta_rate] * len(mse_z_list),
                            name="Politicians<br>Annealing step = {:.4f}<br>Iteration = {}<br>gamma_{}={}".format(delta_rate, iteration, annealing_step, gamma_n),
                            boxpoints='outliers', line=dict(color="green")
                            ))
        fig_xz.show()
        savename = "{}/mse_plots_xz/theta_xz_rotated_translated.html".format(DIR_out)
        pathlib.Path("{}/mse_plots_xz/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig_xz, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=False,
                            print_png=True, print_html=True, print_pdf=False)

    return mse_theta_full, fig_theta_full, mse_x_list, mse_z_list, fig_xz


def log_conditional_posterior_x_vec(xi, i, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1, debug=False):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[:, i] = xi
    theta[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpx_i = 0   
        for j in range(J):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                  
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)        
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)        
            _logpx_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(xi, mean=prior_loc_x, cov=prior_scale_x)
    
    pijs = p_ij_arg(i, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
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
    theta[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpx_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpx_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(x_il, loc=prior_loc_x, scale=prior_scale_x)
    
    pijs = p_ij_arg(i, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
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
    theta[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpphi_i = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpphi_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(phii, mean=prior_loc_phi, cov=prior_scale_phi)

    pijs = p_ij_arg(i, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
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
    theta[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpphi_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpphi_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(phi_il, loc=prior_loc_phi, scale=prior_scale_phi)
    
    pijs = p_ij_arg(i, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
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
    theta[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpz_i = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpz_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(zi, mean=prior_loc_z, cov=prior_scale_z)
    
    pijs = p_ij_arg(i, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
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
    theta[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpz_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpz_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(z_il, loc=prior_loc_z, scale=prior_scale_z)
             
    pijs = p_ij_arg(i, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
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
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["alpha"][0]:param_positions_dict["alpha"][0] + idx] = alpha
    if debug:
        _logpalpha_j = 0
        for j in range(J):        
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpalpha_j += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha)
                
    pijs = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpalpha_j = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha))
    if debug:
        assert(np.allclose(logpalpha_j, _logpalpha_j))
   
    return logpalpha_j*gamma

def log_conditional_posterior_beta_i(beta, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_beta=0, prior_scale_beta=1, gamma=1, debug=False):
    
    # print(param_positions_dict["beta"][0],param_positions_dict["beta"][0] + idx, idx)
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["beta"][0]:param_positions_dict["beta"][0] + idx] = beta
    if debug:
        _logpbeta_k = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpbeta_k += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta)

    pijs = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpbeta_k = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta))
    if debug:
        assert(np.allclose(logpbeta_k, _logpbeta_k))

    return logpbeta_k*gamma

def log_conditional_posterior_gamma(gamma, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=1, debug=False):    
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["gamma"][0]:param_positions_dict["gamma"][1]] = gamma
    if debug:
        _logpgamma = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpgamma += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf

    pijs = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpgamma = np.sum(Y*logcdfs + (1-Y)*log1mcdfs)
    if debug:
        assert(np.allclose(logpgamma, _logpgamma))
                    
    return logpgamma*gamma_annealing

def log_conditional_posterior_delta(delta, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, debug=False):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["delta"][0]:param_positions_dict["delta"][1]] = delta
    if debug:
        _logpdelta = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)           
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpdelta += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf

    pijs = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpdelta = np.sum(Y*logcdfs + (1-Y)*log1mcdfs)   
    if debug:
        assert(np.allclose(logpdelta, _logpdelta))     
        
    return logpdelta**gamma

def log_conditional_posterior_mu_e(mu_e, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, debug=False):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    sigma_e = params_hat["sigma_e"]
    theta[param_positions_dict["mu_e"][0]:param_positions_dict["mu_e"][1]] = mu_e
    if debug:
        _logpmu_e = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpmu_e += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf
    
    pijs = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpmu_e = np.sum(Y*logcdfs + (1-Y)*log1mcdfs)   
    if debug:
        assert(np.allclose(logpmu_e, _logpmu_e))    
        
    return logpmu_e*gamma

def log_conditional_posterior_sigma_e(sigma_e, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, debug=False):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = params_hat["mu_e"]
    theta[param_positions_dict["sigma_e"][0]:param_positions_dict["sigma_e"][1]] = sigma_e
    if debug:
        _logpsigma_e = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpsigma_e += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf
    
    pijs = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)                
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpsigma_e = np.sum(Y*logcdfs + (1-Y)*log1mcdfs)   
    if debug:
        assert(np.allclose(logpsigma_e, _logpsigma_e)) 
        
    return logpsigma_e*gamma

####################### ICM #############################
