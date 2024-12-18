from scipy.stats import qmc
import ipdb
import math
import numpy as np
import math
import pathlib
import plotly.graph_objs as go
import plotly.io as pio
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

def get_global_theta(from_row, to_row, parameter_space_dim, J, K, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e, variance, total_K):
    """
    K: size of row split
    """
    param_positions_dict = dict()
    optim_vector = np.zeros((parameter_space_dim,))
    var_vector = np.zeros((parameter_space_dim,))
    k = from_row*d
    kvar = 0
    print("global theta start {} - local theta start {}".format(k, kvar))
    print("total K: {}".format(total_K))
    print("N: {}".format(K))
    
    ipdb.set_trace()   

    for param in parameter_names:
        if param == "X":
            param_positions_dict[param] = (k, k + K*d)
            
            if k+K*d != to_row*d:
                ipdb.set_trace()   
            
            Xvec = X.reshape((d*K,), order="F").tolist()        
            optim_vector[k:k+K*d] = Xvec
            var_vector[k:k+K*d] = variance[kvar:kvar+K*d]
            k = total_K*d    
            kvar += K*d

            print("after adding X: global theta {} - local theta {}".format(k, kvar))

        elif param in ["Z"]:
            param_positions_dict[param] = (k, k + J*d)            
            Zvec = Z.reshape((d*J,), order="F").tolist()         
            optim_vector[k:k + J*d] = Zvec
            var_vector[k:k + J*d] = variance[kvar:kvar+J*d]
            k += J*d
            kvar += J*d

            print("after adding Z: global theta {} - local theta {}".format(k, kvar))

        elif param in ["Phi"]:            
            param_positions_dict[param] = (k, k + J*d)            
            Phivec = Phi.reshape((d*J,), order="F").tolist()         
            optim_vector[k:k + J*d] = Phivec
            var_vector[k:k + J*d] = variance[kvar:kvar+J*d]
            k += J*d
            kvar += J*d       

            print("after adding Phi: global theta {} - local theta {}".format(k, kvar))
   
        elif param == "beta":
            param_positions_dict[param] = (k+from_row, k + from_row + K)               
            optim_vector[k+from_row:k + from_row + K] = beta            
            var_vector[k+from_row:k + from_row + K] = variance[kvar:kvar+K]
            k += total_K    
            kvar += total_K

            print("after adding beta: global theta {} - local theta {}".format(k, kvar))


        elif param == "alpha":
            param_positions_dict[param] = (k, k + J)               
            optim_vector[k:k + J] = alpha
            var_vector[k:k + J] = variance[kvar:kvar+J]
            k += J    
            kvar += J

            print("after adding alpha: global theta {} - local theta {}".format(k, kvar))

        
        elif param == "gamma":
            param_positions_dict[param] = (k, k + 1)                        
            optim_vector[k:k + 1] = gamma
            var_vector[k:k + 1] = variance[kvar:kvar+1]
            k += 1
            kvar += 1

            print("after adding gamma: global theta {} - local theta {}".format(k, kvar))

        elif param == "delta":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector[k:k + 1] = delta
            var_vector[k:k + 1] = variance[kvar:kvar+1]
            k += 1
            kvar += 1

            print("after adding delta: global theta {} - local theta {}".format(k, kvar))

        elif param == "mu_e":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector[k:k + 1] = mu_e
            var_vector[k:k + 1] = variance[kvar:kvar+1]
            k += 1
            kvar += 1

            print("after adding mu_e: global theta {} - local theta {}".format(k, kvar))

        elif param == "sigma_e":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector[k:k + 1] = sigma_e
            var_vector[k:k + 1] = variance[kvar:kvar+1]
            k += 1
            kvar += 1
        
            print("after adding sigma_e: global theta {} - local theta {}".format(k, kvar))
        
    return optim_vector, var_vector, param_positions_dict 

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
        elif param == "beta":
            param_positions_dict[param] = (k, k + K)               
            optim_vector.extend(beta.tolist())            
            k += K    
        elif param == "alpha":
            param_positions_dict[param] = (k, k + J)               
            optim_vector.extend(alpha.tolist())                        
            k += J    
        elif param in ["Z"]:
            param_positions_dict[param] = (k, k + J*d)            
            Zvec = Z.reshape((d*J,), order="F").tolist()         
            optim_vector.extend(Zvec)            
            k += J*d
        elif param in ["Phi"]:            
            param_positions_dict[param] = (k, k + J*d)            
            Phivec = Phi.reshape((d*J,), order="F").tolist()         
            optim_vector.extend(Phivec)            
            k += J*d
        # elif param == "c":
        #     param_positions_dict[param] = (k, k + d)            
        #     optim_vector.append(c)
        #     k += d
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
            param_out = param_out.reshape((K, d), order="F")                     
        elif param in ["Z"]:            
            param_out = param_out.reshape((J, d), order="F")                      
        elif param in ["Phi"]:            
            param_out = param_out.reshape((J, d), order="F")                                
        params_out[param] = param_out
        
    return params_out

def initialise_optimisation_vector_sobol(m=16, J=2, K=2, d=1):

    sobol_generators = dict()
    sampler = qmc.Sobol(d=1, scramble=False)   
    sample = sampler.random_base2(m=2)                               
    gamma = float(sample[0])    
    delta = float(sample[1])
    sobol_generators["gammadelta"] = [sampler]
    
    # sampler = qmc.Sobol(d=d, scramble=False)   
    # sample = sampler.random_base2(m=2)                               
    # c = sample[0]    
    # sobol_generators["c"] = [sampler]

    sampler = qmc.Sobol(d=K, scramble=False)   
    sample = sampler.random_base2(m=2)                               
    beta = sample[0]
    sobol_generators["beta"] = [sampler]

    sampler = qmc.Sobol(d=J, scramble=False)   
    sample = sampler.random_base2(m=2)                               
    alpha = sample[0]    
    sobol_generators["alpha"] = [sampler]

    sampler = qmc.Sobol(d=d, scramble=False)   
    sample = sampler.random_base2(m=math.ceil(np.log2(J)))                               
    Phi = sample[:J, :]
    sample = sampler.random_base2(m=math.ceil(np.log2(J)))                               
    Z = sample[:J, :]
    sobol_generators["PhiZ"] = [sampler]

    sampler = qmc.Sobol(d=d, scramble=False)   
    sample = sampler.random_base2(m=math.ceil(np.log2(K)))                               
    X = sample[:K, :]
            
    mu_e = 0
    sigma_e = 1

    return X, Z, Phi, alpha, beta, gamma, delta, mu_e, sigma_e

def visualise_hessian(hessian, title='Hessian matrix'):
    
    hessian = np.asarray(hessian)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=hessian,
        colorscale='RdBu_r',  
        zmin=-np.max(np.abs(hessian)), 
        zmax=np.max(np.abs(hessian)),
        colorbar=dict(title='Hessian Value')
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Parameter Index',
        yaxis_title='Parameter Index',
        width=600,
        height=600,
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

    ipdb.set_trace()

    path = pathlib.Path(DIR_out)
    estimates_names = [file.name for file in path.iterdir() if file.is_file() and "estimationresult_dataset" in file.name]
    weighted_estimate = None
    all_weights = []
    all_estimates = []
    for estim in estimates_names:
        with jsonlines.open("{}/{}".format(DIR_out, estim), mode="r") as f:               
            for result in f.iter(type=dict, skip_invalid=True):                      
                weight = result["Theta Variance"]
                theta = result["Theta"]
                all_weights.append(weight)
                all_estimates.append(theta)
    all_weights = np.array(all_weights)
    all_estimates = np.array(all_estimates)
    # sum acrocs each coordinate's weight
    all_weights_sum = np.sum(all_weights, axis=0)
    all_weights_norm = all_weights/all_weights_sum
    # element-wise multiplication
    weighted_estimate = np.sum(all_weights_norm*all_estimates, axis=0)
    param_positions_dict = result["param_positions_dict_global"]
    params_out = optimisation_dict2params(weighted_estimate, param_positions_dict, J, K, d, parameter_names)

    return params_out
####################### MLE #############################