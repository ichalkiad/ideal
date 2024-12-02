from scipy.stats import qmc
import ipdb
import math
import numpy as np
import math

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
            param_out = param_out.reshape((d, K), order="F")                     
        elif param in ["Z"]:            
            param_out = param_out.reshape((d, J), order="F")                      
        elif param in ["Phi"]:            
            param_out = param_out.reshape((d, J), order="F")                                
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