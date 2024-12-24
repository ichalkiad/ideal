import os 
import sys
import time 
import ipdb
import jax
import pathlib
import jsonlines
import numpy as np
from datetime import datetime
from idealpestimation.src.parallel_manager import jsonlines
# from idealpestimation.src.utils import 



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
    total_running_processes = 1                  
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 4
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 3
    print("Parameter space dimensionality: {}".format(parameter_space_dim))