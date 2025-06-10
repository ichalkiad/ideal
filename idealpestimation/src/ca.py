import os 

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMBA_NUM_THREADS"] = "2"
# os.environ["JAX_NUM_THREADS"] = "1000"
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=500"

import sys
import ipdb
# import jax
import pathlib
import jsonlines
import numpy as np
import random
from idealpestimation.src.utils import pickle, optimisation_dict2params,\
                                        time, timedelta, parse_input_arguments, \
                                            rank_and_plot_solutions, print_threadpool_info, \
                                                get_min_achievable_mse_under_rotation_trnsl, clean_up_data_matrix, get_slurm_experiment_csvs
from idealpestimation.src.efficiency_monitor import Monitor
from prince import CA
from prince import utils as ca_utils
from prince import svd as ca_svd
import prince
import pandas as pd
from scipy import sparse
from sklearn.utils import check_array

class CA_custom(CA):
    
    @ca_utils.check_is_dataframe_input
    def fit(self, X, y=None):
        
        if self.check_input:
            check_array(X)

        if (X < 0).any().any():
            raise ValueError("All values in X should be positive")

        _, row_names, _, col_names = ca_utils.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X.astype(float) / np.sum(X)

        # Compute row and column masses
        rsum = X.sum(axis=1)
        csum = X.sum(axis=0)
        # if 0 in rsum:
        #     rsum += 10e-12
        # if 0 in csum:
        #     csum += 10e-12
        self.row_masses_ = pd.Series(rsum, index=row_names)
        self.col_masses_ = pd.Series(csum, index=col_names)

        self.active_rows_ = self.row_masses_.index.unique()
        self.active_cols_ = self.col_masses_.index.unique()

        # Compute standardised residuals
        r = self.row_masses_.to_numpy()
        c = self.col_masses_.to_numpy()
        S = sparse.diags(r**-0.5) @ (X - np.outer(r, c)) @ sparse.diags(c**-0.5)

        # Compute SVD on the standardised residuals
        self.svd_ = ca_svd.compute_svd(
            X=S,
            n_components=min(self.n_components, min(X.shape) - 1),
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine,
        )

        # Compute total inertia
        self.total_inertia_ = np.einsum("ij,ji->", S, S.T)

        self.row_contributions_ = pd.DataFrame(
            sparse.diags(self.row_masses_.values)
            @ np.divide(
                # Same as row_coordinates(X)
                (
                    sparse.diags(self.row_masses_.values**-0.5)
                    @ self.svd_.U
                    @ sparse.diags(self.svd_.s)
                )
                ** 2,
                self.eigenvalues_,
                out=np.zeros((len(self.row_masses_), len(self.eigenvalues_))),
                where=self.eigenvalues_ > 0,
            ),
            index=self.row_masses_.index,
        )

        self.column_contributions_ = pd.DataFrame(
            sparse.diags(self.col_masses_.values)
            @ np.divide(
                # Same as col_coordinates(X)
                (
                    sparse.diags(self.col_masses_.values**-0.5)
                    @ self.svd_.V.T
                    @ sparse.diags(self.svd_.s)
                )
                ** 2,
                self.eigenvalues_,
                out=np.zeros((len(self.col_masses_), len(self.eigenvalues_))),
                where=self.eigenvalues_ > 0,
            ),
            index=self.col_masses_.index,
        )

        return self
    
    @ca_utils.check_is_dataframe_input
    @prince.ca.select_active_columns
    def row_coordinates(self, X):
        """The row principal coordinates."""

        _, row_names, _, _ = ca_utils.make_labels_and_names(X)
        index_name = X.index.name

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()
    
        X_csum = X.sum(axis=1)
        # if 0 in X_csum:
        #     X_csum = X_csum.astype(np.float64)
        #     X_csum += 10e-12
        # Normalise the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X_csum[:, None]
        else:
            X = X / X_csum

        return pd.DataFrame(
            data=X @ sparse.diags(self.col_masses_.to_numpy() ** -0.5) @ self.svd_.V.T,
            index=pd.Index(row_names, name=index_name),
        )



def do_correspondence_analysis(ca_runner, Y, param_positions_dict, args, plot_online=False, seedint=1234):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    # params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
    # X_true = np.asarray(params_true["X"]) # d x K       
    # Z_true = np.asarray(params_true["Z"]) # d x J          

    df = pd.DataFrame(Y, index=np.arange(0, K, 1), columns=np.arange(0, J, 1))
    ca_out = ca_runner.fit(df)
    
    Xhat = ca_out.row_coordinates(df).values.T
    Zhat = ca_out.column_coordinates(df).values.T
    theta_hat = np.zeros((parameter_space_dim,))
    for param in parameter_names:
        if param == "X":
            theta_hat[param_positions_dict[param][0]:param_positions_dict[param][1]] = Xhat.reshape((K*d,), order="F") 
        elif param == "Z":
            theta_hat[param_positions_dict[param][0]:param_positions_dict[param][1]] = Zhat.reshape((J*d,), order="F")
        else:
            # Note: passing true parameter values for compatibility with rank_and_plot_solutions
            theta_hat[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]].copy()

    return theta_hat

def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, temperature_rate=[0, 1], temperature_steps=[1e-3], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False,
        theta_true=None, percentage_parameter_change=1, min_sigma_e=None, fastrun=False,
        max_restarts=2, max_partial_restarts=2, max_halving=2, plot_online=False, seedint=None):
        
        for m in range(trialsmin, trialsmax, 1):
            print(trialsmin)
            DIR_out = "{}/{}/estimation_CA/".format(data_location, m)
            pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
            
            # load data    
            with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
                Y = pickle.load(f)
            Y = Y.astype(np.int8).reshape((K, J), order="F")    

            # done internally in prince.CA
            # ypp = np.sum(Y)
            # vec_ones = np.ones((J,1))
            # wm = (Y @ vec_ones)/ypp
            # wm_diag = np.diag(1/np.sqrt(wm))
            # wn = (vec_ones.T @ Y)/ypp
            # wn_diag = np.diag(1/np.sqrt(wn))
            # Smat = (wm_diag @ (Y - ypp *(wm @ wn)) @ wn_diag)/ypp

            ca = CA_custom(
                n_components=d,
                n_iter=5,
                copy=True,
                check_input=True,
                engine='sklearn',
                random_state=seedint
            )

            param_positions_dict = dict()            
            k = 0
            for param in parameter_names:
                if param == "X":
                    param_positions_dict[param] = (k, k + K*d)                       
                    k += K*d    
                elif param in ["Z"]:
                    param_positions_dict[param] = (k, k + J*d)                                
                    k += J*d
                elif param in ["Phi"]:            
                    param_positions_dict[param] = (k, k + J*d)                                
                    k += J*d
                elif param == "beta":
                    param_positions_dict[param] = (k, k + K)                                   
                    k += K    
                elif param == "alpha":
                    param_positions_dict[param] = (k, k + J)                                       
                    k += J    
                elif param == "gamma":
                    param_positions_dict[param] = (k, k + 1)                                
                    k += 1
                elif param == "delta":
                    param_positions_dict[param] = (k, k + 1)                                
                    k += 1
                elif param == "sigma_e":
                    param_positions_dict[param] = (k, k + 1)                                
                    k += 1

            with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
                for result in f.iter(type=dict, skip_invalid=True):
                    for param in parameter_names:
                        theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
            
            # Y, K, J, theta_true, param_positions_dict, parameter_space_dim = clean_up_data_matrix(Y, K, J, d, theta_true, parameter_names, param_positions_dict)
            
            args = (DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,                     
                    parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                    prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                    prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                    gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)   
            

            # start efficiency monitoring - interval in seconds
            print_threadpool_info()
            monitor = Monitor(interval=0.01, fastprogram=True)
            monitor.start()            

            t_start = time.time()            
            theta_hat = do_correspondence_analysis(ca, Y, param_positions_dict, args, plot_online=plot_online, seedint=seedint)
            t_end = time.time()
            monitor.stop()
            wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
            efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                        avg_threads, max_threads, avg_processes, max_processes)
            elapsedtime = str(timedelta(seconds=t_end - t_start))   
        
            theta = [(theta_hat, None, None, None, None, None, None, None, None)]
            rank_and_plot_solutions(theta, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, dst_func, param_positions_dict, 
                                    DIR_out, args, seedint=seedint, get_RT_error=True)


if __name__ == "__main__":
    
    Ks = [50000]
    Js = [100]
    sigma_es = [0.01, 0.1, 0.5, 1.0, 5.0]
    M = 10
    batchsize = 1504
    dir_in = "/mnt/hdd2/ioannischalkiadakis/idealdata_rsspaper/"
    dir_out = "/mnt/hdd2/ioannischalkiadakis/"
    get_slurm_experiment_csvs(Ks, Js, sigma_es, M, batchsize, dir_in, dir_out)
    sys.exit()

    # python idealpestimation/src/ca.py --trials 1 --K 30 --J 10 --sigmae 001

    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    args = parse_input_arguments()
    
    if args.trials is None or args.K is None or args.J is None or args.sigmae is None:
        parallel = False
        Mmin = 0
        M = 1
        K = 30
        J = 10
        sigma_e_true = 0.01
        total_running_processes = 20   
        elementwise = True
        evaluate_posterior = True
    else:
        parallel = args.parallel
        trialsstr = args.trials
        if "-" in trialsstr:
            trialsparts = trialsstr.split("-")
            Mmin = int(trialsparts[0])
            M = int(trialsparts[1])
        else:
            Mmin = 0
            M = int(trialsstr)
        K = args.K
        J = args.J
        total_running_processes = args.total_running_processes
        sigma_e_true = args.sigmae
        elementwise = args.elementwise
        evaluate_posterior = args.evaluate_posterior

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes, elementwise, evaluate_posterior)
    
    # if not parallel:
    #     jax.default_device = jax.devices("cpu")[0]
    #     jax.config.update("jax_traceback_filtering", "off")
    
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]
    d = 2  
    prior_loc_x = np.zeros((d,))
    prior_scale_x = np.eye(d)
    prior_loc_z = np.zeros((d,))
    prior_scale_z = np.eye(d)
    prior_loc_phi = np.zeros((d,))
    prior_scale_phi = np.eye(d)
    prior_loc_alpha = 0
    prior_scale_alpha = 1    
    prior_loc_beta = 0
    prior_scale_beta = 1
    prior_loc_gamma = 0
    prior_scale_gamma = 1    
    prior_loc_delta = 0
    prior_scale_delta = 1        
    # a
    prior_loc_sigmae = 3
    # b
    prior_scale_sigmae = 0.5
    max_signal2noise_ratio = 25 # in dB   # max snr

    min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
    print(min_sigma_e)

    tol = 1e-6    
    #/home/ioannischalkiadakis/ideal
    # data_location = "./idealpestimation/data_K{}_J{}_sigmae{}_goodsnr/".format(K, J, str(sigma_e_true).replace(".", ""))
    data_location = "/mnt/hdd2/ioannischalkiadakis/idealdata_rsspaper/data_K{}_J{}_sigmae{}/".format(K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 30      

    args = (None, total_running_processes, data_location, None, parameter_names, J, K, d, None, None, tol,                     
            None, None, None, None, None, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
            None, None, None, min_sigma_e, None)  

    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 3
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    main(J=J, K=K, d=d, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=None, 
        dst_func=None, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, trialsmax=M, 
        penalty_weight_Z=None, constant_Z=None, retries=None, 
        elementwise=elementwise, evaluate_posterior=evaluate_posterior, 
        temperature_rate=None, temperature_steps=None, L=None, tol=tol, 
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, 
        prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
        prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, 
        prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, 
        prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma, 
        prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta,         
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
        gridpoints_num=None, diff_iter=None, disp=None, theta_true=theta_true,
        percentage_parameter_change=None, min_sigma_e=min_sigma_e, fastrun=None,
        max_restarts=None, max_partial_restarts=None, 
        max_halving=None, plot_online=None, seedint=seed_value)
    


    