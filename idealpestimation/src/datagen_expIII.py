import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline as pyo
from typing import Union, Optional
from scipy.stats import norm
import jsonlines
import pathlib
import pickle
import math
import random
import time
from tabulate import tabulate
from sklearn.utils import check_random_state
from idealpestimation.src.utils import p_ij_arg, p_ij_arg_numbafast, params2optimisation_dict, optimisation_dict2params
import ipdb

def generate_normal_data(n_samples, n_dimensions, mu=0, sigma=1, rng=None):
    """
    Generate multi-dimensional normally distributed data.
    
    Parameters:
    - n_samples: Number of data points
    - n_dimensions: Number of dimensions
    - mu: Mean of the normal distribution
    - sigma: Standard deviation of the normal distribution
    """    
    if n_dimensions == 1:
        return rng.normal(mu, sigma, n_samples)
    else:
        return rng.multivariate_normal(mu, sigma, size=n_samples)

def create_histogram_matrix(data, title):
    """
    Create a matrix of histograms for each dimension.
    """
    n_dims = data.shape[1]
    fig = make_subplots(rows=n_dims, cols=1, subplot_titles=[f'Dimension {i+1}' for i in range(n_dims)])
    
    for i in range(n_dims):
        fig.add_trace(
            go.Histogram(x=data[:, i], name=f'Dim {i+1}', nbinsx=30),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=300 * n_dims,
        title_text=f'{title} - Histogram per Dimension',
        showlegend=False
    )
    
    return fig

def plot_array_heatmap(
    array: np.ndarray,
    title: str = "Array Heatmap",
    colorscale: Optional[Union[str, list]] = None,
    show_scale: bool = True,
    boundcolorscale: bool = False,
    x_labels: Optional[list] = None,
    y_labels: Optional[list] = None,
    width: int = 800,
    height: int = 600,
    annotation_format: str = ".2f", xtitle="Columns", ytitle="Rows", show_values=False, colorbar=None
):
    """
    Creates a heatmap visualization of a 2D array using plotly.
    Automatically detects binary arrays and uses black/white colorscale.

    Parameters:
    -----------
    array : np.ndarray
        2D array to visualize
    title : str, optional
        Title of the heatmap
    colorscale : str or list, optional
        Custom colorscale. If None, uses 'black/white' for binary arrays
        and 'Viridis' for non-binary arrays
    show_scale : bool, optional
        Whether to show the colorbar
    x_labels : list, optional
        Custom labels for x-axis
    y_labels : list, optional
        Custom labels for y-axis
    width : int, optional
        Width of the figure in pixels
    height : int, optional
        Height of the figure in pixels
    annotation_format : str, optional
        Format string for cell annotations (e.g., ".2f" for 2 decimal places)
    """
    
    # Input validation
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    # Check if array is binary
    is_binary = np.array_equal(array, array.astype(bool))
    
    # Set default colorscale based on array type
    if colorscale is None:
        colorscale = ['white', 'black'] if is_binary else 'Viridis'
    
    # Create axis labels if not provided
    if x_labels is None:
        x_labels = [str(i) for i in range(array.shape[1])]
    if y_labels is None:
        y_labels = [str(i) for i in range(array.shape[0])]
    
    # Format annotations based on binary/continuous values
    if is_binary:
        text = array.astype(int).astype(str)
    else:
        text = np.array([[f"{x:{annotation_format}}" for x in row] for row in array])
    
    # Create the heatmap
    if boundcolorscale:           
        cb = colorbar
        cb["tickmode"] = "array"
        cb["tickvals"] = [-1, -0.5, 0, 0.5, 1]
        cb["ticktext"] = ['-1.0', '-0.5', '0.0', '0.5', '1.0']
        fig = go.Figure(data=go.Heatmap(
            z=array,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            showscale=show_scale,
            zmid=0.5,    # Center the colorscale at 0
            zmin=0,   # Set minimum value
            zmax=1, 
            colorbar=cb,
            text=text if show_values else None,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>"
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=array,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            showscale=show_scale,   
            colorbar=colorbar,
            text=text if show_values else None,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ))
       
    return fig

def plot_side_by_side_subplots(fig1, fig2, fig3, title="Subplots"):
    """
    Arrange three Plotly figures side by side using subplots.
    
    Parameters:
    fig1, fig2, fig3: Plotly figure objects
    title: Main title for the combined plot
    """
    # Create subplot figure
    if fig2 is not None:
        fig = make_subplots(
            rows=1, 
            cols=3,
            subplot_titles=[fig1.layout.title.text, 
                           fig2.layout.title.text, 
                           fig3.layout.title.text]
        )
    else:
        fig = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=[fig1.layout.title.text,  
                           fig3.layout.title.text]
        )
    
    # Add traces from each figure
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    if fig2 is not None:
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)
        for trace in fig3.data:
            fig.add_trace(trace, row=1, col=3)
    else:
        for trace in fig3.data:
            fig.add_trace(trace, row=1, col=2)
      
    return fig

def create_scatter_plot(data_sets, labels=None, colors=None, sizes=None, symbols=None, title="Interactive Scatter Plot"):
    """
    Create an interactive scatter plot using Plotly for multiple sets of 2D data with customizable appearance.
    
    Parameters:
    data_sets (list): List of tuples, each containing (x_data, y_data) for each dataset
    labels (list): List of strings for legend labels
    colors (list): List of colors for each dataset
    sizes (list): List of marker sizes for each dataset
    symbols (list): List of marker symbols for each dataset
    title (str): Title of the plot
    """
    
    # Set default values if not provided
    n_sets = len(data_sets)
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(n_sets)]
    if colors is None:
        colors = ['blue', 'red', 'green']        
    if sizes is None:
        sizes = [10] * n_sets  # Plotly uses different size scale than matplotlib
    if symbols is None:
        symbols = ['circle', 'square', 'diamond']
    print(labels, colors, symbols, sizes)
    # Create figure
    fig = go.Figure()
    
    # Add each dataset as a separate trace
    for i, ((x_data, y_data), label, color, size, symbol) in enumerate(
            zip(data_sets, labels, colors, sizes, symbols)):
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=label,
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertemplate=
                f"{label}<br>" +
                "X: %{x:.2f}<br>" +
                "Y: %{y:.2f}<br>" +
                "<extra></extra>"  # This removes the secondary box in the hover tooltip
            )
        )
    
    # Update layout with more customization options
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # Center the title
            font=dict(size=24)
        ),
        xaxis=dict(
            title="X-axis",
            title_font=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='LightGrey'
        ),
        yaxis=dict(
            title="Y-axis",
            title_font=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='LightGrey'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    
    return fig  # Return the figure object for potential further modifications

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
                            # autorange="reversed"
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

def generate_trial_data(parameter_names, m, J, K, d, distance_func, utility_func, data_location, param_positions_dict, theta, x_var=None, z_var=None, 
                            alpha_var=None, beta_var=None, debug=False, rng=None):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if "delta" in params_hat.keys():
        delta = params_hat["delta"]
        phis = params_hat["Phi"]
    else:
        delta = -1
        phis = None
    alpha = params_hat["alpha"]
    beta = params_hat["beta"]
    xs = np.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    zs = np.asarray(params_hat["Z"]).reshape((d, J), order="F")        
    pijs = p_ij_arg_numbafast(xs, zs, alpha, beta, gamma, K)
    if debug:              
        utilities_matrix = np.zeros((K, J))    
        # assuming linear utility in this formulation
        for i in range(K):
            for j in range(J):            
                pij = p_ij_arg(i, j, theta, J, K, d, parameter_names, distance_func, param_positions_dict, use_jax=False)            
                utilities_matrix[i,j] = pij
        assert(np.allclose(pijs, utilities_matrix))        
    else:
        utilities_matrix = pijs.copy()
    
    # 50% polarised regime
    # utilities_matrix[:, 0:math.ceil(J/2)] *= -1
    # 25% polarised regime
    # utilities_matrix[:, 0:math.ceil(J/4)] *= -0.25
    # utilities_matrix[:, math.ceil(J/4):math.ceil(2*J/4)] *= -0.9
    # utilities_matrix[:, math.ceil(2*J/4):math.ceil(3*J/4)] *= -1.5
    # utilities_matrix[:, math.ceil(3*J/4):math.ceil(J)] *= -1
    # # 20% polarised regime
    # utilities_matrix[:, 0:math.ceil(J/5)] *= -0.25
    # utilities_matrix[:, math.ceil(J/5):math.ceil(2*J/5)] *= -0.9
    # utilities_matrix[:, math.ceil(2*J/5):math.ceil(3*J/5)] *= -1.5
    # utilities_matrix[:, math.ceil(3*J/5):math.ceil(4*J/5)] *= -1
    # utilities_matrix[:, math.ceil(4*J/5):math.ceil(J)] *= -1.8
    # 10% polarised regime
    # utilities_matrix[:, 0:math.ceil(J/10)] *= 1.5
    # utilities_matrix[:, math.ceil(J/10):math.ceil(2*J/10)] *= 1
    # utilities_matrix[:, math.ceil(2*J/10):math.ceil(3*J/10)] *= 0.5
    # utilities_matrix[:, math.ceil(3*J/10):math.ceil(4*J/10)] *= 0.25
    # utilities_matrix[:, math.ceil(4*J/10):math.ceil(5*J/10)] *= -0.25
    # utilities_matrix[:, math.ceil(5*J/10):math.ceil(6*J/10)] *= -0.5
    # utilities_matrix[:, math.ceil(6*J/10):math.ceil(7*J/10)] *= -0.75
    # utilities_matrix[:, math.ceil(7*J/10):math.ceil(8*J/10)] *= -1.0
    # utilities_matrix[:, math.ceil(8*J/10):math.ceil(9*J/10)] *= -1.25
    # utilities_matrix[:, math.ceil(9*J/10):math.ceil(10*J/10)] *= -1.5

    # save data
    pathlib.Path("{}/".format(data_location)).mkdir(parents=True, exist_ok=True)
    with open("{}/Utilities.pickle".format(data_location), "wb") as f:
        pickle.dump(utilities_matrix, f, protocol=4)
    # utilities_matrix = generate_normal_data(n_samples=K, n_dimensions=J, mu=0.6*np.ones((J,)), sigma=0.1*np.eye(J))
    sigma_noise = sigma_e*np.eye(J)
    stochastic_component = generate_normal_data(n_samples=K, n_dimensions=J, mu=mu_e*np.ones((J,)), sigma=sigma_noise, rng=rng)
    pijs_noise = pijs + stochastic_component
    utilities_mat_probab = norm.cdf(pijs_noise, loc=mu_e, scale=sigma_e)
    follow_matrix = utilities_mat_probab > 0.5 

    ##################################################################
    # k_idx = np.argwhere(np.all(follow_matrix<1, axis=1))
    # # uninformative lead users
    # j_idx = np.argwhere(np.all(follow_matrix<1, axis=0))

    # follow_matrix_new = follow_matrix[:, ~np.all(follow_matrix < 1, axis=0)]
    # follow_matrix_new = follow_matrix_new[~np.all(follow_matrix_new < 1, axis=1), :]
    ##################################################################

    with open("{}/Y.pickle".format(data_location), "wb") as f:
        pickle.dump(follow_matrix, f, protocol=4)    
    with open("{}/Utilities_mat_probabilities.pickle".format(data_location), "wb") as f:
        pickle.dump(utilities_mat_probab, f, protocol=4)

    snr = 10*np.log10((K*x_var + J*z_var + J*alpha_var + K*beta_var)/(K*J*sigma_e))
    # print(K, J, sigma_e, snr)
    
    # full, with status quo
    if delta > 0:
        parameter_space_dim = (K+2*J)*d + J + K + 3
    else:
        # no status quo
        parameter_space_dim = (K+J)*d + J + K + 2
    # for distributing per N rows
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    N = math.ceil(parameter_space_dim/J)
    print("Subset row number: {}".format(N))
    print("Observed data points per data split: {}".format(N*J))
    # subset rows (users)   
    Ninit = N
    for nbs in [1]: #range(1, 2, 1):
        N = Ninit*nbs
        print("Subset row number: {}".format(N))
        print("Observed data points per data split: {}".format(N*J))

        for i in range(0, K, N):
            from_row = i 
            to_row = np.min([i+N, K])
            # print(from_row, to_row)
            if i+2*N > K:
                to_row = K
            pathlib.Path("{}/{}/dataset_{}_{}".format(data_location, N, from_row, to_row)).mkdir(parents=True, exist_ok=True)
            with open("{}/{}/dataset_{}_{}/dataset_{}_{}.pickle".format(data_location, N, from_row, to_row, from_row, to_row), "wb") as f:
                pickle.dump(follow_matrix[from_row:to_row, :], f, protocol=4)             
            fig = plot_array_heatmap(
                utilities_matrix[from_row:to_row, :],
                title="Computed utilities",
                colorscale="sunsetdark",
                colorbar=dict(x=0.3, thickness=10, title='U'),            
                xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=True
                )  
            probabfig = plot_array_heatmap(
                utilities_mat_probab[from_row:to_row, :],
                title="Pij CDF matrix",
                colorscale="Viridis", boundcolorscale=True,
                colorbar=dict(thickness=15, title='CDF'),
                xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=True
                )  
            errfig = plot_array_heatmap(
                stochastic_component[from_row:to_row, :],
                title="Error component - SNR = {} dB".format(snr),
                colorbar=dict(thickness=15, title='E'),
                colorscale="blues",
                xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=True
            )    
            followfig = plot_array_heatmap(
                    follow_matrix[from_row:to_row, :].astype(np.int8),
                    title="Following",
                    xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=False
                )    
            try:
                allplots = plot_side_by_side_subplots(fig, followfig, errfig, title="Synthetic data")        
                fix_plot_layout_and_save(allplots, "{}/{}/dataset_{}_{}/utilities_following_relationships.html".format(data_location, N, from_row, to_row), 
                                        xaxis_title="", yaxis_title="", title="Synthetic data", showgrid=False, showlegend=False,
                                        print_png=True, print_html=True, print_pdf=False)
                fix_plot_layout_and_save(probabfig, "{}/{}/dataset_{}_{}/utilities_mat_probab.html".format(data_location, N, from_row, to_row), xaxis_title="", yaxis_title="", title="CDF(Pij)", 
                                    showgrid=False, showlegend=False,
                                    print_png=True, print_html=True, print_pdf=False)    
            except:
                print("Plotting for K = {}, J = {}, batchsize = {} failed.".format(K, J, N))    
            if i+2*N > K:
                break

    # plots
    try:
        fig = plot_array_heatmap(
            utilities_matrix,
            title="Computed utilities",
            colorscale="sunsetdark",
            colorbar=dict(x=0.3, thickness=10, title='U'),
            xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=True
        )    
        
        followfig = plot_array_heatmap(
                follow_matrix.astype(np.int8),
                title="Following",
                xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=False
            )    
        errfig = plot_array_heatmap(
                stochastic_component,
                title="Error component - SNR = {} dB".format(snr),
                colorbar=dict(thickness=15, title='E'),
                colorscale="blues",
                xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=True
            ) 
        probabfig = plot_array_heatmap(
                utilities_mat_probab,
                title="Pij CDF matrix",
                colorscale="Viridis", boundcolorscale=True,
                colorbar=dict(thickness=15, title='CDF'),
                xtitle="Leaders", ytitle="Followers", show_values=False, show_scale=True
                )         
        allplots = plot_side_by_side_subplots(fig, followfig, errfig, title="Synthetic data") 
        fix_plot_layout_and_save(allplots, "{}/utilities_following_relationships.html".format(data_location), xaxis_title="", yaxis_title="", title="Synthetic data", 
                                showgrid=False, showlegend=False,
                                print_png=True, print_html=True, print_pdf=False)
        fix_plot_layout_and_save(probabfig, "{}/utilities_mat_probab.html".format(data_location), xaxis_title="", yaxis_title="", title="CDF(Pij)", 
                                showgrid=False, showlegend=False,
                                print_png=True, print_html=True, print_pdf=False)
    except:
        print("Global plotting for K = {}, J = {} failed.".format(K, J))    

    try:
        if delta > 0:
            fig = create_scatter_plot(
                    data_sets=[(xs[0, :], xs[1, :]), (zs[0, :], zs[1, :]), (phis[0, :], phis[1, :])],
                    labels=["Followers", "Leaders", "Status quo"],
                    colors=["blue", "orange", "green"],
                    sizes=[8, 12, 10],
                    symbols=["circle", "diamond", "star"],
                    title=""
                )
        else:
            fig = create_scatter_plot(
                    data_sets=[(xs[0, :], xs[1, :]), (zs[0, :], zs[1, :])],
                    labels=["Followers", "Leaders"],
                    colors=["blue", "orange"],
                    sizes=[8, 12],
                    symbols=["circle", "diamond"],
                    title=""
                )
        fig.layout.height = 700    
        fix_plot_layout_and_save(fig, "{}/network_users_vis.html".format(data_location), 
                                xaxis_title="", yaxis_title="", title="Ideal points", 
                                showgrid=False, showlegend=False, print_png=True, 
                                print_html=True, print_pdf=False)
    except:
        print("Followers plotting for K = {}, J = {} failed.".format(K, J))    


if __name__ == "__main__":


    seed_value = 1812
    random.seed(seed_value)
    np.random.seed(seed_value)
    check_random_state(seed_value)
    rng = np.random.default_rng()

    # dimensionality of ideal points space
    d = 2
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]
    mu_e = 0
    # utility model parameters
    gamma = 0.01
    delta = 0.0
    # leaders popularity
    alpha_mean = 0.0
    alpha_var = 0.5
    # followers' political interest
    beta_mean = 0.5
    beta_var = 0.5    
    # followers' ideal points
    xs_mean_1 = np.zeros((d,))
    xs_sigma_1 = 1*np.eye(d)
    # leaders' ideal points - unimodal distribution
    zs_mean_1 = np.zeros((d,))
    zs_sigma_1 = 1*np.eye(d)
    # utility and distance functions
    distance_func = lambda x,y : np.sum((x-y)**2)
    utility_func = lambda x : x

    # Generate synthetic data
    # trials
    M = 10
    # number of leaders
    Js = [50, 500, 1000]  #[100] #, 500, 1000] 50, 500,  
    # number of followers
    Ks = [10000] #, 50000, 100000]    
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]

    # Fixing polarity for leaders and very the noise level depending on hierarchy: lower indices -> less noise
    polarity = np.random.choice(np.asarray([-1,1]), size=Js[-1], replace=True, p=[0.5, 0.5])
    for K in Ks:        
        for J in Js:
            if J==50:
                sigma_es = [1]                
            elif J==500:
                sigma_es = [3]                
            elif J==1000:
                sigma_es = [5]                
            else:
                sigma_es = []
                continue
            for sigma_e in sigma_es:                           
                snr = 10*np.log10((K*xs_sigma_1[0, 0] + J*zs_sigma_1[0, 0] + J*alpha_var + K*beta_var)/(K*J*sigma_e))
                print(K, J, sigma_e, snr)
                
                parameter_space_dim = (K+J)*d + J + K + 2                              

                alpha_js = polarity[:J]
                beta_is = generate_normal_data(n_samples=K, n_dimensions=1, mu=beta_mean, sigma=beta_var, rng=rng)

                # followers' ideal points
                # K x d
                xs = generate_normal_data(n_samples=K, n_dimensions=d, mu=xs_mean_1, sigma=xs_sigma_1, rng=rng)
                xs = xs.transpose()
                # leaders' ideal points - unimodal distribution
                zs1 = generate_normal_data(n_samples=J, n_dimensions=d, mu=zs_mean_1, sigma=zs_sigma_1, rng=rng)
                zs = zs1.transpose()

                theta, param_positions_dict = params2optimisation_dict(J, K, d, parameter_names, xs, zs, None, alpha_js, beta_is, gamma, delta, sigma_e)    
                theta = np.asarray(theta)
                params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
                X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                     
                Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")      
                alpha1 = params_hat["alpha"]
                beta1 = params_hat["beta"]
                gamma1 = params_hat["gamma"]      
                assert(np.allclose(X, xs))
                assert(np.allclose(Z, zs))
                assert(np.allclose(alpha_js, alpha1))
                assert(np.allclose(beta_is, beta1))
                assert(gamma1==gamma)

                for m in range(M):
                    print(m)
                    data_location = "/tmp/idealdata_expIII/data_K{}_J{}_sigmae{}/{}/".format(K, J, str(sigma_e).replace(".", ""), m)                    
                    generate_trial_data(parameter_names, m, J, K, d, distance_func, utility_func, data_location, param_positions_dict, theta, x_var=xs_sigma_1[0,0], z_var=zs_sigma_1[0,0], 
                                        alpha_var=alpha_var, beta_var=beta_var, debug=False, rng=rng)
                    time.sleep(1)

                    # Save parameters
                    parameters = dict()
                    parameters["J"] = J
                    parameters["K"] = K
                    parameters["mu_e"] = 0
                    parameters["sigma_e"] = sigma_e
                    parameters["gamma"] = gamma
                    parameters["delta"] = delta
                    parameters["alpha"] = alpha_js.tolist()
                    parameters["alpha_mean"] = alpha_mean
                    parameters["alpha_cov"] = alpha_var
                    parameters["beta"] = beta_is.tolist()
                    parameters["beta_mean"] = beta_mean
                    parameters["beta_cov"] = beta_var
                    parameters["d"] = d
                    if d > 1:
                        parameters["xs_mean_1"] = xs_mean_1.tolist()
                        parameters["xs_sigma_1"] = xs_sigma_1.reshape((d*d,), order="F").tolist()
                        parameters["zs_mean_1"] = zs_mean_1.tolist()
                        parameters["zs_sigma_1"] = zs_sigma_1.reshape((d*d,), order="F").tolist()
                        # if "phis" in parameter_names:
                        #     parameters["phis_mean_1"] = phis_mean_1.tolist()
                        #     parameters["phis_sigma_1"] = phis_sigma_1.reshape((d*d,), order="F").flatten().tolist()
                        # parameters["xs_mean_2"] = xs_mean_2.tolist()
                        # parameters["xs_sigma_2"] = xs_sigma_1.reshape((d*d,), order="F").flatten().tolist()
                        # parameters["zs_mean_2"] = zs_mean_2.tolist()
                        # parameters["zs_sigma_2"] = zs_sigma_2.reshape((d*d,), order="F").flatten().tolist()    
                        # parameters["phis_mean_2"] = phis_mean_2.tolist()
                        # parameters["phis_sigma_2"] = phis_sigma_2.reshape((d*d,), order="F").flatten().tolist()
                    else:
                        parameters["xs_mean_1"] = xs_mean_1
                        parameters["xs_sigma_1"] = xs_sigma_1
                        parameters["zs_mean_1"] = zs_mean_1
                        parameters["zs_sigma_1"] = zs_sigma_1
                        # parameters["phis_mean_1"] = phis_mean_1
                        # parameters["phis_sigma_1"] = phis_sigma_1

                    parameters["Z"] = zs.reshape((d*J,), order="F").flatten().tolist()
                    parameters["X"] = xs.reshape((d*K,), order="F").flatten().tolist()
                    # parameters["Phi"] = phis.reshape((d*J,), order="F").flatten().tolist()
                    DATA_dir = data_location
                    pathlib.Path(DATA_dir).mkdir(parents=True, exist_ok=True)     
                    with jsonlines.open("{}/synthetic_gen_parameters.jsonl".format(DATA_dir), "a") as f:
                        f.write(parameters)
