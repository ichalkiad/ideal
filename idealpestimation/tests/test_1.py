from idealpestimation.src.mle import aux_test_parameter_estimation


def test_1():
    """
    Run tests for different distributions and sample sizes
    """
    print("Maximum Likelihood Estimation Tests:")
    
    # # Test Poisson distribution
    # print("\nPoisson Distribution Test:")
    # poisson_results = aux_test_parameter_estimation(
    #     distribution_type='poisson', 
    #     sample_size=5000, 
    #     num_trials=20
    # )
    # print("Poisson Lambda MSE:", poisson_results.get('poisson', 'N/A'))
    # assert(poisson_results["poisson"] < 0.02)
    # print(poisson_results)
    
    # Test Normal distribution
    print("\nNormal Distribution Test:")
    normal_results = aux_test_parameter_estimation(
        distribution_type='normal', 
        sample_size=5000, 
        num_trials=200
    )
    print("Normal Mu MSE:", normal_results.get('normal_mu', 'N/A'))
    print("Normal Sigma MSE:", normal_results.get('normal_sigma', 'N/A'))
    assert(normal_results["normal_mu"] < 0.005)
    assert(normal_results["normal_sigma"] < 0.005)

    print(normal_results)

if __name__ == "__main__":
    test_1()