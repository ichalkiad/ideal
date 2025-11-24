## Efficient computational methods for the estimation of ideal points in social sciences

A compact codebase for estimating ideal-point models with multiple estimation algorithms (Correspondence Analysis, Iterated Conditional Modes with posterior power and data annealing, distributed Maximum Likelihood) and utilities for synthetic data generation, aggregation and plotting. The codebase accompanies the paper:

*Efficient computational methods for the estimation of ideal points in social sciences*, by Dr Ioannis Chalkiadakis, Prof. Dr Gareth W. Peters, and Dr Pedro Ramaciotti.

Estimation of ideological positions among voters, legislators, social media users, and other actors is central to many subfields in social sciences. The use of a particular class of Item-Response models, referred to as 'ideal points' models, uses a wide range of collective choice data: roll call votes, co-sponsorship records, engagement data from social media, and word occurrences in speech. Inference in ideal points models typically suffers from model identification, unknown robustness against noise, and data size and sparsity issues, while ideological position measurements inherently lack ground truth data. These challenges have led to a diversity of approaches, from Markov Chain Monte Carlo (MCMC), which is notoriously computationally costly for large empirical settings, to fast factor analysis approximations lacking a priori error estimates. In this paper, we formulate a unifying framework for a large family of ideal point methods across disciplines and examine their computational complexity and accuracy against noisy data, leveraging synthetic scenarios, and introducing Iterated Conditional Modes, an estimation method from the signal processing literature, for Item-Response models - for the first time, to the best of our knowledge. Finally, we illustrate the applicability of our framework in a multi-dimensional ideal points inference problem for ~20,000,000 X (previously Twitter) users in 3 countries (France, the UK, US) along several ideology and issue dimensions calibrated with survey data.

Key contents
- Python package: idealpestimation (code and command-line scripts in idealpestimation/src)
- src/ contains:
  - core estimation modules: implementations of estimation algorithms and related helpers;
  - data & DB helpers: utils.py contains connectors, table and matrix utilities used across scripts;
  - scripts: results aggregation and plotting scripts;
  - environment provisioning: singularity_ideal.def for container builds and environment reproducibility;  
  - scheduler wrappers and estimators:
    - ca_slurm.py — Slurm-oriented wrapper to run CA estimations at scale: prepares jobs, sets resource parameters and collects results.
    - icm_annealing_*.py — ICM (Iterated Conditional Modes) variants that include annealing schedules on the posterior power and data; multiple files implement different annealing strategies and configuration entry points.
    - mle_slurm.py - Slurm-oriented wrapper for MLE estimation runs: batching, resource configuration and job orchestration for maximum-likelihood estimation experiments.

Quick start
1. Create the environment and install editable package:
   - Conda: use environment_ideal.yml
   - Singularity (for cluster runs): see idealpestimation/src/singularity_ideal.def
2. Install the package:
   pip install -e .


