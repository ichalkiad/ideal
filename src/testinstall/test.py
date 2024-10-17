import os

from cmdstanpy import CmdStanModel

stan_file = os.path.join('./src/testinstall/', 'bernoulli.stan')

model = CmdStanModel(stan_file=stan_file)

print(model)

print(model.exe_info())

data_file = os.path.join('./src/testinstall/', 'bernoulli.data.json')

fit = model.sample(data=data_file)

print(fit.stan_variable('theta'))
print(fit.draws_pd('theta')[:3])
print(fit.draws_xr('theta'))

for k, v in fit.stan_variables().items():
    print(f'{k}\t{v.shape}')
for k, v in fit.method_variables().items():
    print(f'{k}\t{v.shape}')

print(f'numpy.ndarray of draws: {fit.draws().shape}')

fit.draws_pd()

print(fit.metric_type)
print(fit.metric)
print(fit.step_size)

print(fit.metadata.cmdstan_config['model'])
print(fit.metadata.cmdstan_config['seed'])

fit.summary()
print(fit.diagnose())

