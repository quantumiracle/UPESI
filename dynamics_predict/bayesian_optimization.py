'''
System identification (embedding) according to trajectories in simulation and reality with Bayesian Optimization (BO) method (non-parametric optimization).
BO reference: https://github.com/fmfn/BayesianOptimization
'''
import numpy as np
from matplotlib import pyplot as plt
import torch
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from environment import envs
from dynamics_predict.defaults import DYNAMICS_PARAMS, HYPER_PARAMS
from bayes_opt import BayesianOptimization
from dynamics_predict.dynamics_networks import DynamicsNetwork


class EmbeddingBayesianOptimization():
    def __init__(self, Env_name, real_data_path, dynamics_model_path):
        self.x = torch.Tensor(np.load(real_data_path+'sa.npy'))
        self.y = torch.Tensor(np.load(real_data_path+'s_.npy'))
        env = envs[Env_name]()
        self.dynamics_model = DynamicsNetwork(state_space=env.observation_space, action_space=env.action_space, num_hidden_layers=6, param_dim=HYPER_PARAMS['latent_dim'])
        self.dynamics_model.load_state_dict(torch.load(dynamics_model_path))

    def get_params_bound(self,):
        v_limit = 0.1
        bounds = {'alpha_{}'.format(i): [-v_limit, v_limit] for i in range(HYPER_PARAMS['latent_dim'])}
        return bounds
    
    def black_box_function(self, **alpha):
        """Function with unknown internals we wish to maximize.
        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        x=torch.cat((self.x, torch.Tensor(list(alpha.values())).repeat(self.x.shape[0],1)),axis=-1)
        y_=self.dynamics_model(x).detach().cpu().numpy()

        loss = np.square(np.subtract(self.y, y_)).mean()

        return -loss  # BO gives the maximum, so negagive loss


    def optimize(self, random_init=20, n_iter=500, result_path='params_bo.npy'):
        params_bound = self.get_params_bound()

        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=params_bound,  # Here it defines the parameters to be randomized, which should also match with input arguments of black_box_function()
            verbose=2,  # verbose =2 print each iteration, verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(
            init_points=random_init,
            n_iter=n_iter,
        )

        print(optimizer.max['params'])
        optimized_params = optimizer.max['params']
        np.save(result_path, [params_bound, optimized_params])


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    Env_name = 'pandapushik2dsimple'
    Type = ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork', 'VAEDynamicsNetwork'][2]
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    param_dim = len(DYNAMICS_PARAMS[Env_name+'dynamics'])

    dynamics_model_path = path+'/data/dynamics_data/model/{}_dim{}/dynamics'.format(Type, str(param_dim))

    real_data_path = path+'/data/real/processed_traj/'
    ebo = EmbeddingBayesianOptimization(Env_name, real_data_path, dynamics_model_path)
    ebo.optimize()
