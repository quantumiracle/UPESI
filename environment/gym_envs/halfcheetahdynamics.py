# FROM https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py

import numpy as np
import gym.spaces as spaces
from .param_wrapper import MujocoEnvWithParams, EzPickle
import os
import os
from .halfcheetah import HalfCheetahEnv
from robosuite.class_wrappers import latent_dynamics_provider
from dynamics_predict.defaults import DYNAMICS_PARAMS, HYPER_PARAMS


with open(os.path.join(os.path.dirname(__file__), 'assets/halfcheetah/model.template.xml')) as f:
    template = f.read()

params_to_attach = DYNAMICS_PARAMS['halfcheetahdynamics']
_HalfCheetahDynamics = latent_dynamics_provider(HalfCheetahEnv, params_to_attach=params_to_attach)

class HalfCheetahDynamicsEnv(_HalfCheetahDynamics):
    name = 'halfcheetahdynamics'
    dynamics_norm = False # whether the encoder is trained with normalized dynamics parameters as input
    ori_obs_dim = HalfCheetahEnv.observation_space.shape[0]
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

    obs_dim = ori_obs_dim+len(params_to_attach)
    observation_space = spaces.Box(np.zeros(obs_dim, dtype=np.float32), np.zeros(obs_dim, dtype=np.float32))
    
    def __init__(self, norm_factor_path='/data/dynamics_data/halfcheetah/norm_factor.npy', **kwargs):
        super().__init__(**kwargs)
        if norm_factor_path:
            self.dynamics_norm = True
            [self.norm_mean, self.norm_std] = np.load(self.path+norm_factor_path)
        else:
            self.dyanmics_norm = False

    def norm(self, param):
        param = (param - self.norm_mean)/self.norm_std
        return param

    def step(self, action,  given_params=None):
        obs, reward, done, info = super().step(action)
        if given_params is not None:
            params = given_params # no need to normalize
        else:
            params = info['dynamics_params']
            if self.dynamics_norm:
                params = self.norm(params)

        return np.concatenate((obs, params), -1), reward, done, info

    def reset(self, given_params=None, **kwargs):
        obs, info = super().reset(**kwargs)
        if given_params is not None:
            params = given_params  # no need to normalize

        else:
            params = info['dynamics_params']
            if self.dynamics_norm:
                params = self.norm(params)

        return np.concatenate((obs, params), -1)
