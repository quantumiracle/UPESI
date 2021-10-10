# FROM https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py

import numpy as np
import torch
import gym.spaces as spaces
from .param_wrapper import MujocoEnvWithParams, EzPickle
import os
from .halfcheetah import HalfCheetahEnv
from robosuite.class_wrappers import latent_dynamics_provider
from dynamics_predict.defaults import DYNAMICS_PARAMS, HYPER_PARAMS
from dynamics_predict.dynamics_networks import DynamicsEncoder, DynamicsVariationalEncoder


with open(os.path.join(os.path.dirname(__file__), 'assets/halfcheetah/model.template.xml')) as f:
    template = f.read()

params_to_attach = DYNAMICS_PARAMS['halfcheetahdynamics']
_HalfCheetahDynamics = latent_dynamics_provider(HalfCheetahEnv, params_to_attach=params_to_attach)

class HalfCheetahDynamicsEmbeddingEnv(_HalfCheetahDynamics):
    name = 'halfcheetahdynamicsembedding'
    dynamics_norm = False # whether the encoder is trained with normalized dynamics parameters as input
    EmbeddingDynamicsNetworkType = ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork', 'VAEDynamicsNetwork'][2]
    ori_obs_dim = HalfCheetahEnv.observation_space.shape[0]
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

    obs_dim = ori_obs_dim+HYPER_PARAMS['halfcheetahdynamics']['latent_dim']
    observation_space = spaces.Box(np.zeros(obs_dim, dtype=np.float32), np.zeros(obs_dim, dtype=np.float32))

    #  the following cannot be put in __init__()
    if EmbeddingDynamicsNetworkType in ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork']:  # normal encoder
        encoder = DynamicsEncoder(param_dim=len(DYNAMICS_PARAMS['halfcheetahdynamics']), latent_dim=HYPER_PARAMS['halfcheetahdynamics']['latent_dim'])  # latent dimension needs to align with the trained DynamicsEncoder 
    elif EmbeddingDynamicsNetworkType == 'VAEDynamicsNetwork': # varmiational auto-encoder
        encoder = DynamicsVariationalEncoder(param_dim=len(DYNAMICS_PARAMS['halfcheetahdynamics']), latent_dim=HYPER_PARAMS['halfcheetahdynamics']['latent_dim'])  # latent dimension needs to align with the trained DynamicsEncoder 
    try:
        # print(path+'/data/dynamics_data/{}/model/{}_dim{}/encoder'.format('halfcheetah', EmbeddingDynamicsNetworkType, str(len(DYNAMICS_PARAMS['halfcheetahdynamics']))))
        encoder.load_state_dict(torch.load(path+'/data/dynamics_data/{}/model/{}_dim{}/encoder'.format('halfcheetah', EmbeddingDynamicsNetworkType, str(len(DYNAMICS_PARAMS['halfcheetahdynamics'])))))
        encoder.eval()
        # print(encoder)
    except: 
        pass
        # print("Error: encoder for {} not found!".format(name))
    alpha=None

    def __init__(self, norm_factor_path='/data/dynamics_data/halfcheetah/norm_factor.npy', **kwargs):
        super().__init__(**kwargs)
        if norm_factor_path:
            self.dynamics_norm = True
            [self.norm_mean, self.norm_std] = np.load(self.path+norm_factor_path)
        else:
            self.dyanmics_norm = False

    def encode(self, param):
        if self.dynamics_norm:
            param = (param - self.norm_mean)/self.norm_std
        param = torch.FloatTensor([param])
        if self.encoder:
            if self.EmbeddingDynamicsNetworkType in ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork']:
                alpha = self.encoder(param).detach().cpu().numpy()[0]
            elif self.EmbeddingDynamicsNetworkType == 'VAEDynamicsNetwork':
                mu, logvar = self.encoder(param)
                alpha = mu.detach().cpu().numpy()[0]
        else:
            print('No such type: {}'.format(self.EmbeddingDynamicsNetworkType))
            alpha = param
        return alpha

    def step(self, action, given_alpha=None):
        obs, reward, done, info = super().step(action)

        if self.encoder:
            if given_alpha:
                self.alpha = given_alpha
            elif self.alpha is None:
                self.alpha = self.encode(info['dynamics_params'])
            return np.concatenate((obs, self.alpha), -1), reward, done, info
        else:
            # print(self.alpha)
            return np.concatenate((obs, info['dynamics_params']), -1), reward, done, info

    def reset(self, given_alpha=None, **kwargs):
        obs, info = super().reset(**kwargs)
        if self.encoder:
            if given_alpha:
                self.alpha = given_alpha
            else:
                self.alpha = self.encode(info['dynamics_params'])
            return np.concatenate((obs, self.alpha), -1)
        else:
            return np.concatenate((obs, info['dynamics_params']), -1)