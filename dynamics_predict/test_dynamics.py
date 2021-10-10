import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from dynamics_networks import DynamicsNetwork, DynamicsParamsOptimizer, EncoderDynamicsNetwork, EncoderDecoderDynamicsNetwork, VAEDynamicsNetwork
from rl.policy_networks import DPG_PolicyNetwork
from utils.load_params import load_params
from utils.common_func import rand_params
from environment import envs
from defaults import DYNAMICS_PARAMS, HYPER_PARAMS

# torch.manual_seed(1234)  # Reproducibility
# np.random.seed(1234)
# random.seed(1234)


def load_policy(env, load_from, params, policy_class=DPG_PolicyNetwork):
    """
    Load a model-free reinforcement learning policy.
    """
    policy = policy_class(env.observation_space, env.action_space, params['hidden_dim'], params['action_range'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if load_from:
        policy.load_state_dict(torch.load(load_from, map_location=device))
        print(policy)
    return policy

def param_norm(param, norm_mean, norm_std):
    return (param-norm_mean)/norm_std

def test_dynamics(Env, norm_factor=None, Type = 'EncoderDynamicsNetwork', load_policy_from=None, load_model_from=None, save_to=None, episodes=50, env_settings={}, default_params={},):
    """
    Test the trained dynamics prediction and dynamics embedding networks.
    """
    print(Type)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = Env(**env_settings, **default_params)
    params_key  = ['max_steps', 'hidden_dim', 'action_range']
    params = {k:v for k,v in zip (params_key, load_params('td3', env.__class__.__name__.lower(), params_key))}
    policy = load_policy(env, load_policy_from, params)
    model = eval(Type)(env.observation_space, env.action_space, param_dim, latent_dim=HYPER_PARAMS[Env.name]['latent_dim'])
    if load_model_from:
        model.load_model(load_model_from)
        print('Load dynamics model from {}'.format(load_model_from))

    data = []

    for ep in range(episodes):
        print(ep)
        param_dict, param_vec = rand_params(env, DYNAMICS_PARAMS[Env.name+'dynamics'])
        param = param_norm(param_vec, *norm_factor)
        s = env.reset(**param_dict)
        pre_s_ = s
        diff_list =[]
        relative_diff_list = []
        for step in range(params['max_steps']):
            a = policy.get_action(s, noise_scale=0.0)
            s_, r, d, _ = env.step(a)
            # sa = np.concatenate((s,a))  # single step forward prediction
            sa = np.concatenate((pre_s_,a))  # consecutive forward prediction
            _, pre_s_ = model([sa], [param])
            pre_s_ = pre_s_.detach().cpu().numpy()[0]
            # env.render()
            s=s_
            # print('Compare: simulator{} \n prediction{}'.format(s_, pre_s_.detach().cpu().numpy()[0]))
            diff = ((s_ - pre_s_)**2).mean()
            relative_diff = diff/(s_**2).mean()
            # print("Step: {} Difference: {} Relative Difference: {}".format(step, diff, relative_diff))
            diff_list.append(diff)
            relative_diff_list.append(relative_diff)

        data.append([diff_list, relative_diff_list])
    
    if save_to: 
        os.makedirs(save_to, exist_ok=True)
        np.save(save_to+'/{}_dynamics_test.npy'.format(Type), data)

def test_dynamics_offline(Env, data_path, norm_factor=None, Type = 'EncoderDynamicsNetwork', load_policy_from=None, load_model_from=None, ):
    print(Type)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = Env()
    model = eval(Type)(env.observation_space, env.action_space, param_dim, latent_dim=HYPER_PARAMS[Env.name]['latent_dim'])
    if load_model_from:
        model.load_model(load_model_from)
        print('Load dynamics model from {}'.format(load_model_from))

    data = np.load(data_path, allow_pickle=True)
    data = data.tolist()

    x_test, y_test = zip(*random.sample(data, min(len(data), 12800)))
    s,a,param = np.stack(x_test, axis=1) 
    param = np.vstack(param)
    sa = torch.FloatTensor(np.concatenate((np.vstack(s),np.vstack(a)), axis=-1))   
    param = torch.FloatTensor(param)            
    s_ = torch.FloatTensor(y_test).to(device)
    pre_param, pre_s_ = model(sa, param)
    loss1_test= model.loss_dynamics(pre_s_, s_).item()
    print('Test loss: ', loss1_test)


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    Env = envs['pandapushik2dsimple']
    param_dim = len(DYNAMICS_PARAMS[Env.name+'dynamics'])
    print('current path: {}, env: {}, parameters dimension: {}'.format(path,Env,param_dim))
    Type = ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork', 'VAEDynamicsNetwork'][0]
    dynamics_norm = True # whether the encoder is trained with normalized dynamics parameters as input
    if dynamics_norm:
        norm_factor = np.load(path+'/data/dynamics_data/norm_factor.npy')
    else:
        norm_factor = None

    test_dynamics(Env, Type = Type, norm_factor=norm_factor, load_policy_from=path+'/data/weights/20201209_1457/2500'+'_td3_policy',
    load_model_from=path+'/data/dynamics_data/model/{}_dim{}/'.format(Type, str(param_dim)), save_to=path+'/data/dynamics_data'
    )

    # test_dynamics_offline(Env, data_path=path+'/data/dynamics_data/norm_dynamics.npy', 
    # Type = Type, norm_factor=norm_factor, load_policy_from=path+'/data/weights/20201209_1457/2500'+'_td3_policy',
    # load_model_from=path+'/data/dynamics_data/model/{}_dim{}/'.format(Type, str(param_dim))
    # )
