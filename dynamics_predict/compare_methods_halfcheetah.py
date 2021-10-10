""" Compare several different methods for domain transfer """ 
import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.abspath(os.path.join(os.getcwd(),".."))
print(path)
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from dynamics_networks import DynamicsNetwork, SINetwork, DynamicsParamsOptimizer, EncoderDynamicsNetwork, EncoderDecoderDynamicsNetwork, VAEDynamicsNetwork
from rl.policy_networks import DPG_PolicyNetwork
from utils.load_params import load_params
from utils.common_func import rand_params
from defaults import DYNAMICS_PARAMS, HYPER_PARAMS
from environment import envs
from bayes_opt import BayesianOptimization
import torch
from test_dynamics import load_policy


class EmbeddingBayesianOptimization():
    def __init__(self, Env_name, data_path, dynamics_model_path):
        self.x=sa
        self.y=s_
#         self.x = tocrch.Tensor(np.load(data_path+'sa.npy'))
#         self.y = torh.Tensor(np.load(data_path+'s_.npy'))
        env = envs[Env_name]()
        self.env_name = Env_name
        self.dynamics_model = DynamicsNetwork(state_space=env.observation_space, action_space=env.action_space, \
                                              num_hidden_layers=6, param_dim=HYPER_PARAMS[self.env_name+'dynamics']['latent_dim'])
        self.dynamics_model.load_state_dict(torch.load(dynamics_model_path))

    def get_params_bound(self,):
        v_limit = 2.
        bounds = {'alpha_{}'.format(i): [-v_limit, v_limit] for i in range(HYPER_PARAMS[self.env_name+'dynamics']['latent_dim'])}
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
            verbose=0,  # verbose =2 print each iteration, verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        
#         optimizer.probe(  # probe a certain datapoint at the beginning
#         params={"alpha_0": -1.0045, "alpha_1": 0.3682}, lazy=True)

        optimizer.maximize(
            init_points=random_init,
            n_iter=n_iter,
        )

        # print(optimizer.max['params'])
        optimized_params = optimizer.max['params']
        optim_records = [optimized_params, params_bound, optimizer.res]
        np.save(result_path+'optim_records.npy', optim_records)
        return list(optimized_params.values())

def rollout_policy(env_name, dynamics_params, optimized_embedding=None, episodes=10, load_policy_from=None, env_settings={}, default_params={}):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    Env = envs[env_name]
    env = Env(**env_settings, **default_params)
    params_key  = ['max_steps', 'hidden_dim', 'action_range']
    params = {k:v for k,v in zip (params_key, load_params('td3', env.__class__.__name__.lower(), params_key))}
    if not params['action_range']:
        params['action_range'] = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric
    policy = load_policy(env, load_policy_from, params)

    data = []

    for ep in range(episodes):
        print(ep)
        param_dict = env.params_dict
        param_dict = {k: new_v for (k,v),new_v in zip(param_dict.items(), dynamics_params)}
        if optimized_embedding is not None:
            state = env.reset(optimized_embedding, **param_dict)
        else:
            state = env.reset(**param_dict)
        episode_reward = 0

        for step in range(params['max_steps']):
            action = policy.get_action(state, noise_scale=0.0)
            if optimized_embedding is not None:
                next_state, reward, done, _ = env.step(action, optimized_embedding)
            else:
                next_state, reward, done, _ = env.step(action)
            # env.render() 
            episode_reward += reward
            state=next_state
            if done:
                break

        data.append(episode_reward)

    return data    


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

    model_list = {
                'no_dr': '20210204_112549', 
                # 'dr': '20210226_114251', 
                # 'up': '20210225_20426',
                # 'up_si': '20210225_20426',
                # 'up_embedding': '20210301_16092',
                # 'up_embedding_true': '20210301_16092',  # no BO but get the embedding through encoder given true parameters
    }

    env_name = 'halfcheetah'
    frame_stack = 5

    data_path = path+'/data/dynamics_data/'+env_name+'/test_dynamics.npy'
    param_dim = len(DYNAMICS_PARAMS[env_name+'dynamics'])
    print('parameter dimension: ', param_dim)

    test_data = np.load(data_path, allow_pickle=True)
    print('number of samples in test data: ', len(test_data))

    log=model_list.copy()

    # load normalization factors
    [s_norm_mean, s_norm_std] = np.load(path+'/data/dynamics_data/{}/norm_factor_s.npy'.format(env_name))
    [s__norm_mean, s__norm_std] = np.load(path+'/data/dynamics_data/{}/norm_factor_s_.npy'.format(env_name))
    s_dim = envs[env_name].observation_space.shape[0]

    for name, model in model_list.items():
        data=[]
        print('Evaluate method: ', name)
        for idx in range(10):  # index of sample to test: 0-10
            print('Param No.{}: \n ---------------------'.format(idx))
            params = test_data[idx]['params'] # true parameters
            print('true parameters: ', params)

            if name == 'up_embedding':
                sa = test_data[idx]['sa']
                s_ = test_data[idx]['s_']
                # normalize s and s_
                sa = np.vstack(sa)
                sa = np.concatenate([(sa[:, :s_dim] - s_norm_mean)/s_norm_std, sa[:, s_dim:]], axis=-1)
                s_ = (s_ - s__norm_mean)/s__norm_std
                
                sa = torch.Tensor(sa)
                s_ = torch.Tensor(s_)
                print(sa.shape, s_.shape)
                Type = ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork', 'VAEDynamicsNetwork'][2]
                dynamics_model_path = path+'/data/dynamics_data/{}/model/{}_dim{}/dynamics'.format(env_name, Type, str(param_dim))
                data_path = path+'/data/dynamics_data/'+env_name+'/test_dynamics.npy'
                ebo = EmbeddingBayesianOptimization(env_name, data_path, dynamics_model_path)
                optimized_embedding = ebo.optimize(result_path='../data/compare_methods/{}/{}_'.format(env_name, idx))
                env_name_ = env_name+'dynamicsembedding'
            elif name == 'up_embedding_true':
                env_name_ = env_name+'dynamicsembedding'
                optimized_embedding = None
            elif name == 'up':
                env_name_ = env_name+'dynamics'
                optimized_embedding = None
            elif name == 'up_si': # offline SI
                env_name_ = env_name+'dynamics'
                env = envs[env_name]()
                SImodel = SINetwork(env.observation_space, env.action_space, param_dim)
                sa = test_data[idx]['sa']
                optimized_embedding = SImodel.forward(torch.FloatTensor(np.array(sa[:frame_stack]).reshape(-1)).unsqueeze(0))[0].detach().numpy()
            else:
                optimized_embedding = None
                env_name_ = env_name

            model_id='24999'
            rewards = rollout_policy(env_name_, params, optimized_embedding, load_policy_from=path+'/data/weights/{}/{}_td3_policy'.format(model, model_id))
            data.append(rewards)
        log[name]=data
    np.save('../data/compare_methods/{}/compare_log4.npy'.format(env_name), log)