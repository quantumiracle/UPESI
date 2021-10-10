"""

Proximal Policy Optimization (PPO) version 2
----------------------------
1 actor and 1 critic
Old policy is given by previous actor policy before updating.
Batch size can be larger than episode length, only update when batch size is reached,
therefore the trick of increasing batch size for stabilizing training can be applied.

"""
import numpy as np
import torch
import torch.nn as nn

from rl.value_networks import ValueNetwork
from rl.policy_networks import PPO_PolicyNetwork
from utils.load_params import load_params
from utils.common_func import rand_params

from mujoco_py import MujocoException

###############################  PPO  ####################################

class PPOContinuous_Trainer(nn.Module):
    """
    PPO class
    """

    def __init__(self, state_space, action_space, hidden_dim, a_lr, c_lr, \
        action_range, method='clip', reward_bias=0., reward_normalization=False, gamma=0.99, \
        a_update_itr=10, c_update_itr=10, penalty_kl_target=0.01, penalty_lambda=0.5, clip_epsilon=0.2, machine_type='gpu'):
        super(PPOContinuous_Trainer, self).__init__()

        self.reward_bias = reward_bias
        self.reward_normalization = reward_normalization
        self.gamma = gamma
        self.a_update_itr = a_update_itr
        self.c_update_itr = c_update_itr
        self.machine_type = machine_type
        
        self.actor = PPO_PolicyNetwork(state_space, action_space, hidden_dim, action_range)
        self.critic = ValueNetwork(state_space, hidden_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=a_lr) 
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c_lr)  

        self.method = method
        if method == 'penalty':
            self.kl_target = penalty_kl_target
            self.lam = penalty_lambda
        elif method == 'clip':
            self.epsilon = clip_epsilon

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []

    def share_memory(self):
        super().share_memory()
        # implements `def ShareParameters(adamoptim):` in Zihan's multiprocessing implementation
        for adamoptim in [self.actor_opt, self.critic_opt]:
            for group in adamoptim.param_groups:
                for p in group['params']:
                    state = adamoptim.state[p]
                    # initialize: have to initialize here, or else cannot find
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    # share in memory
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()

    def to_cuda(self):
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()

    def a_train(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        mu, sigma = self.actor(state)
        pi = torch.distributions.Normal(mu, sigma)
        ratio = torch.exp(pi.log_prob(action) - old_pi.log_prob(action))
        surr = ratio * adv
        if self.method == 'penalty':
            kl = torch.distributions.kl_divergence(old_pi, pi)
            kl_mean = kl.mean()
            aloss = -(surr - self.lam * kl).mean()
        else:  # clipping method, find this is better
            aloss = -torch.mean(
                torch.min(
                    surr,
                    torch.clamp(
                        ratio,
                        1. - self.epsilon,
                        1. + self.epsilon
                    ) * adv
                )
            )
        self.actor_opt.zero_grad()
        aloss.backward()
        self.actor_opt.step()

        if self.method == 'kl_pen':
            return kl_mean

    def c_train(self, cumulative_r, state):
        """
        Update actor network
        :param cumulative_r: cumulative reward batch
        :param state: state batch
        :return: None
        """
        advantage = cumulative_r - self.critic(state)
        closs = (advantage ** 2).mean()
        self.critic_opt.zero_grad()
        closs.backward()
        self.critic_opt.step()

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        if self.machine_type == 'gpu': 
            s = torch.Tensor(self.state_buffer).cuda()
            a = torch.Tensor(self.action_buffer).cuda()
            r = torch.Tensor(self.cumulative_reward_buffer).cuda()
        else:
            s = torch.Tensor(self.state_buffer)
            a = torch.Tensor(self.action_buffer)
            r = torch.Tensor(self.cumulative_reward_buffer) 

        with torch.no_grad():
            mean, std = self.actor(s)
            pi = torch.distributions.Normal(mean, std)
            adv = r - self.critic(s)

        if self.reward_normalization:
            if hasattr(self, 'rn_mean'):
                self.rn_mean = self.rn_mean * 0.9 + adv.mean() * 0.1
                self.rn_std = self.rn_std * 0.9 + adv.std() * 0.1
            else:
                self.rn_mean = adv.mean()
                self.rn_std = adv.std()
            adv = (adv - self.rn_mean) / (self.rn_std + 1e-6)  # prevent numerical issue

        # update actor
        if self.method == 'kl_pen':
            for _ in range(self.a_update_itr):
                kl = self.a_train(s, a, adv, pi)
                if kl > 4 * self.kl_target:  # this in in google's paper
                    break
            if kl < self.kl_target / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
            self.lam = np.clip(
                self.lam, 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(self.a_update_itr):
                self.a_train(s, a, adv, pi)

        # update critic
        for _ in range(self.c_update_itr):
            self.c_train(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def choose_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        s = torch.Tensor(s)
        mean, std = self.actor(s)
        if greedy:
            a = mean.cpu().detach().numpy()[0]
        else:
            pi = torch.distributions.Normal(mean, std)
            a = pi.sample().cpu().numpy()[0]
        return np.clip(a, -self.actor.action_range, self.actor.action_range)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            if self.machine_type == 'gpu': 
                v_s_ = self.critic(torch.Tensor([next_state]).cuda()).cpu().detach().numpy()[0, 0]
            else:
                v_s_ = self.critic(torch.Tensor([next_state])).detach()[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + self.gamma * v_s_   # no future reward if next state is terminal
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path+'_a')
        torch.save(self.critic.state_dict(), path+'_c')

    def load_model(self, path):
        device = 'cuda:0' if torch.cuda.is_available() and self.machine_type == 'gpu' else 'cpu'
        self.actor.load_state_dict(torch.load(path+'_a', map_location=device))
        self.critic.load_state_dict(torch.load(path+'_c', map_location=device))


def worker(id, ppo_trainer, envs, env_name, rewards_queue, eval_interval, max_episodes, max_steps, batch_size, \
    gamma, model_path, render, randomized_params):
    with torch.cuda.device(id % torch.cuda.device_count()):
        ppo_trainer.to_cuda()
        print("PPO trainer: ", ppo_trainer)
        try:
            env = gym.make(env_name)  # mujoco env
        except:
            env = envs[env_name]()  # robot env
        for ep in range(max_episodes):
            if randomized_params:
                s = env.reset(**(rand_params(env, params=randomized_params)[0]))
            else:
                s = env.reset()
            ep_r = 0
            for t in range(max_steps):  # in one episode
                # env.render()
                a = ppo_trainer.actor.get_action(s)
                try:
                    s_, r, done, _ = env.step(a)
                    if render: 
                        env.render()  
                except KeyboardInterrupt:
                    print('Finished')
                    ppo_trainer.save_model(model_path)
                except MujocoException:
                    print('MujocoException')
                   # recreate an env, since sometimes reset not works, the env might be broken
                    try:
                        env = gym.make(env_name)  # mujoco env
                    except:
                        env = envs[env_name]()  # robot env
                ppo_trainer.store_transition(s, a, r)  # useful for pendulum

                s = s_
                ep_r += r
                # update ppo
                if len(ppo_trainer.state_buffer) == batch_size:
                    ppo_trainer.finish_path(s_, done)
                    ppo_trainer.update()
                if done:
                    break
            ppo_trainer.finish_path(s_, done)

            print('Worker: ', id, '|Episode: ', ep, '| Episode Reward: ', ep_r, '| Step: ', t)
            rewards_queue.put(ep_r)        

            if ep%eval_interval==0:
                ppo_trainer.save_model(model_path)

        ppo_trainer.save_model(model_path)
        env.close()

    
def cpu_worker(id, ppo_trainer, envs, env_name, rewards_queue, eval_interval, max_episodes, max_steps, batch_size, \
    gamma, model_path, render, randomized_params):
    print("PPO trainer: ", ppo_trainer)
    try:
        env = gym.make(env_name)  # mujoco env
    except:
        env = envs[env_name]()  # robot env
    for ep in range(max_episodes):
        if randomized_params:
            s = env.reset(**(rand_params(env, params=randomized_params)[0]))
        else:
            s = env.reset()
        ep_r = 0
        for t in range(max_steps):  # in one episode
            # env.render()
            a = ppo_trainer.choose_action([s])
            try:
                s_, r, done, _ = env.step(a)
                if render: 
                    env.render()  
            except KeyboardInterrupt:
                print('Finished')
                ppo_trainer.save_model(model_path)
            except MujocoException:
                print('MujocoException')
                # recreate an env, since sometimes reset not works, the env might be broken
                try:
                    env = gym.make(env_name)  # mujoco env
                except:
                    env = envs[env_name]()  # robot env
            ppo_trainer.store_transition(s, a, r)  # useful for pendulum

            s = s_
            ep_r += r
            # update ppo
            if len(ppo_trainer.state_buffer) == batch_size:
                ppo_trainer.finish_path(s_, done)
                ppo_trainer.update()
            if done:
                break
        ppo_trainer.finish_path(s_, done)

        print('Worker: ', id, '|Episode: ', ep, '| Episode Reward: ', ep_r, '| Step: ', t)
        rewards_queue.put(ep_r)        

        if ep%eval_interval==0:
            ppo_trainer.save_model(model_path)

    ppo_trainer.save_model(model_path)
    env.close()
