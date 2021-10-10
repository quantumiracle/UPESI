import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import time
import queue
import math
import random
import datetime
import os
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from rl.buffers import ReplayBuffer
from utils.load_params import load_params
from utils.common_func import rand_params
from rl.ppo.ppo_continuous import PPOContinuous_Trainer, worker, cpu_worker

def train_ppo(env, envs, train, test, finetune, path, model_id, render, process, seed):
    torch.manual_seed(seed)  # Reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # hyper-parameters for RL training
    try: # custom env
        env_name = env.name 
    except: # gym env
        env_name = env.spec.id

    num_workers = process # or: mp.cpu_count()
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_path = './data/weights/{}'.format(prefix)
    if not os.path.exists(model_path) and train:
        os.makedirs(model_path)
    print('Model Path: ', model_path)

    # load other default parameters
    [max_steps, max_episodes, action_range, batch_size, eval_interval, \
        gamma, hidden_dim, a_lr, c_lr, randomized_params] = \
        load_params('ppo', env_name, ['max_steps', 'max_episodes', 'action_range', 'batch_size', 'eval_interval', \
        'gamma', 'hidden_dim', 'a_lr', 'c_lr', 'randomized_params'] )
    if not action_range:
        action_range = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric

    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    manager = BaseManager()
    manager.start()

    action_space = env.action_space
    state_space = env.observation_space

    ppo_trainer=PPOContinuous_Trainer(state_space, action_space, hidden_dim, a_lr, c_lr,\
        action_range=action_range, machine_type='cpu' )

    if train: 
        if finetune is True:
            ppo_trainer.load_model('./data/weights/'+ path +'/{}_ppo'.format(model_id))
        ppo_trainer.share_memory()

        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        eval_rewards_queue = mp.Queue()  # used for get offline evaluated rewards from all processes and plot the curve
        success_queue = mp.Queue()  # used for get success events from all processes
        eval_success_queue = mp.Queue()

        processes=[]
        rewards=[]
        success = []
        eval_rewards = []
        eval_success = []

        for i in range(num_workers):
            process = Process(target=cpu_worker, args=(i, ppo_trainer, envs, env_name, rewards_queue, eval_interval, max_episodes, max_steps, batch_size, \
            gamma, model_path, render, randomized_params))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep getting the episode reward from the queue
            r = rewards_queue.get()
            rewards.append(r)

            if len(rewards)%20==0 and len(rewards)>0:
                # plot(rewards)
                np.save('log/'+prefix+'ppo_rewards', rewards)

        [p.join() for p in processes]  # finished at the same time

        ppo_trainer.save_model(model_path)
        
    if test:
        import time
        model_path = './data/weights/'+ path +'/{}_ppo'.format(str(model_id))
        print('Load model from: ', model_path)
        ppo_trainer.load_model(model_path)
        # ppo_trainer.to_cuda()
        # print(env.action_space.high, env.action_space.low)

        DR = True
        
        if DR:
            randomized_params=[]
            if DR and 'pandaopendoor' in env_name:
                randomized_params = randomized_params + [
                'knob_friction',
                'hinge_stiffness',
                'hinge_damping',
                'hinge_frictionloss',
                'door_mass',
                'knob_mass',
                'table_position_offset_x',
                'table_position_offset_y'
                ]
            else:
                randomized_params = 'all'
        for eps in range(10):
            if DR:
                param_dict, param_vec = rand_params(env, randomized_params)
                state = env.reset(**param_dict)
                print('Randomized parameters value: ', param_dict)
            else:
                state = env.reset()
            # print(state)
            env.render()   
            episode_reward = 0

            s_list = []
            for step in range(max_steps):
                action = ppo_trainer.policy_net.get_action(state, noise_scale=0.0)
                # print(action)
                next_state, reward, done, _ = env.step(action)
                env.render() 
                # time.sleep(0.1)
                # print(step, state[0])
                episode_reward += reward
                state=next_state
                # if np.any(state[-30:]>0):  # when there is tactile signal
                #     print(step, state)
                # print(step, action, reward)
                s_list.append(state)
                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            # np.save('data/s.npy', s_list)
