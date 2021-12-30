def get_hyperparams(env_name):
    if 'pandaopendoorfk' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        # 'max_steps': 300,
        'max_steps': 1000,
        'max_episodes': 10000,
        'action_range': 0.05,  # on joint
        # 'action_range': 0.1,  # on joint
        'batch_size': 640,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 500, # evaluate the model and save it
        'explore_noise_scale': 0.02, 
        'eval_noise_scale': 0.02,  # noisy evaluation trick
        'reward_scale': 1., # reward normalization in a batch
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 512,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e6,
        'randomized_params': ['knob_friction', 'hinge_stiffness', 'hinge_damping', 'hinge_frictionloss', 'door_mass', 'knob_mass', 'table_position_offset_x', 'table_position_offset_y'],  # choose in: 'all', None, or a list of parameter keys
        'deterministic': True,
        }
    elif 'pandaopendoorreal' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        # 'max_steps': 300,
        'max_steps': 1000,
        'max_episodes': 5000,
        'action_range': 0.02,  # on joint
        # 'action_range': 0.1,  # on joint
        'batch_size': 640,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 500, # evaluate the model and save it
        'explore_noise_scale': 0.01, 
        'eval_noise_scale': 0.01,  # noisy evaluation trick
        'reward_scale': 1., # reward normalization in a batch
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 512,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e6,
        'randomized_params': ['knob_friction', 'hinge_stiffness', 'hinge_damping', 'hinge_frictionloss', 'door_mass', 'knob_mass', 'table_position_offset_x', 'table_position_offset_y'],  # choose in: 'all', None, or a list of parameter keys
        'deterministic': True,
        }
    elif 'pandaopendoorik' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        'max_steps': 1000,
        'max_episodes': 5000,
        'action_range': 0.02,
        'batch_size': 640,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 500, # evaluate the model and save it
        'explore_noise_scale': 0.01, 
        'eval_noise_scale': 0.01,  # noisy evaluation trick
        'reward_scale': 1., # reward normalization
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 512,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e6,
        'randomized_params': ['knob_friction', 'hinge_stiffness', 'hinge_damping', 'hinge_frictionloss', 'door_mass', 'knob_mass', 'table_position_offset_x', 'table_position_offset_y'],  # choose in: 'all', None, or a list of parameter keys
        'deterministic': True,
        }
    elif 'pandapush' in env_name:  # 'pandapushik2dsimpledynamics' or 'pandapushik2dsimple'
        hyperparams_dict={
        'alg_name': 'td3',
        'max_steps': 500, 
        'max_episodes': 5000,  # 2000
        # 'action_range': 0.05,
        'action_range': 0.1,
        'batch_size': 640,
        'explore_steps': 100000,
        'update_itr': 100,  # iterative update
        'eval_interval': 500, # evaluate the model and save it
        # 'explore_noise_scale': 0.025, 
        # 'eval_noise_scale': 0.025,  # noisy evaluation trick
        'explore_noise_scale': 0.05, 
        'eval_noise_scale': 0.05,  # noisy evaluation trick
        'reward_scale': 1., # reward normalization
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 512,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e6,
        'randomized_params': 
        [
            'joint1_damping', 'joint2_damping', 'joint3_damping', 'joint4_damping', 'joint5_damping', 'joint6_damping', 'joint7_damping', 
            'joint1_armature', 'joint2_armature', 'joint3_armature', 'joint4_armature', 'joint5_armature', 'joint6_armature', 'joint7_armature', 
            'actuator_velocity_joint1_kv', 'actuator_velocity_joint2_kv', 'actuator_velocity_joint3_kv', 'actuator_velocity_joint4_kv', 
            'actuator_velocity_joint5_kv', 'actuator_velocity_joint6_kv', 'actuator_velocity_joint7_kv', 
            'boxobject_size_0', 'boxobject_size_1', 'boxobject_size_2', 
            'boxobject_friction_0', 'boxobject_friction_1', 'boxobject_friction_2', 
            'boxobject_density_1000', 
        ]
        # None
        ,
        'deterministic': True,
        }     
    elif env_name == 'pendulum':
        hyperparams_dict={
        'alg_name': 'td3',
        'max_steps': 1000,
        'max_episodes': 200,
        'action_range': None,  # automatically query from env
        'batch_size': 640,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 100, # evaluate the model and save it
        'explore_noise_scale': 0.6, 
        'eval_noise_scale': 0.2,  # noisy evaluation trick
        'reward_scale': 1., # reward normalization
        'gamma': 0.9, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 64,
        'noise_decay': 1., # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e5,
        'randomized_params': 'all',
        'deterministic': True,
        } 
    elif 'inverteddoublependulum' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        'max_steps': 1000,
        'max_episodes': 2000,
        'action_range': None,
        'batch_size': 256,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 200, # evaluate the model and save it
        'explore_noise_scale': 0.1, 
        'eval_noise_scale': 0.2,  # noisy evaluation trick
        'reward_scale': None, # reward normalization
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 256,
        'noise_decay': 1., # decaying exploration noise
        'policy_target_update_interval': 2, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e5,
        'randomized_params': 'all',
        'deterministic': True,
        }  
    elif 'InvertedDoublePendulum-v2' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        'max_steps': 1000,
        'max_episodes': 2000,
        'action_range': None,
        'batch_size': 256,
        'explore_steps': 0,
        'update_itr': 100,  # iterative update
        'eval_interval': 100, # evaluate the model and save it
        'explore_noise_scale': 0.1, 
        'eval_noise_scale': 0.2,  # noisy evaluation trick
        'reward_scale': None, # reward normalization
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 256,
        'noise_decay': 1., # decaying exploration noise
        'policy_target_update_interval': 2, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e5,
        'randomized_params': None,
        'deterministic': True,
        }   
    elif 'halfcheetah' in env_name:   
        hyperparams_dict={
        'alg_name': 'td3',
        'max_steps': 1000,
        'max_episodes': 25000,
        'action_range': None,
        'batch_size': 640,
        'explore_steps': 100000,
        'update_itr': 100,  # iterative update
        'eval_interval': 5000, # evaluate the model and save it
        'explore_noise_scale': 0.8, 
        'eval_noise_scale': 0.2,  # noisy evaluation trick
        'reward_scale': None, # reward normalization
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 64,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 1e5,
        'randomized_params': None,
        'deterministic': True,
        }      
    else:
        raise NotImplementedError

    print('Hyperparameters: ', hyperparams_dict)
    return hyperparams_dict
