# FROM https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py

import numpy as np
import gym.spaces as spaces
from .param_wrapper import MujocoEnvWithParams, EzPickle
import os

with open(os.path.join(os.path.dirname(__file__), 'assets/halfcheetah/model.template.xml')) as f:
    template = f.read()
DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}
class HalfCheetahEnv(MujocoEnvWithParams, EzPickle):
    name = 'halfcheetah'
    observation_space = spaces.Box(np.zeros(17, dtype=np.float32), np.zeros(17, dtype=np.float32))
    action_space = spaces.Box(np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32))
    parameters_spec = {
        # 'gravity': [8.5, 11.],
        # 'bthigh_damping': [5., 7.],
        # 'bshin_damping': [3.5, 5.5],
        # 'bfoot_damping': [2., 4.],
        # 'fthigh_damping': [3.5, 5.5],
        # 'fshin_damping': [2., 4.],
        # 'ffoot_damping': [0.5, 2.5],
        # 'bthigh_stiffness': [200, 280],
        # 'bshin_stiffness': [120, 240],
        # 'bfoot_stiffness': [80, 160],
        # 'fthigh_stiffness': [120, 240],
        # 'fshin_stiffness': [80, 160],
        # 'ffoot_stiffness': [40, 80],

        # larger range
        'gravity': [5.5, 14.],
        'bthigh_damping': [3., 9.],
        'bshin_damping': [1.5, 7.5],
        'bfoot_damping': [1., 5.],
        'fthigh_damping': [1.5, 7.5],
        'fshin_damping': [1., 5.],
        'ffoot_damping': [0.2, 2.8],
        'bthigh_stiffness': [100, 380],
        'bshin_stiffness': [20, 340],
        'bfoot_stiffness': [10, 230],
        'fthigh_stiffness': [20, 340],
        'fshin_stiffness': [20, 220],
        'ffoot_stiffness': [10, 110],
    }

    def reset_props(self, gravity = 9.81, bthigh_damping = 6., bshin_damping = 4.5, bfoot_damping = 3.,
    fthigh_damping = 4.5, fshin_damping = 3., ffoot_damping = 1.5, bthigh_stiffness = 240, bshin_stiffness = 180,
    bfoot_stiffness = 120,  fthigh_stiffness = 180, fshin_stiffness = 120, ffoot_stiffness = 60,
    ):
        self.gravity = gravity
        self.bthigh_damping = bthigh_damping
        self.bshin_damping = bshin_damping
        self.bfoot_damping = bfoot_damping
        self.fthigh_damping = fthigh_damping
        self.fshin_damping = fshin_damping
        self.ffoot_damping = ffoot_damping
        self.bthigh_stiffness = bthigh_stiffness
        self.bshin_stiffness = bshin_stiffness
        self.bfoot_stiffness = bfoot_stiffness
        self.fthigh_stiffness = fthigh_stiffness
        self.fshin_stiffness = fshin_stiffness
        self.ffoot_stiffness = ffoot_stiffness


        self.init_model_from_xml(template.format(
            gravity=gravity,
            bthigh_damping=bthigh_damping,
            bshin_damping=bshin_damping,
            bfoot_damping = bfoot_damping,
            fthigh_damping = fthigh_damping,
            fshin_damping = fshin_damping,
            ffoot_damping = ffoot_damping,
            bthigh_stiffness = bthigh_stiffness,
            bshin_stiffness = bshin_stiffness,
            bfoot_stiffness = bfoot_stiffness,
            fthigh_stiffness = fthigh_stiffness,
            fshin_stiffness = fshin_stiffness,
            ffoot_stiffness = ffoot_stiffness,
        ))

        self.params_dict = {
        'gravity': gravity,
        'bthigh_damping': bthigh_damping,
        'bshin_damping': bshin_damping,
        'bfoot_damping': bfoot_damping, 
        'fthigh_damping': fthigh_damping, 
        'fshin_damping': fshin_damping, 
        'ffoot_damping': ffoot_damping, 
        'bthigh_stiffness': bthigh_stiffness, 
        'bshin_stiffness': bshin_stiffness, 
        'bfoot_stiffness': bfoot_stiffness, 
        'fthigh_stiffness': fthigh_stiffness, 
        'fshin_stiffness': fshin_stiffness, 
        'ffoot_stiffness': ffoot_stiffness, 

        }
    """ This is HalfCheetah-v3 in https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah_v3.py"""
    def __init__(self,
                #  xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        # utils.EzPickle.__init__(**locals())
        EzPickle.__init__(self)

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        # mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        MujocoEnvWithParams.__init__(self, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
    
    """ Below is HalfCheetah-v2 in https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py"""
    # def __init__(self):
    #     MujocoEnvWithParams.__init__(self, 5)
    #     EzPickle.__init__(self)

    # def step(self, action):
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(action, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     ob = self._get_obs()
    #     reward_ctrl = - 0.1 * np.square(action).sum()
    #     reward_run = (xposafter - xposbefore)/self.dt
    #     reward = reward_ctrl + reward_run
    #     done = False
    #     return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         self.sim.data.qvel.flat,
    #     ])

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

