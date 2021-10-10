# FROM https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_double_pendulum.py

import numpy as np
import gym.spaces as spaces
from .param_wrapper import MujocoEnvWithParams, EzPickle
import os

with open(os.path.join(os.path.dirname(__file__), 'assets/inverteddoublependulum/model.template.xml')) as f:
    template = f.read()

class InvertedDoublePendulumEnv(MujocoEnvWithParams, EzPickle):
    name = 'inverteddoublependulum'
    observation_space = spaces.Box(np.zeros(11, dtype=np.float32), np.zeros(11, dtype=np.float32))
    action_space = spaces.Box(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32))
    parameters_spec = {
        'damping': [0.02, 0.3],
        'gravity': [8.5, 11.],
        'length_1': [0.3, 0.9],
        'length_2': [0.3, 0.9],
        'density': [0.5, 1.5],
    }

    def reset_props(self, damping = 0.05, gravity = 9.81, length_1 = 0.6, length_2 = 0.6, density = 1.):
        self.damping = damping
        self.gravity = gravity
        self.length_1 = length_1
        self.length_2 = length_2
        self.density = density

        self.init_model_from_xml(template.format(
            damping=damping,
            gravity=gravity,
            length_1=length_1,
            length_2=length_2,
            density=density*1000.,
        ))

        self.params_dict = {
        'damping': damping,
        'gravity': gravity,
        'length_1': length_1,
        'length_2': length_2,
        'density': density, 
        }


    def __init__(self, **kwargs):
        MujocoEnvWithParams.__init__(self, 5, **kwargs)
        EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= (self.length_1 + self.length_2) / 1.2)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        try:
            self.set_state(
                self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
                self.init_qvel + self.np_random.randn(self.model.nv) * .1
            )
        except:
            self.set_state(
                self.init_qpos + np.random.uniform(low=-.1, high=.1, size=self.model.nq),
                self.init_qvel + np.random.randn(self.model.nv) * .1
            )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]