import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.utils import EzPickle
import mujoco_py

class MujocoEnvWithParams(mujoco_env.MujocoEnv):
    def reset_props(self):
        """
        Implement this by generating a xml string with parameters filled in, then CALL self.init_model_from_xml(xmlstring), don't forget it!
        """
        raise NotImplementedError

    def init_model_from_xml(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        
        # GZZ: Need to do this for a consistent display window
        # method `update_sim`: https://github.com/openai/mujoco-py/blob/1fe312b09ae7365f0dd9d4d0e453f8da59fae0bf/mujoco_py/mjrendercontext.pyx#L74
        if hasattr(self, '_viewers'):
            for mode, v in self._viewers.items():
                v.update_sim(self.sim)
        else:
            self.viewer = None
            self._viewers = {}
        
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

    def __init__(self, frame_skip, **kwargs):
        self.frame_skip = frame_skip

        self.reset_props(**kwargs)
        assert self.model is not None
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        
        self._set_action_space()
        
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        
        self._set_observation_space(observation)
        
        self.seed()

    def reset(self, **kwargs):
        # modified from MujocoEnv.reset
        self.reset_props(**kwargs)
        ob = self.reset_model()
        return ob



