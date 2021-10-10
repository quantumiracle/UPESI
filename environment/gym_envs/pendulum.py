import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    name = 'pendulum'
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    ref_high = np.array([1., 1., 8], dtype=np.float32)
    observation_space = spaces.Box(low=-ref_high, high=ref_high)
    action_space = spaces.Box(low=-1., high=1., shape=(1,))

    parameters_spec = {
        'gravity': [8.5, 11.],
        'mass': [0.5, 2.],
        'length': [0.5, 1.5],
    }

    default_model_settings = {
        'learning_rate': 0.00001,
        'reward_bias': 8.,
    }

    def reset_props(self, gravity=10.0, mass=1., length=1., **kwargs):
        self.g = gravity
        self.m = mass
        self.l = length

    def __init__(self, **kwargs):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.reset_props(**kwargs)
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u * self.max_torque, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self, **kwargs):
        self.reset_props(**kwargs)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            self.pole_transform = rendering.Transform()
            fname = path.join(path.dirname(__file__), "assets/pendulum/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        rod = rendering.make_capsule(self.l, .2)
        rod.set_color(.8, .3, .3)
        rod.add_attr(self.pole_transform)
        self.viewer.add_onetime(rod)
        
        axle = rendering.make_circle(.05)
        axle.set_color(0, 0, 0)
        self.viewer.add_onetime(axle)
        
        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
