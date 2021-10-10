from .gym_envs.acrobot import AcrobotEnv
from .gym_envs.cartpole import CartPoleEnv
from .gym_envs.mountaincar import MountainCarEnv
from .gym_envs.pendulum import PendulumEnv
from .gym_envs.inverteddoublependulumdisc import InvertedDoublePendulumDiscreteEnv
from .gym_envs.inverteddoublependulum import InvertedDoublePendulumEnv
from .gym_envs.inverteddoublependulumdynamics import InvertedDoublePendulumDynamicsEnv
from .gym_envs.inverteddoublependulumdynamicsembedding import InvertedDoublePendulumDynamicsEmbeddingEnv
from .gym_envs.halfcheetah import HalfCheetahEnv
from .gym_envs.halfcheetahdynamics import HalfCheetahDynamicsEnv
from .gym_envs.halfcheetahdynamicsembedding import HalfCheetahDynamicsEmbeddingEnv

envs = {
    'acrobot': AcrobotEnv,
    'cartpole': CartPoleEnv,
    'mountaincar': MountainCarEnv,
    'pendulum': PendulumEnv,
    'inverteddoublependulumdisc': InvertedDoublePendulumDiscreteEnv,
    'inverteddoublependulum': InvertedDoublePendulumEnv,
    'inverteddoublependulumdynamics': InvertedDoublePendulumDynamicsEnv,
    'inverteddoublependulumdynamicsembedding': InvertedDoublePendulumDynamicsEmbeddingEnv,
    'halfcheetah': HalfCheetahEnv,
    'halfcheetahdynamics': HalfCheetahDynamicsEnv,
    'halfcheetahdynamicsembedding': HalfCheetahDynamicsEmbeddingEnv,
}
