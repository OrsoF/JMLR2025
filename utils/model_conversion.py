import gymnasium as gym
from utils.generic_model import GenericModel
import numpy as np
from scipy import sparse

rng_gen = np.random.default_rng(seed=0)


class ModelToGym(gym.Env):
    def __init__(self, model: GenericModel, max_step: int = 100):
        self.model = model
        self.max_step = max_step

        self.action_space = gym.spaces.Discrete(self.model.action_dim)
        self.observation_space = gym.spaces.Discrete(self.model.state_dim)

    def reset(self, seed: float = 0.0):
        self.steps_done = 0
        self.state = rng_gen.integers(self.model.state_dim)
        obs = self.get_observation(self.state)
        info = {}
        return obs, info

    def step(self, action):
        assert hasattr(self, "state"), "Use env.reset before env.step."
        self.steps_done += 1
        try:
            transitions = (
                self.model.transition_matrix[action]
                .getrow(self.state)
                .todense()
                .reshape(-1)
            )
        except AttributeError:
            transitions = self.model.transition_matrix[action][self.state].reshape(-1)
        reward = self.model.reward_matrix[self.state, action]
        self.state = rng_gen.choice(self.model.state_dim, p=transitions)
        obs = self.get_observation(self.state)
        done = False
        terminated = self.steps_done >= self.max_step
        info = {}
        return obs, reward, done, terminated, info

    def get_observation(self, state: int):
        try:
            obs = self.model.state_list[self.state]
        except AttributeError:
            obs = (self.state,)
        return obs
