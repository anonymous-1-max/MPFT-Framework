import gymnasium as gym
import numpy as np


class Walker2D(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super(Walker2D, self).__init__()
        self.env_name = 'Walker2d-v5'
        self.env = gym.make(self.env_name, max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        action = np.tanh(action)
        rewards = []
        obs, _, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        v = abs(info.get('x_velocity'))
        energy = np.sum(np.square(action))
        alive_reward = 1.0 if not terminated else 0.
        # Calculate multi-objective reward
        reward1 = 1.0*v + alive_reward
        reward2 = 3 - 1.0*energy + alive_reward
        rewards.append(reward1)
        rewards.append(reward2)
        return obs, np.array(rewards), done, terminated, info

    def reset(self, seed=None):
        if seed is None:
            return self.env.reset()[0]
        return self.env.reset(seed=seed)[0]

    def render(self,):
        return self.env.render()

    def close(self,):
        return self.env.close()
