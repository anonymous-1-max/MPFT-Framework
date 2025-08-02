import gymnasium as gym
import numpy as np


class HalfCheetah(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super(HalfCheetah, self).__init__()
        self.env_name = 'HalfCheetah-v5'
        self.env = gym.make(self.env_name, max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        action = np.tanh(action)
        rewards = []
        obs, _, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        vx_speed = info.get('x_velocity')
        alive_reward = 0.0 if not terminated else 0.
        energy = np.sum(np.square(action))
        reward1 = min(0.5*vx_speed, 2) + alive_reward
        reward2 = 2.0 - energy + alive_reward
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
