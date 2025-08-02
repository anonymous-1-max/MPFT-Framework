import gymnasium as gym
import numpy as np


class Hopper_2(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super(Hopper_2, self).__init__()
        self.env_name = 'Hopper-v5'
        self.env = gym.make(self.env_name, max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.initial_height = self.env.reset()[0][0]

    def step(self, action):
        action = np.tanh(action)
        rewards = []
        obs, _, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        v = abs(info.get('x_velocity'))
        alive_reward = 1.0 if not terminated else 0.
        reward_others = alive_reward - 2e-4 * np.square(action).sum()
        # Calculate multi-objective reward
        reward1 = 2 * v + reward_others
        reward2 = max(0, 20.0 * info["z_distance_from_origin"]) + reward_others
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