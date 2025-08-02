import gymnasium as gym
import numpy as np


class Ant(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super(Ant, self).__init__()
        self.env_name = 'Ant-v5'
        self.env = gym.make(self.env_name, max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        action = np.tanh(action)
        rewards = []
        obs, _, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        speed_x = info.get('x_velocity')
        speed_y = info.get('y_velocity')
        energy = np.square(action).sum()  # Energy consumption is often proportional to the square of actions
        alive_reward = 1.0 if not terminated else 0.
        # Calculate multi-objective reward
        vx_reward = 0.35*speed_x - 1 * energy + alive_reward
        vy_reward = 0.35*speed_y - 1 * energy + alive_reward
        rewards.append(vx_reward)
        rewards.append(vy_reward)
        return obs, np.array(rewards), done, terminated, info

    def reset(self, seed=None):
        if seed is None:
            return self.env.reset()[0]
        return self.env.reset(seed=seed)[0]

    def render(self,):
        return self.env.render()

    def close(self,):
        return self.env.close()
