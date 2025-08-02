# [Function/Library]: TD7.buffer
# Source: TD7 by Scott Fujimoto
# Original Copyright (c) 2023 Scott Fujimoto
# Modifications: Extend to multi-objective version, and add Pareto-ascent direction
# License: MIT
# Repository: https://github.com/sfujim/TD7?tab=MIT-1-ov-file

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer


@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_rate: int = 250
    exploration_noise: float = 0.1

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # LAP
    alpha: float = 0.4
    min_priority: float = 1

    # TD3+BC
    lmbda: float = 0.1

    # Checkpointing
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4
    reset_weight: float = 0.9

    # Encoder Model
    zs_dim: int = 256
    enc_hdim: int = 256
    enc_activ: Callable = F.elu
    encoder_lr: float = 3e-4

    # Critic Model
    critic_hdim: int = 256
    critic_activ: Callable = F.elu
    critic_lr: float = 3e-4

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4


def copy_from(agent, new_agent):
    # Copy model parameters
    new_agent.actor.load_state_dict(agent.actor.state_dict())
    new_agent.actor_target.load_state_dict(agent.actor_target.state_dict())
    new_agent.critic.load_state_dict(agent.critic.state_dict())
    new_agent.critic_target.load_state_dict(agent.critic_target.state_dict())
    new_agent.encoder.load_state_dict(agent.encoder.state_dict())
    new_agent.fixed_encoder.load_state_dict(agent.fixed_encoder.state_dict())
    new_agent.fixed_encoder_target.load_state_dict(agent.fixed_encoder_target.state_dict())
    new_agent.checkpoint_actor.load_state_dict(agent.checkpoint_actor.state_dict())
    new_agent.checkpoint_encoder.load_state_dict(agent.checkpoint_encoder.state_dict())

    # Copy optimizer status
    new_agent.actor_optimizer.load_state_dict(agent.actor_optimizer.state_dict())
    new_agent.critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())
    new_agent.encoder_optimizer.load_state_dict(agent.encoder_optimizer.state_dict())

    # Copy other training states and parameters
    new_agent.training_steps = agent.training_steps
    new_agent.omega = np.copy(agent.omega)
    new_agent.grad_normalize = agent.grad_normalize
    new_agent.max = agent.max
    new_agent.min = agent.min
    new_agent.max_target = agent.max_target
    new_agent.min_target = agent.min_target
    new_agent.eps_since_update = agent.eps_since_update
    new_agent.timesteps_since_update = agent.timesteps_since_update
    new_agent.max_eps_before_update = agent.max_eps_before_update
    new_agent.min_return = agent.min_return
    new_agent.best_min_return = agent.best_min_return


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


def getOmega(rewards):
    """
        The objective weight adjustment method in the paper.
        The larger rewards[i] is, the smaller omega[i] is.
     """
    return rewards / np.sum(rewards)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        super(Actor, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu, num_obj=2):
        super(Critic, self).__init__()

        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, num_obj)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, num_obj)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return q1, q2


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, hp=Hyperparameters(), num_obj=2):
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hp
        self.num_obj = num_obj
        self.omega = np.ones((self.num_obj,)) / self.num_obj
        self.grad_normalize = False

        self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ, num_obj).to(
            self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.buffer = buffer.LAP(state_dim, action_dim, num_obj, self.device, hp.buffer_size, hp.batch_size,
                                 max_action, normalize_actions=True, prioritized=True)

        self.max_action = max_action

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        # self.max = torch.tensor([-1e8] * self.num_obj).to(device=self.device)
        # self.min = torch.tensor([1e8] * self.num_obj).to(device=self.device)
        # self.max_target = torch.zeros((self.num_obj,)).to(device=self.device)
        # self.min_target = torch.zeros((self.num_obj,)).to(device=self.device)
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

    def evaluate_policy(self, eval_env, seed, eval_episodes=2):
        total_returns = np.zeros(self.num_obj)
        t = 0
        for i in range(eval_episodes):
            state = eval_env.reset(seed=seed + i + 2025)
            done = False
            returns = np.zeros((self.num_obj,))
            discount = 1
            while not done:
                with torch.no_grad():
                    action = self.select_action(state, False, False)
                next_state, reward, done, _, _ = eval_env.step(action)
                returns += discount * reward
                # discount *= self.gamma
                state = next_state
                t += 1
            total_returns += returns
        total_returns = total_returns / eval_episodes
        print(f"Rewards: {list(total_returns)}, steps:{t // eval_episodes} ")
        return total_returns

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)

            if use_exploration:
                action = action + torch.randn_like(action) * self.hp.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self, omega=None, obj_id=-1, update_info=""):
        self.training_steps += 1

        state, action, next_state, reward, not_done = self.buffer.sample()

        #########################
        # Update Encoder
        #########################
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        #########################
        # Update Critic
        #########################
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip,
                                                                                   self.hp.noise_clip)
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            # Calculate target Q-values (multi-objective)
            Q_target_next_q1, Q_target_next_q2 = self.critic_target(next_state, next_action,
                                                                    fixed_target_zsa, fixed_target_zs)
            Q_target = torch.min(Q_target_next_q1, Q_target_next_q2)
            Q_target = reward + not_done * self.hp.discount * Q_target.clamp(self.min_target, self.max_target)

            # self.max = torch.maximum(self.max, Q_target.max(dim=0).values)
            # self.min = torch.minimum(self.min, Q_target.min(dim=0).values)
            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        # Calculate the Q-value of the current Critic (multi-objective)
        current_q1, current_q2 = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss1 = (current_q1 - Q_target).abs()
        td_loss2 = (current_q2 - Q_target).abs()

        # Apply preference weighting and calculate Huber loss
        huber_loss1 = LAP_huber(td_loss1).mean()
        huber_loss2 = LAP_huber(td_loss2).mean()
        critic_loss = (huber_loss1 + huber_loss2) / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #########################
        # Update LAP
        #########################
        td_loss = torch.cat([td_loss1, td_loss2], dim=1)
        # priority = td_loss.mean().clamp(min=self.hp.min_priority).pow(self.hp.alpha)
        priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
        self.buffer.update_priority(priority)

        #########################
        # Update Actor
        #########################
        if self.training_steps % self.hp.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            current_q1, current_q2 = self.critic(state, actor, fixed_zsa, fixed_zs)
            min_Q = torch.min(current_q1, current_q2)
            # Obtain weight omega
            if omega is None:
                if self.training_steps % 2000 == 0:  # Each episode is updated once.
                    omega = self.obtain_alpha_star(min_Q, obj_id)
                    self.omega = omega
                    print(f"{update_info}: {omega}")
                else:
                    omega = self.omega
            omega = torch.tensor(omega, device=self.device, dtype=torch.float32).reshape(1, -1)
            weighted_Q = (min_Q * omega).sum(dim=1)
            actor_loss = -weighted_Q.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        #########################
        # Update Iteration
        #########################
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            self.buffer.reset_max_priority()
            self.max_target = self.max
            self.min_target = self.min

    # If using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return, omega, obj_id=-1, update_info=""):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps
        if omega is not None:
            ep_return = (ep_return * np.array(omega)).sum()
        else:
            ep_return = (ep_return * np.array(self.omega)).sum()
        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset(omega, obj_id, update_info)

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset(omega, obj_id, update_info)

    # Batch training
    def train_and_reset(self, omega, obj_id=-1, update_info=""):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.hp.steps_before_checkpointing:
                self.best_min_return *= self.hp.reset_weight
                self.max_eps_before_update = self.hp.max_eps_when_checkpointing

            self.train(omega, obj_id, update_info)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def obtain_alpha_star(self, min_Q, obj_id=-1):
        if obj_id != -1 and (self.num_obj == 2):
            alpha_star = np.zeros((2,))
            alpha_star[obj_id] = 1
            return 1 - alpha_star
        grads = []
        for i in range(self.num_obj):
            if i == obj_id:  # Skip objective i
                continue
            policy_loss = -min_Q[:, i].mean()
            # Forward calculation (policy network only)
            self.actor_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            grad = []
            for param in self.actor.parameters():
                if param.grad is not None:
                    grad.append(param.grad.view(-1))
            grads.append(torch.cat(grad).detach().cpu().numpy())

        # Solve Pareto-ascent direction or Pareto-reverse direction alpha_star
        if self.num_obj == 2 or obj_id != -1:
            grad1, grad2 = grads
            # Regularization, avoiding optimization directions dominated by smaller gradients.
            if self.grad_normalize:
                grad1 /= np.linalg.norm(grad1)
                grad2 /= np.linalg.norm(grad2)
            numerator = np.dot(grad2 - grad1, grad2)
            denominator = np.linalg.norm(grad1 - grad2) ** 2 + 1e-8
            alpha1 = max(0.0, min(1.0, numerator / denominator))
            alpha_star = np.array([alpha1, 1 - alpha1])
            if self.num_obj == 3:
                if obj_id == 0:
                    alpha_star = np.array([0, alpha1, 1 - alpha1])
                elif obj_id == 1:
                    alpha_star = np.array([alpha1, 0, 1 - alpha1])
                else:
                    alpha_star = np.array([alpha1, 1 - alpha1, 0])
        else:
            alpha_star = self.solve_pareto_weights(grads)
        return alpha_star

    def solve_pareto_weights(self, grads, max_iter=100, lr=0.1):
        alpha = np.ones(self.num_obj) / self.num_obj
        grads = np.array(grads)
        if self.grad_normalize:
            grads /= np.linalg.norm(grads)
        for i in range(max_iter):
            grad_alpha = 2 * grads @ grads.T @ alpha
            alpha -= lr * (10 / (i + 10)) * grad_alpha
            alpha = np.clip(alpha, 0, None)
            sum_alpha = alpha.sum()
            if sum_alpha == 0:
                alpha = np.ones_like(alpha) / self.num_obj
            else:
                alpha /= sum_alpha
        return alpha

