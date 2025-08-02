from collections import deque
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from normalization import Normalization, RewardScaling
import paretoAscent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


def init_sequential(seq):
    linear_layers = []
    for i, layer in enumerate(seq):
        if isinstance(layer, nn.Linear):
            linear_layers.append((i, layer))

    for idx, (i, layer) in enumerate(linear_layers):
        if i + 1 < len(seq):
            next_layer = seq[i + 1]
            if isinstance(next_layer, nn.ReLU):
                gain = np.sqrt(2)
            elif isinstance(next_layer, nn.Tanh):
                gain = np.sqrt(0.5)
            else:
                gain = 1.0
        else:
            gain = 1.0
        orthogonal_init(layer, gain=gain)


class Policy_Value(nn.Module):
    def __init__(self, state_dim, action_dim, value_dim, hidden_dim=256, ort_init=True, use_Tanh=True, max_action=1):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh() if use_Tanh else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() if use_Tanh else nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ).to(device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device))

        self.fc2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh() if use_Tanh else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() if use_Tanh else nn.ReLU(),
            nn.Linear(hidden_dim, value_dim)
        ).to(device)

        if ort_init:
            init_sequential(self.fc1)
            init_sequential(self.fc2)

    def forward(self, state):
        mean = self.fc1(state) * self.max_action
        log_std = self.log_std.exp().expand_as(mean).clamp(min=1e-6, max=20)
        return torch.distributions.Normal(mean, log_std), self.fc2(state)


class PPO:
    def __init__(self, env, eval_env, seed, num_objectives, lr=3e-4, gamma=0.99, clip_epsilon=0.2, max_train_steps=5e6):
        self.max_train_steps = max_train_steps
        self.clip_param = 0.2
        self.use_clipped_value_loss = False
        self.seed = seed
        self.env = env
        self.eval_env = eval_env
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.PPO_epoch = 10
        self.mini_batch = 64
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.use_gae = True
        self.state_normalize = True
        self.reward_scaling = True
        self.use_grad_clip = True
        self.use_lr_decay = False
        self.pareto_tracking = False
        self.lr = lr
        self.buffer_size = int(2048 * 2)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.clip_epsilon = clip_epsilon
        self.grad_normalize = True
        self.PA = paretoAscent.ParetoAscentDirection()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.state_norm = Normalization(shape=self.state_dim)
        self.reward_scal = RewardScaling(shape=self.num_objectives, gamma=self.gamma)
        self.total_steps = 0

    def evaluate_policy(self, policy, state_norm, eval_episodes=2):
        total_returns = np.zeros(self.num_objectives)
        for i in range(eval_episodes):
            state = self.eval_env.reset(seed=self.seed + i + 1)
            done = False
            returns = np.zeros((self.num_objectives,))
            discount = 1
            while not done:
                if self.state_normalize:
                    state = state_norm(state, update=False)  # During the evaluating,update=False
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action, _ = policy(state_tensor)
                    action = action.sample().cpu().numpy()
                next_state, reward, done, _, _ = self.eval_env.step(action)
                returns += discount * reward
                # discount *= self.gamma
                state = next_state
            total_returns += returns
        total_returns = total_returns / eval_episodes
        if np.random.random() > 0.95:
            print(f"Rewards: {list(total_returns)} ")
        return total_returns

    def collectData(self, policy):
        self.replay_buffer.clear()
        state = self.env.reset(seed=self.seed)
        if self.state_normalize:
            state = self.state_norm(state)
        while len(self.replay_buffer) < self.replay_buffer.maxlen:
            self.total_steps += 1
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                dist, values = policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                value_np = values.cpu().numpy()
            next_state, reward, done, terminated, _ = self.env.step(action.cpu().numpy())
            if self.state_normalize:
                next_state = self.state_norm(next_state)
            if self.reward_scaling:
                reward = self.reward_scal(reward)
            # store transition
            self.replay_buffer.append((
                state,
                action.cpu().numpy(),
                reward,
                next_state,
                done,
                terminated,
                log_prob.item(),
                value_np
            ))
            if done:
                state = self.env.reset()
                if self.state_normalize:
                    state = self.state_norm(state)
                if self.reward_scaling:
                    self.reward_scal.reset()
            else:
                state = next_state
        states = torch.FloatTensor(np.array([t[0] for t in self.replay_buffer])).to(device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.replay_buffer])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in self.replay_buffer])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in self.replay_buffer])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in self.replay_buffer])).to(device)
        terminated = torch.FloatTensor(np.array([t[5] for t in self.replay_buffer])).to(device)
        old_log_probs = torch.FloatTensor(np.array([t[6] for t in self.replay_buffer])).to(device)
        old_values = torch.FloatTensor(np.array([t[7] for t in self.replay_buffer])).to(device)
        return states, actions, rewards, next_states, dones, terminated, old_log_probs, old_values

    def updatePolicy_Value(self, policy, optimizer, states, actions, next_states, old_log_probs, weighted_advantages,
                           returns, old_values):
        # create dataset
        dataset = TensorDataset(states, actions, next_states, old_log_probs, weighted_advantages, returns, old_values)
        dataloader = DataLoader(dataset, batch_size=self.mini_batch, shuffle=True)

        # PPO Updating
        for _ in range(self.PPO_epoch):
            for batch in dataloader:
                s, a, n_s, old_lp, adv, ret, old_v = batch
                dist, values = policy(s)
                new_lp = dist.log_prob(a).sum(-1)
                entropy = dist.entropy().mean()
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.use_clipped_value_loss:
                    value_pred_clipped = old_v + \
                                         (values - old_v).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - ret).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - ret).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (ret - values).pow(2).mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                if self.use_grad_clip:  # Gradient clip
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
            if self.use_lr_decay and (not self.pareto_tracking):  # learning rate Decay
                self.lr_decay(optimizer)

    def computingAdvantage(self, policy, rewards, next_states, dones, terminated, old_values):
        with torch.no_grad():
            _, next_values = policy(next_states)
            next_values *= (1 - terminated.unsqueeze(-1))
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        # For each objective calculate the GAE
        if self.use_gae:
            for obj_idx in range(self.num_objectives):
                running_gae = 0
                for t in reversed(range(len(self.replay_buffer))):
                    next_value = next_values[t, obj_idx]
                    delta = rewards[t, obj_idx] + self.gamma * next_value * (1 - terminated[t]) - old_values[t, obj_idx]
                    running_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * running_gae
                    advantages[t, obj_idx] = running_gae
                    returns[t, obj_idx] = advantages[t, obj_idx] + old_values[t, obj_idx]
        else:
            for obj_idx in range(self.num_objectives):
                for t in reversed(range(len(self.replay_buffer))):
                    next_value = next_values[t, obj_idx]
                    returns[t, obj_idx] = next_value * self.gamma * (1 - terminated[t]) + rewards[t, obj_idx]
                    advantages[t, obj_idx] = returns[t, obj_idx] - old_values[t, obj_idx]
        return advantages, returns

    def update(self, policy, optimizer, omega):
        # Rollout
        states, actions, rewards, next_states, dones, terminated, old_log_probs, old_values = self.collectData(policy)
        advantages, returns = self.computingAdvantage(policy, rewards, next_states, dones, terminated, old_values)
        # Scalar GAE
        weighted_advantages = (advantages * torch.FloatTensor(omega).to(device)).sum(dim=1)
        # Normalized GAE
        weighted_advantages = (weighted_advantages - weighted_advantages.mean(0)) / (weighted_advantages.std(0) + 1e-5)
        self.updatePolicy_Value(policy, optimizer, states, actions, next_states, old_log_probs, weighted_advantages,
                                returns, old_values)

    def solve_pareto_weights(self, grads):
        if self.grad_normalize:
            for i in range(self.num_objectives):
                grads[i] /= np.linalg.norm(grads[i])
        alpha = self.PA.solve(np.array(grads))
        return alpha

    def obtain_alpha_star(self, policy, optimizer, obj_id=-1):
        if obj_id != -1 and (self.num_objectives == 2):
            alpha_star = np.zeros((2,))
            alpha_star[obj_id] = 1
            return 1 - alpha_star
        states, actions, rewards, next_states, dones, terminated, old_log_probs, old_values = self.collectData(policy)
        advantages, returns = self.computingAdvantage(policy, rewards, next_states, dones, terminated, old_values)
        grads = []
        for i in range(self.num_objectives):
            if i == obj_id:
                continue
            for param in policy.fc2.parameters():
                param.requires_grad = False
            dist, _ = policy(states)
            log_probs = dist.log_prob(actions).sum(-1)
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages[:, i]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[:, i]
            policy_loss = -torch.min(surr1, surr2).mean()
            optimizer.zero_grad()
            policy_loss.backward()
            grad = []
            for param in policy.fc1.parameters():
                if param.grad is not None:
                    grad.append(param.grad.view(-1))
            if policy.log_std.grad is not None:
                grad.append(policy.log_std.grad.view(-1))
            # 恢复值网络梯度计算
            for param in policy.fc2.parameters():
                param.requires_grad = True
            grads.append(torch.cat(grad).detach().cpu().numpy())
        if self.num_objectives == 2 or obj_id != -1:
            grad1, grad2 = grads
            if self.grad_normalize:
                grad1 /= np.linalg.norm(grad1)
                grad2 /= np.linalg.norm(grad2)
            numerator = np.dot(grad2 - grad1, grad2)
            denominator = np.linalg.norm(grad1 - grad2) ** 2 + 1e-8
            alpha1 = max(0.0, min(1.0, numerator / denominator))
            alpha_star = np.array([alpha1, 1 - alpha1])
            if self.num_objectives == 3:
                if obj_id == 0:
                    alpha_star = np.array([0, alpha1, 1 - alpha1])
                elif obj_id == 1:
                    alpha_star = np.array([alpha1, 0, 1 - alpha1])
                else:
                    alpha_star = np.array([alpha1, 1 - alpha1, 0])
        else:
            alpha_star = self.solve_pareto_weights(grads)
        return alpha_star

    def lr_decay(self, optimizer):
        lr = self.lr * (1 - self.total_steps / self.max_train_steps)
        for p in optimizer.param_groups:
            p['lr'] = lr
