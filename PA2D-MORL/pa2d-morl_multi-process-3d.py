import os
import random
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from environments import hopper_3
import hypervolume
import ppo
from normalization import Normalization, RewardScaling
import multiprocessing as mp
import paretoAscent
import tracemalloc
import psutil

mp.set_start_method('spawn', force=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dominated(obj1, obj2):
    return np.all(obj2 >= obj1) and np.any(obj2 > obj1)


def generate_weights_batch_dfs(i, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch):
    """
        [Function/Library]: PGMORL morl/util.generate_weights_batch_dfs
        Source: PGMORL by MIT Graphics Group
        Copyright (c) 2020 MIT Graphics Group
        License:  MIT
        Repository: https://github.com/mit-gfx/PGMORL/blob/master/morl/utils.py
    """
    if i == obj_num - 1:
        weight.append(1.0 - np.sum(weight[0:i]))
        weights_batch.append(deepcopy(weight))
        weight = weight[0:i]
        return
    w = min_weight
    while w < max_weight + 0.5 * delta_weight and np.sum(weight[0:i]) + w < 1.0 + 0.5 * delta_weight:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, obj_num, min_weight, max_weight, delta_weight, weight, weights_batch)
        weight = weight[0:i]
        w += delta_weight


class PA2D_MORL:
    def __init__(self, environment, eval_env, num_objectives, seed=42, total_generations=60, p_size=15, gamma=0.99,
                 clip_epsilon=0.2):
        self.seed = seed
        self.env = environment
        self.eval_env = eval_env
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.action_space.seed(self.seed)  # 动作空间的种子
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.steps_np = np.zeros((100, 100))
        self.episodes_np = np.zeros((100, 100))
        self.PPO_epoch = 10
        self.mini_batch = 32
        self.gae_lambda = 0.95
        self.lr = 3e-4
        self.agent = ppo.PPO(self.env, self.eval_env, self.seed, num_objectives)
        self.buffer_size = 4096
        self.clip_epsilon = clip_epsilon
        # Written outside, it will be reset when called by multiple threads.
        # self.state_norm = [Normalization(shape=self.state_dim) for _ in range(p_size + 2)]
        # self.reward_scal = [RewardScaling(shape=self.num_objectives, gamma=self.gamma) for _ in range(p_size + 2)]
        self.state_normalize = True  # PPO trick
        self.reward_scaling = True  # PPO trick
        self.set_adam_eps = True
        self.use_intermediate_policy = True
        self.grad_normalize = False
        self.PA = paretoAscent.ParetoAscentDirection()
        self.non_dominated_set = []  # (policy, optimizer, objectives, policy_id), policy_id: root policy id
        self.M = total_generations
        self.G_theta = 21
        self.G_phi = 10
        self.G = self.G_theta * self.G_phi  # G in our appendix, M in original paper
        self.Top_k = 2  # Maintain the original paper setting.
        self.m_w = 200  # Iterations of warmup.
        self.m = 40  # Iterations of each generation; m_t in our appendix.
        self.M_ft = self.M // 2  # PA-FT generations
        self.p = p_size  # The number of selected policies in each generation; n in our appendix.
        # It must be stored on the CPU, otherwise parallel computing will result in an error.
        self.population = [[] for _ in range(self.G)]  # policy population (policy, optimizer, distance, policy_id)
        self.population_size = 0
        self.hv_history = []
        self.sp_history = []

    def save_policies(self, save_dir="saved_policies"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, (policy, _, objectives, _) in enumerate(self.non_dominated_set):
            obj_str = "_".join([f"{o:.2f}" for o in objectives])
            filename = f"policy_{idx}_{obj_str}.pth"
            torch.save(policy.state_dict(), os.path.join(save_dir, filename))

    # 添加可视化方法
    def visualize_training(self, save_path="training_metrics.png"):
        print(f"HV: {self.hv_history}")
        print(f"SP: {self.sp_history}")
        plt.figure(figsize=(12, 6))
        # HV
        plt.subplot(1, 2, 1)
        plt.plot(self.hv_history, marker='o', linestyle='-', color='b')
        plt.title("Hypervolume (HV) Trend")
        plt.xlabel("Generation")
        plt.ylabel("HV Value")
        plt.grid(True)
        # SP
        plt.subplot(1, 2, 2)
        plt.plot(self.sp_history, marker='o', linestyle='-', color='r')
        plt.title("Spread (SP) Trend")
        plt.xlabel("Generation")
        plt.ylabel("SP Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_pareto_front(self, save_path="pareto_front.png"):
        if not self.non_dominated_set:
            return
        objectives = np.array([entry[2] for entry in self.non_dominated_set])
        print(f"Objectives: {list(objectives)}")
        plt.figure(figsize=(8, 6))
        if self.num_objectives == 2:
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50, edgecolors='k')
            plt.title("2D Pareto Front")
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
        elif self.num_objectives == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_title("3D Pareto Front")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def calculate_distance(self, policy, state_norm):
        """Calculate the Euclidean distance from the strategy to the reference point."""
        objectives = self.agent.evaluate_policy(policy, state_norm)
        Z = np.zeros(self.num_objectives)  # Default reference point
        distance = np.linalg.norm(objectives - Z)
        return distance, objectives

    def update_population(self, policy_optimizers, policy_ids, state_norms):
        """
            The trained policies and their corresponding objectives are inserted into the population self.populations. 、
            with the angle of the line connecting the objective to the reference point to the x-axis is divided into G
            groups (0~90 degrees) each group is sorted according to the distance from the objective to the reference
            point from the largest to the smallest.
        """
        add_size = len(policy_optimizers)
        for (policy, optimizer), policy_id in zip(policy_optimizers, policy_ids):
            new_policy = deepcopy(policy).to(device)
            policy_id = deepcopy(policy_id)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr,
                                             eps=1e-5 if self.set_adam_eps else 1e-8)
            new_optimizer.load_state_dict(optimizer.state_dict())
            # Evaluate
            distance, objectives = self.calculate_distance(new_policy, state_norms[policy_id])

            new_policy.to("cpu")
            for param in new_optimizer.state.values():
                for k, v in param.items():
                    if isinstance(v, torch.Tensor):
                        param[k] = v.cpu()

            x, y, z = objectives[0], objectives[1], objectives[2]
            norm_xy = np.sqrt(x ** 2 + y ** 2)

            # Avoid division by zero
            theta = np.degrees(np.arctan2(y, x)) if norm_xy > 0 else 0.0
            theta = theta % 90.0
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            phi = np.degrees(np.arctan2(z, norm_xy)) if r > 0 else 0.0
            phi = max(0.0, min(90.0, phi))

            # Determine the group id
            theta_idx = int(theta // (90.0 / self.G_theta))
            theta_idx = min(theta_idx, self.G_theta - 1)
            phi_idx = int(phi // (90.0 / self.G_phi))
            phi_idx = min(phi_idx, self.G_phi - 1)
            group_idx = phi_idx * self.G_theta + theta_idx
            group_idx = min(group_idx, self.G - 1)

            # Insert into population and sort
            self.population[group_idx].append((new_policy, new_optimizer, distance, policy_id))
            self.population[group_idx].sort(key=lambda x: -x[2])  # Sorted by distance in descending order
        self.population_size += add_size
        print(f" Population size: {self.population_size}")

    def collectData(self, policy, policy_id, process_id, state_norm, reward_scal):
        replay_buffer = deque(maxlen=self.buffer_size)
        state = self.env.reset(seed=self.seed)
        pre_steps = self.steps_np[policy_id, process_id]
        rewards_temp = np.zeros((self.num_objectives,))
        returns_temp = np.zeros((self.num_objectives,))
        discount = 1
        if self.state_normalize:
            state = state_norm(state)
        while len(replay_buffer) < replay_buffer.maxlen:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                dist, values = policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                value_np = values.cpu().numpy()
            next_state, reward, done, terminated, _ = self.env.step(action.cpu().numpy())
            if self.state_normalize:
                next_state = state_norm(next_state)
            rewards_temp += reward
            returns_temp += discount * reward
            discount *= self.gamma
            if self.reward_scaling:
                reward = reward_scal(reward)
            self.steps_np[policy_id, process_id] += 1
            # Store the transition.
            replay_buffer.append((
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
                state = self.env.reset(seed=self.seed)
                if self.state_normalize:
                    state = state_norm(state)
                if self.reward_scaling:
                    reward_scal.reset()
                self.episodes_np[policy_id, process_id] += 1
                if self.episodes_np[policy_id, process_id] % 20 == 0:
                    print(f"Policy: {policy_id}, Total_steps: {np.sum(self.steps_np[policy_id])} || "
                          f"Episode: {self.episodes_np[policy_id, process_id]}, steps: {self.steps_np[policy_id, process_id] - pre_steps},"
                          f"rewards {list(rewards_temp)}, returns {list(returns_temp)}")
                pre_steps = self.steps_np[policy_id, process_id]
                rewards_temp = np.zeros((self.num_objectives,))
                returns_temp = np.zeros((self.num_objectives,))
                discount = 1
            else:
                state = next_state
        states = torch.FloatTensor(np.array([t[0] for t in replay_buffer])).to(device)
        actions = torch.FloatTensor(np.array([t[1] for t in replay_buffer])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in replay_buffer])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in replay_buffer])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in replay_buffer])).to(device)
        terminated = torch.FloatTensor(np.array([t[5] for t in replay_buffer])).to(device)
        old_log_probs = torch.FloatTensor(np.array([t[6] for t in replay_buffer])).to(device)
        old_values = torch.FloatTensor(np.array([t[7] for t in replay_buffer])).to(device)
        return states, actions, rewards, next_states, dones, terminated, old_log_probs, old_values

    def computingAdvantage(self, policy, rewards, next_states, dones, terminated, old_values):
        """
        Calculate multi-objective advantage function
        """
        with torch.no_grad():
            _, next_values = policy(next_states)
            next_values *= (1 - terminated.unsqueeze(-1))  # Termination status set to zero
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        for obj_idx in range(self.num_objectives):
            running_gae = 0
            for t in reversed(range(self.buffer_size)):
                next_value = next_values[t, obj_idx]
                delta = rewards[t, obj_idx] + self.gamma * next_value * (1 - terminated[t]) - old_values[t, obj_idx]
                running_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * running_gae
                advantages[t, obj_idx] = running_gae
                returns[t, obj_idx] = advantages[t, obj_idx] + old_values[t, obj_idx]
        advantages = (advantages - advantages.mean(0)) / (advantages.std(0) + 1e-8)
        return advantages, returns

    def update(self, policy, optimizer, omega, policy_id, process_id, state_norm, reward_scal):
        # Rollout
        states, actions, rewards, next_states, dones, terminated, old_log_probs, old_values \
            = self.collectData(policy, policy_id, process_id, state_norm, reward_scal)
        advantages, returns = self.computingAdvantage(policy, rewards, next_states, dones, terminated, old_values)
        # Scalar advantage
        weighted_advantages = (advantages * torch.FloatTensor(omega).to(device)).sum(dim=1)
        self.agent.updatePolicy_Value(policy, optimizer, states, actions, next_states, old_log_probs,
                                      weighted_advantages,
                                      returns, old_values)

    def _worker_warmup(self, args):
        omega, policy_id, state_norm, reward_scal = args
        process_id = policy_id
        device_index = torch.cuda.current_device() if device.type == 'cuda' else 0
        torch.cuda.set_device(device_index)  # Use clear device indexes
        policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5 if self.set_adam_eps else 1e-8)

        print(f"===== Warm-up of start policy {policy_id} , weights: {omega} =======")
        for _ in range(self.m_w):
            self.update(policy, optimizer, omega, policy_id, process_id, state_norm, reward_scal)

        policy.to("cpu")
        for param in optimizer.state.values():
            for k, v in param.items():
                if isinstance(v, torch.Tensor):
                    param[k] = v.cpu()

        # Return serializable results
        return {
            'policy_state': policy.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'policy_id': policy_id,
            'omega': omega,
            'state_norm': state_norm,
            'reward_scal': reward_scal
        }

    def warmup(self, omega, pool):
        """Multiprocess version warm-up"""
        args = [(omega[i], i, Normalization(shape=self.state_dim),
                 RewardScaling(shape=self.num_objectives, gamma=self.gamma)) for i in range(len(omega))]
        results = pool.map(self._worker_warmup, args)
        # 主进程重建对象
        state_norms = []
        reward_scals = []
        initial_policies = []
        initial_optimizers = []
        policy_ids = []
        for res in results:
            policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives).to("cpu")
            policy.load_state_dict(res['policy_state'])
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5 if self.set_adam_eps else 1e-8)
            optimizer.load_state_dict(res['optimizer_state'])
            state_norms.append(res['state_norm'])
            reward_scals.append(res['reward_scal'])
            initial_policies.append(policy)
            initial_optimizers.append(optimizer)
            policy_ids.append(res['policy_id'])

        self.update_population(list(zip(initial_policies, initial_optimizers)), policy_ids, state_norms)
        return state_norms, reward_scals

    def select_policies(self, p_a):
        """
        Select p_a policies from the population to update
        """
        candidates = []
        total_elements = 0
        for group in self.population:
            if len(group) > 0:
                group_candidates = deepcopy(group[:self.Top_k])
                candidates.append(group_candidates)
                total_elements += len(group_candidates)
        if total_elements < p_a:
            return [element for sublist in candidates for element in sublist]
        m = len(candidates)
        if m >= p_a:
            selected_sublists = random.sample(candidates, p_a)
            return [random.choice(sub) for sub in selected_sublists]
        else:
            selected_initial = []
            remaining_elements = []
            for sublist in candidates:
                chosen = random.choice(sublist)
                selected_initial.append(chosen)
                remaining = [element for element in sublist if element != chosen]
                remaining_elements.extend(remaining)
            additional = random.sample(remaining_elements, p_a - m)
            return selected_initial + additional

    def solve_pareto_weights(self, grads):
        """
            Pareto-ascent direction (optimization problem) solver
        """
        if self.grad_normalize:
            for i in range(self.num_objectives):
                grads[i] /= np.linalg.norm(grads[i])
        alpha = self.PA.solve(np.array(grads))
        return alpha

    def obtain_alpha_star(self, policy, optimizer, policy_id, process_id, state_norm, reward_scal):
        """
            Calculate alpha^* based on our Appendix A.1
        """
        # Rollout
        states, actions, rewards, next_states, dones, terminated, old_log_probs, old_values = \
            self.collectData(policy, policy_id, process_id, state_norm, reward_scal)
        advantages, returns = self.computingAdvantage(policy, rewards, next_states, dones, terminated, old_values)
        # Calculate policy gradient
        grads = []
        for i in range(self.num_objectives):
            # Freeze value network parameters
            for param in policy.fc2.parameters():
                param.requires_grad = False

            # Forward computation (only policy network)
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
            # Restore value network gradient
            for param in policy.fc2.parameters():
                param.requires_grad = True
            grads.append(torch.cat(grad).detach().cpu().numpy())

        if self.num_objectives == 2:
            grad1, grad2 = grads
            if self.grad_normalize:
                grad1 /= np.linalg.norm(grad1)
                grad2 /= np.linalg.norm(grad2)
            numerator = np.dot(grad2 - grad1, grad2)
            denominator = np.linalg.norm(grad1 - grad2) ** 2 + 1e-8
            alpha1 = max(0.0, min(1.0, numerator / denominator))
            alpha_star = np.array([alpha1, 1 - alpha1])
        else:
            alpha_star = self.solve_pareto_weights(grads)
        return alpha_star

    def update_non_dominated_set(self, policy_optimizers, policy_ids, state_norms):
        """
            Update non-dominated set self.non_dominated_set
        """
        new_entries = []
        for (policy, optimizer), policy_id in zip(policy_optimizers, policy_ids):
            new_policy = deepcopy(policy).to(device)
            policy_id = deepcopy(policy_id)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr,
                                             eps=1e-5 if self.set_adam_eps else 1e-8)

            new_optimizer.load_state_dict(optimizer.state_dict())
            obj = self.agent.evaluate_policy(new_policy, state_norms[policy_id])
            new_policy.to("cpu")
            for param in new_optimizer.state.values():
                for k, v in param.items():
                    if isinstance(v, torch.Tensor):
                        param[k] = v.cpu()
            new_entries.append((new_policy, new_optimizer, obj, policy_id))
        for new_entry in new_entries:
            to_remove = []
            add_flag = True
            for idx, entry in enumerate(self.non_dominated_set):
                if is_dominated(new_entry[2], entry[2]):
                    add_flag = False
                    break
                elif is_dominated(entry[2], new_entry[2]):
                    to_remove.append(idx)
            for idx in reversed(to_remove):
                del self.non_dominated_set[idx]
            if add_flag:
                self.non_dominated_set.append(new_entry)
        print(f" Pareto policies size: {len(self.non_dominated_set)}")

    def _worker_iteration(self, args):
        policy_state, optimizer_state, policy_id, process_id, state_norm, reward_scal, omega = args

        state_norm = deepcopy(state_norm)
        reward_scal = deepcopy(reward_scal)

        device_index = torch.cuda.current_device() if device.type == 'cuda' else 0
        torch.cuda.set_device(device_index)
        policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives).to('cpu')
        policy.load_state_dict(policy_state)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)
        optimizer.load_state_dict(optimizer_state)

        # Move the model to the GPU or CPU
        policy = policy.to(device)

        # Ensure that the optimizer status is on the GPU.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if omega is None:
            alpha_star = self.obtain_alpha_star(policy, optimizer, policy_id, process_id, state_norm, reward_scal)
        else:
            alpha_star = omega
        print(f"Policy {policy_id} || alpha_star {list(alpha_star)}")
        intermediate_policy_states = []
        intermediate_optimizer_states = []
        for i in range(self.m):
            self.update(policy, optimizer, alpha_star, policy_id, process_id, state_norm, reward_scal)
            new_policy = deepcopy(policy).to("cpu")
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr,
                                             eps=1e-5 if self.set_adam_eps else 1e-8)
            new_optimizer.load_state_dict(optimizer.state_dict())
            for state in new_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
            if i % 4 == 0:
                intermediate_policy_states.append(new_policy.state_dict())
                intermediate_optimizer_states.append(new_optimizer.state_dict())
        policy = policy.to("cpu")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        # Return serialization results
        return {
            'policy_state': policy.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'policy_id': policy_id,
            'alpha_star': alpha_star,
            'state_norm': state_norm,
            'reward_scal': reward_scal,
            'intermediate_policy_states': intermediate_policy_states,
            'intermediate_optimizer_states': intermediate_optimizer_states
        }

    def PAFT(self, p_b):
        """
            The Pareto adaptive fine-tuning (PAFT) method described in the original paper.
        """
        n = len(self.non_dominated_set)
        if n < 2:
            return [], []
        neighbor_pairs = []
        for i in range(n):
            min_dist = float('inf')
            nearest = -1
            for j in range(n):
                if j == i:
                    continue
                dist = np.linalg.norm(self.non_dominated_set[i][2] - self.non_dominated_set[j][2])
                if dist < min_dist:
                    min_dist = dist
                    nearest = j
            if nearest != -1:
                neighbor_pairs.append((i, nearest, min_dist))
        unique_pairs = set()
        for i, j, d in neighbor_pairs:
            if (j, i, d) not in unique_pairs:
                unique_pairs.add((i, j, d))
        selected_pairs = sorted(unique_pairs, key=lambda x: -x[2])[:p_b // 2]
        policy_optimizers = []
        opposite_directions = []
        for pair in selected_pairs:
            i, j, _ = pair
            s_i = self.non_dominated_set[i]
            s_j = self.non_dominated_set[j]
            dir_i = [1 if s_i[2][k] < s_j[2][k] else 0 for k in range(self.num_objectives)]
            dir_j = [1 if s_j[2][k] < s_i[2][k] else 0 for k in range(self.num_objectives)]
            policy_optimizers.extend([s_i, s_j])
            opposite_directions.append(dir_i)
            opposite_directions.append(dir_j)
        omega = np.array(opposite_directions)
        return policy_optimizers, omega / omega.sum(axis=1).reshape(-1, 1)

    def PAFT_update(self, p_b, state_norms, reward_scals, pool):
        """
            Update self.non_dominated_set using the PAFT method.
        """
        selected_policies, omegas = self.PAFT(p_b)
        if len(selected_policies) > 0:
            policies, optimizers, policy_ids, intermediate_policies, intermediate_optimizers, \
                intermediate_policy_ids = self.multi_process_iteration(selected_policies, state_norms, reward_scals,
                                                                       pool,
                                                                       omegas=omegas)

            self.update_non_dominated_set(list(zip(intermediate_policies, intermediate_optimizers)),
                                          intermediate_policy_ids, state_norms)
            self.update_population(list(zip(policies, optimizers)), policy_ids, state_norms)

    def HV(self):
        if not self.non_dominated_set:
            return 0.0
        objs = np.array([entry[2] for entry in self.non_dominated_set])
        ref_point = np.zeros((self.num_objectives,))
        HV = hypervolume.InnerHyperVolume(ref_point)
        return HV.compute(objs)

    def SP(self):
        if len(self.non_dominated_set) < 2:
            return 0.0
        objs = np.array([entry[2] for entry in self.non_dominated_set])
        sp = 0.0
        for i in range(self.num_objectives):
            sorted_obj = np.sort(objs[:, i])
            sp += np.sum((sorted_obj[1:] - sorted_obj[:-1]) ** 2)
        return sp / (len(objs) - 1)

    def multi_process_iteration(self, selected_policies, state_norms, reward_scals, pool, omegas=None):
        """
            Multiprocess training of selected policies
        """
        args = []
        for process_id, element in enumerate(selected_policies):
            policy, optimizer, _, pid = element
            args.append((policy.state_dict(), optimizer.state_dict(), pid, process_id, state_norms[pid],
                         reward_scals[pid], None if omegas is None else omegas[process_id]))
        results = pool.map(self._worker_iteration, args)
        policies = []
        optimizers = []
        policy_ids = []
        intermediate_policies = []
        intermediate_optimizers = []
        intermediate_policy_ids = []
        for res in results:
            policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives).to("cpu")
            policy.load_state_dict(res['policy_state'])
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr,
                                         eps=1e-5 if self.set_adam_eps else 1e-8)
            optimizer.load_state_dict(res['optimizer_state'])
            state_norms[res['policy_id']] = deepcopy(res['state_norm'])
            reward_scals[res['policy_id']] = deepcopy(res['reward_scal'])
            policies.append(policy)
            optimizers.append(optimizer)
            policy_ids.append(res['policy_id'])
            intermediate_policies.append(policy)
            intermediate_optimizers.append(optimizer)
            intermediate_policy_ids.append(res['policy_id'])
            if self.use_intermediate_policy:
                for policy_state, optimizer_state in zip(res['intermediate_policy_states'],
                                                         res['intermediate_optimizer_states']):
                    policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives).to("cpu")
                    policy.load_state_dict(policy_state)
                    optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr,
                                                 eps=1e-5 if self.set_adam_eps else 1e-8)
                    optimizer.load_state_dict(optimizer_state)
                    intermediate_policies.append(policy)
                    intermediate_optimizers.append(optimizer)
                    intermediate_policy_ids.append(res['policy_id'])
        return policies, optimizers, policy_ids, intermediate_policies, intermediate_optimizers, intermediate_policy_ids

    # ---------------------------> Training <-----------------------------
    def train(self):
        # Obtain the initial population during the warm-up phase
        omega = []
        generate_weights_batch_dfs(0, self.num_objectives, 0, 1, 0.2, [], omega)
        print(f"Needed warm-up policies number: {len(omega)}")
        ctx = mp.get_context('spawn')
        pool_size = min(len(omega), 21)
        with ctx.Pool(processes=pool_size) as pool:
            state_norms, reward_scals = self.warmup(omega, pool)
            with tqdm(total=self.M, desc="PA2D-MORL Training", unit="gen") as pbar:
                for gen in range(self.M):
                    print(f"====== generation: {gen} start iterating, Population size: {self.population_size} ======")
                    if gen < self.M_ft:
                        p_a = self.p
                        p_b = 0
                    else:
                        p_a = p_b = int(self.p / 2)
                    # Select p_a policies from self.population
                    print(f" ========= start select policies =========")
                    selected_policies = self.select_policies(p_a)
                    print(f" ========= {len(selected_policies)} policies are selected =========")

                    policies, optimizers, policy_ids, intermediate_policies, intermediate_optimizers, \
                        intermediate_policy_ids = self.multi_process_iteration(selected_policies, state_norms,
                                                                               reward_scals,
                                                                               pool)

                    # Update non-dominated sets and populations
                    self.update_non_dominated_set(list(zip(intermediate_policies, intermediate_optimizers)),
                                                  intermediate_policy_ids, state_norms)
                    self.update_population(list(zip(policies, optimizers)), policy_ids, state_norms)

                    if gen >= self.M_ft:
                        print(f"=============== start PAFT Update ================")
                        self.PAFT_update(p_b, state_norms, reward_scals, pool)
                    # Record performance metrics at the end of each generation.
                    current_hv = self.HV()
                    current_sp = self.SP()
                    self.hv_history.append(current_hv)
                    self.sp_history.append(current_sp)
                    print(f"generation: {gen}, HV: {current_hv}, SP: {current_sp}, "
                          f",Population size: {self.population_size}, Pareto front size: {len(self.non_dominated_set)}")
                    pbar.update(1)


# Run
if __name__ == "__main__":
    max_episode_steps = 1000
    env = hopper_3.Hopper_3(max_episode_steps=max_episode_steps)
    eval_env = hopper_3.Hopper_3(max_episode_steps=max_episode_steps)
    obj_num = 3
    morl = PA2D_MORL(env, eval_env, obj_num, seed=42)
    print(f"Training on device: {device}")
    tracemalloc.start()
    p = psutil.Process(os.getpid())
    p.cpu_percent(interval=None)
    morl.train()
    cpu_usage = p.cpu_percent(interval=None) / psutil.cpu_count()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"memory peak: {peak / 1024 / 1024:.2f} MB")
    print(f"CPU usage (approximate avg/core): {cpu_usage:.2f}%")

    # print("\nSaving trained policies...")
    # morl.save_policies()
    print("Generating visualizations...")
    morl.visualize_training()
    morl.visualize_pareto_front()
    print("All done!")
