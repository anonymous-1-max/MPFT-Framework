import os
import random
from copy import deepcopy

from matplotlib import pyplot as plt
import torch
import TD7
import numpy as np
from tqdm import tqdm
from environments import ant, half_cheetah, hopper_2, hopper_3, humanoid, swimmer, walker2d
import hypervolume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dominated(obj1, obj2):
    return np.all(obj2 >= obj1) and np.any(obj2 > obj1)


class TD7_EDGE:
    def __init__(self, environment, eval_env, num_objectives, lr=3e-4, seed=42, warm_up_steps=0, Xi=4000, Psi=1200,
                 edge_direction=0, steps=2000, pareto_ascent_num=2, opposite_num=1,
                 J_max=np.array([1000., 1000.]), use_seed=False):
        self.edge_direction = edge_direction
        self.pareto_ascent_num = pareto_ascent_num
        self.opposite_num = opposite_num
        self.pareto_front_size = Psi
        self.obj_episodes = Xi
        self.warm_up = warm_up_steps
        self.steps = steps
        self.reward_max = J_max
        self.clear_buffer = False
        self.seed = seed
        self.lr = lr
        self.env = environment
        self.eval_env = eval_env
        if use_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.action_space.seed(self.seed)
            self.eval_env.action_space.seed(self.seed + 2025)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.num_objectives = num_objectives
        self.max_action = float(self.env.action_space.high[0])
        self.agent = TD7.Agent(self.state_dim, self.action_dim, self.max_action, num_obj=num_objectives)
        # Pareto-approximation policy set self.non_dominated_set, whose elements are: (policy, Q, optimizer, objectives)
        self.non_dominated_set = []
        self.hv_history = []
        self.sp_history = []

    def save_policies(self, obj, save_dir="saved_policies/TD7/"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"ParetoEdge_{obj}_encoder.pth"
        torch.save(self.agent.fixed_encoder.state_dict(), os.path.join(save_dir, filename))
        for idx, (policy, q_net, _, objectives) in enumerate(self.non_dominated_set):
            obj_str = "_".join([f"{o:.2f}" for o in objectives])
            filename1 = f"/ParetoEdge_{obj}_policy_{idx}_{obj_str}.pth"
            filename2 = f"/ParetoEdge_{obj}_Qnet_{idx}_{obj_str}.pth"
            torch.save(policy.state_dict(), os.path.join(save_dir, filename1))
            torch.save(q_net.state_dict(), os.path.join(save_dir, filename2))

    def visualize_training(self, save_path="training_metrics"):
        save_path += f"_Pareto_Edge_{self.edge_direction + 1}.png"
        print(f"HV: {self.hv_history}")
        print(f"SP: {self.sp_history}")
        plt.figure(figsize=(12, 6))
        # HV
        plt.subplot(1, 2, 1)
        plt.plot(self.hv_history, marker='o', linestyle='-', color='b')
        plt.title(f"Pareto Edge{self.edge_direction + 1}: Hypervolume (HV) Trend")
        plt.xlabel("Episodes")
        plt.ylabel("HV Value")
        plt.grid(True)
        # SP
        plt.subplot(1, 2, 2)
        plt.plot(self.sp_history, marker='o', linestyle='-', color='r')
        plt.title(f"Pareto Edge{self.edge_direction + 1}: Spread (SP) Trend")
        plt.xlabel("Episodes")
        plt.ylabel("SP Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_pareto_front(self, save_path="Pareto_front"):
        save_path += f"_Pareto_Edge_{self.edge_direction + 1}.png"
        if not self.non_dominated_set:
            return
        objectives = np.array([entry[3] for entry in self.non_dominated_set])
        print(f"Objectives: {list(objectives)}")
        plt.figure(figsize=(8, 6))
        if self.num_objectives == 2:
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50, edgecolors='k')
            plt.title(f"Pareto Edge {self.edge_direction + 1}, 2D Pareto Front")
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
        elif self.num_objectives == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_title(f"Pareto Edge {self.edge_direction + 1}, 3D Pareto Front")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def update_non_dominated_set(self, policy_Qnet_optimizers):
        """
        Update self.non_dominated_set
        :param policy_Qnet_optimizers:
        :return:
        """
        new_entries = []
        for policy, Qnet, optimizer in policy_Qnet_optimizers:
            new_policy = deepcopy(policy)
            new_Qnet = deepcopy(Qnet)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr)
            new_optimizer.load_state_dict(optimizer.state_dict())
            objectives = self.agent.evaluate_policy(self.eval_env, self.seed)
            new_policy.to("cpu")
            new_Qnet.to("cpu")
            for param in new_optimizer.state.values():
                for k, v in param.items():
                    if isinstance(v, torch.Tensor):
                        param[k] = v.cpu()
            new_entries.append((new_policy, new_Qnet, new_optimizer, objectives))
        for new_entry in new_entries:
            to_remove = []
            add_flag = True
            for idx, entry in enumerate(self.non_dominated_set):
                if is_dominated(new_entry[3], entry[3]):
                    add_flag = False
                    break
                elif is_dominated(entry[3], new_entry[3]):
                    to_remove.append(idx)
            if add_flag:
                for idx in reversed(to_remove):
                    del self.non_dominated_set[idx]
                self.non_dominated_set.append(new_entry)
        print(f"TD7 Pareto Edge {self.edge_direction + 1}, pareto policies size: {len(self.non_dominated_set)}")

    def HV(self):
        if not self.non_dominated_set:
            return 0.0
        objs = np.array([entry[3] for entry in self.non_dominated_set])
        ref_point = np.zeros((self.num_objectives,))
        HV = hypervolume.InnerHyperVolume(ref_point)
        return HV.compute(objs)

    def SP(self):
        if len(self.non_dominated_set) < 2:
            return 0.0
        objs = np.array([entry[3] for entry in self.non_dominated_set])
        sp = 0.0
        for i in range(self.num_objectives):
            sorted_obj = np.sort(objs[:, i])
            sp += np.sum((sorted_obj[1:] - sorted_obj[:-1]) ** 2)
        return sp / (len(objs) - 1)

    def update_ParetoFront(self, use_checkpoints=False):
        """
        The Pareto-tracking mechanism in the paper.
        In the implementation, we also use the intermediate results to update the  Pareto-approximation policy set.
        """
        for _ in range(self.opposite_num):
            self.updateAEpisode(omega=None, obj_id=self.edge_direction, update_info="++++omega1",
                                use_checkpoints=use_checkpoints)  # Pareto-reverse direction of objective 'obj_id'
            self.update_non_dominated_set([(self.agent.actor, self.agent.critic, self.agent.actor_optimizer)])
        for _ in range(self.pareto_ascent_num):
            self.updateAEpisode(omega=None, obj_id=-1, update_info="++++omega2",
                                use_checkpoints=use_checkpoints)  # Pareto-ascent direction
            self.update_non_dominated_set([(self.agent.actor, self.agent.critic, self.agent.actor_optimizer)])
        self.agent.evaluate_policy(self.eval_env, self.seed)

    def updateAEpisode(self, omega, obj_id, update_info, use_checkpoints=True):
        """
        An episode update
        """
        state = self.env.reset()
        ep_total_reward, ep_timesteps = np.zeros((self.num_objectives,)), 0
        t = 0
        done = False
        while t < self.steps or (not done):
            action = self.agent.select_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            ep_total_reward += reward
            ep_timesteps += 1
            self.agent.buffer.add(state, action, next_state, reward, terminated)
            state = next_state
            if not use_checkpoints:
                self.agent.train(omega=omega, obj_id=obj_id, update_info=update_info)
            if done:
                state = self.env.reset()
                if use_checkpoints:
                    self.agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward, omega=omega,
                                                          obj_id=obj_id, update_info=update_info)
                ep_total_reward, ep_timesteps = 0, 0
            t += 1

    # ---------------------------> Training <-----------------------------
    def train(self):
        with tqdm(total=self.obj_episodes + self.pareto_front_size,
                  desc=f"MPFT-MOTD7 Edge {self.edge_direction + 1} Training") as pbar:
            # Searching for Pareto-edge policies
            for i in range(self.obj_episodes):
                rewards = self.agent.evaluate_policy(self.eval_env, self.seed)
                if i < self.warm_up:
                    # Use objective weight adjustment method to warmup
                    r = self.reward_max / rewards
                    omega = TD7.getOmega(r)
                else:
                    omega = np.zeros((self.num_objectives,))
                    omega[self.edge_direction] = 1
                print(f"========== Edge {self.edge_direction + 1}, Episodes {i}, omega {list(omega)} =========")
                self.updateAEpisode(omega=omega, obj_id=-1, update_info="", use_checkpoints=True)
                if i > self.obj_episodes / 2:
                    self.update_non_dominated_set([(self.agent.actor, self.agent.critic, self.agent.actor_optimizer)])
                pbar.update(1)
            # Pareto front tracking
            if self.clear_buffer:
                self.agent.buffer.clear_buffer()  # Clear buffer so that previous data does not affect tracking
            for i in range(self.pareto_front_size):
                self.update_ParetoFront(use_checkpoints=True)
                if i % 10 == 0:
                    current_hv = self.HV()
                    current_sp = self.SP()
                    self.hv_history.append(current_hv)
                    self.sp_history.append(current_sp)
                    print(f"Edge{self.edge_direction + 1}: | HV:{current_hv}, SP:{current_sp}")
                pbar.update(1)
            self.visualize_training()
            self.visualize_pareto_front()
