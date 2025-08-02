import multiprocessing as mp
import os
import random
from copy import deepcopy
from multiprocessing import Process, Queue
import numpy as np
import torch
from matplotlib import pyplot as plt
import TD7_edge
import hypervolume
import TD7_inner
from environments import ant, half_cheetah, hopper_2, hopper_3, humanoid, swimmer, walker2d
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
import argparse

# Set up global devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dominated(obj1, obj2):
    """ Is obj1 dominated by obj2? """
    return np.all(obj2 >= obj1) and np.any(obj2 > obj1)


def run_process(target_func, args, q, idx):
    result = target_func(*args)
    q.put((idx, result))


def find_top_k_sparse_regions(arr, topK=2):
    """
    Find the top K sparse regions in the Pareto Front:

   - Objective number = 2: Use sorting + neighboring point distance calculation;
   - Objective number = 3: Project the three-dimensional points onto the best-fit plane (using PCA);
                        use Delaunay triangulation on the two-dimensional plane to obtain a triangular mesh;
                        Calculate the area of each triangle (larger areas indicate greater sparsity);
    Parameters:
        arr: shape (n, m), objective value array of the Pareto Front,
        where n is the number of solutions and m is the number of objectives
        k: the number of Top K sparse regions to find
    Returns:
        list of tuples: each tuple contains (point 1, point 2,....)
    """
    n, m = arr.shape
    if m == 2:
        sorted_arr = arr[arr[:, 0].argsort()]
        distances = np.linalg.norm(sorted_arr[1:] - sorted_arr[:-1], axis=1)

        # Find the indices of the top K the largest distances
        top_k_indices = np.argsort(distances)[::-1][:topK]

        # Build results
        result = []
        for idx in top_k_indices:
            p1 = sorted_arr[idx]
            p2 = sorted_arr[idx + 1]
            result.append([p1, p2])
    else:
        # 3-dimensional situation
        result = find_sparse_regions_3_dim(arr, topK)
    return np.array(result)


def project_to_plane(arr):
    """
    Use PCA to project 3D points onto the best-fitting 2D plane
    Returns:
        arr_proj: Projected 2D points (N, 2)
        pca: PCA model, which can be used to restore back to 3D
    """
    pca = PCA(n_components=2)
    arr_proj = pca.fit_transform(arr)
    return arr_proj, pca


def find_sparse_regions_3_dim(arr, topK=1):
    arr_2d, pca = project_to_plane(arr)
    tri = Delaunay(arr_2d)
    simplices = tri.simplices  # Triangular index (M, 3)

    def area(tri_2d):
        a, b, c = tri_2d
        return 0.5 * np.abs(np.cross(b - a, c - a))

    triangles_2d = arr_2d[simplices]
    areas = np.array([area(tri) for tri in triangles_2d])
    topK_idx = np.argsort(areas)[-topK:]
    sparse_tris_2d = triangles_2d[topK_idx]
    sparse_tris_3d = [pca.inverse_transform(tri) for tri in sparse_tris_2d]
    return np.asarray(sparse_tris_3d)


class mpft_motd7:
    def __init__(self, environment, eval_env, num_objectives, lr=3e-4, seed=42, steps=2000, use_seed=False):
        self.edge1 = None
        self.edge2 = None
        self.edge3 = None
        self.inner = None
        self.topK = 3
        self.pareto_ascent_num = 2  # The hyperparameter v in the paper
        self.opposite_num = 1  # The hyperparameter u in the paper
        self.edge_direction = [0, 1, 2]
        # Policy warm-up episodes.
        # The purpose of warm-up is to enable the policy to escape from its initial random state.
        self.Warmup_steps = [600, 600]
        self.Xi = [800, 800, 800]
        self.Psi = [800, 800, 800]
        self.J_max = [np.array([2000, 1200]), np.array([1200, 2000]), np.array([1800, 2600])]
        self.use_priority_knowledge = False
        self.seed = seed
        self.lr = lr
        self.steps = steps
        self.use_seed = use_seed
        self.env = environment
        self.eval_env = eval_env
        if self.use_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.action_space.seed(self.seed)  # 动作空间的种子
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.num_objectives = num_objectives
        # Pareto-approximation policy set self.non_dominated_set, whose elements are: (policy, Q, optimizer, objectives)
        self.non_dominated_set = []

    # Save policies
    def save_policies(self, save_dir="saved_policies"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, (policy, _, _, objectives) in enumerate(self.non_dominated_set):
            obj_str = "_".join([f"{o:.2f}" for o in objectives])
            filename = f"policy_{idx}_{obj_str}.pth"
            torch.save(policy.state_dict(), os.path.join(save_dir, filename))

    def visualize_pareto_front(self, save_path="pareto_front"):
        """Plot Pareto front"""
        save_path += f"_pareto_combine.png"
        if not self.non_dominated_set:
            return
        objectives = np.array([entry[3] for entry in self.non_dominated_set])
        print(f"Objectives: {list(objectives)}")
        plt.figure(figsize=(8, 6))
        if self.num_objectives == 2:
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50, edgecolors='k')
            plt.title(f"2D Pareto Front")
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
        elif self.num_objectives == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_title(f"3D Pareto Front")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def update_non_dominated_set(self, Pareto_set):
        """
            Update self.non_dominated_set
            :return:
        """
        new_entries = []
        for policy, q_net, optimizer, objectives in Pareto_set:
            new_policy = deepcopy(policy)
            new_q_net = deepcopy(q_net)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr)
            new_optimizer.load_state_dict(optimizer.state_dict())
            objectives = deepcopy(objectives)
            new_entries.append((new_policy, new_q_net, new_optimizer, objectives))
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
        print(f"Pareto policies size: {len(self.non_dominated_set)}")

    def getJMax(self):
        arr = np.array([obj[3] for obj in self.non_dominated_set])
        n, m = arr.shape
        if n < 3:
            return np.array([m * [3000]] * self.topK)
        result = find_top_k_sparse_regions(arr, self.topK)
        J_max = np.max(result, axis=1)
        return J_max

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

    def initialize_environment(self, ):
        self.edge1 = TD7_edge.TD7_EDGE(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                                       warm_up_steps=self.Warmup_steps[0], Xi=self.Xi[0], Psi=self.Psi[0],
                                       edge_direction=self.edge_direction[0], steps=self.steps,
                                       pareto_ascent_num=self.pareto_ascent_num,
                                       opposite_num=self.opposite_num, J_max=self.J_max[0], use_seed=self.use_seed)
        self.edge2 = TD7_edge.TD7_EDGE(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                                       warm_up_steps=self.Warmup_steps[1], Xi=self.Xi[1], Psi=self.Psi[1],
                                       edge_direction=self.edge_direction[1], steps=self.steps,
                                       pareto_ascent_num=self.pareto_ascent_num,
                                       opposite_num=self.opposite_num, J_max=self.J_max[1], use_seed=self.use_seed)
        self.inner = TD7_inner.TD7_INNER(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                                         Xi_k=self.Xi[2], Psi_k=self.Psi[2], steps=self.steps,
                                         pareto_ascent_num=self.pareto_ascent_num,
                                         opposite_num=self.opposite_num, J_max=self.J_max[2], use_seed=self.use_seed)
        if self.num_objectives == 3:
            self.edge3 = TD7_edge.TD7_EDGE(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                                           warm_up_steps=self.Warmup_steps[2], Xi=self.Xi[3], Psi=self.Psi[3],
                                           edge_direction=self.edge_direction[2], steps=self.steps,
                                           pareto_ascent_num=self.pareto_ascent_num,
                                           opposite_num=self.opposite_num, J_max=self.J_max[3], use_seed=self.use_seed)

    @staticmethod
    def _train_edge(edge_class, env, eval_env, num_objectives, lr, seed, warm_up_steps, Xi, Psi, edge_direction, steps,
                    pareto_ascent_num, opposite_num, J_max, use_seed):
        """Parallel Training Edge"""
        edge = edge_class(env, eval_env, num_objectives, lr=lr, seed=seed, warm_up_steps=warm_up_steps, Xi=Xi,
                          Psi=Psi, edge_direction=edge_direction, steps=steps,
                          pareto_ascent_num=pareto_ascent_num,
                          opposite_num=opposite_num, J_max=J_max, use_seed=use_seed)
        edge.train()
        return edge.non_dominated_set

    @staticmethod
    def _train_inner(inner_class, env, eval_env, num_objectives, lr, seed, Xi, Psi, steps,
                     pareto_ascent_num, opposite_num, J_max, use_seed):
        """Parallel Training Interior"""
        inner = inner_class(env, eval_env, num_objectives, lr=lr, seed=seed, Xi_k=Xi,
                            Psi_k=Psi, steps=steps,
                            pareto_ascent_num=pareto_ascent_num,
                            opposite_num=opposite_num, J_max=J_max, use_seed=use_seed)
        inner.train()
        return inner.non_dominated_set

    # ---------------------------> Training <-----------------------------
    def train(self):
        task_list = []
        queue = Queue()
        if not self.use_priority_knowledge:
            edge_tasks = [
                (self._train_edge, (TD7_edge.TD7_EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[0], self.Xi[0], self.Psi[0],
                                    self.edge_direction[0], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[0], self.use_seed)),
                (self._train_edge, (TD7_edge.TD7_EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[1], self.Xi[1], self.Psi[1],
                                    self.edge_direction[1], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[1], self.use_seed))
            ]
            if self.num_objectives == 3:
                edge_tasks.append(
                    (self._train_edge, (TD7_edge.TD7_EDGE, self.env, self.eval_env, self.num_objectives,
                                        self.lr, self.seed, self.Warmup_steps[2], self.Xi[3], self.Psi[3],
                                        self.edge_direction[2], self.steps,
                                        self.pareto_ascent_num, self.opposite_num, self.J_max[3], self.use_seed)))
            for idx, (func, args) in enumerate(edge_tasks):
                p = Process(target=run_process, args=(func, args, queue, idx))
                task_list.append(p)
                p.start()
            for _ in range(len(task_list)):
                idx, result = queue.get()
                print(f"+++++++++Edge {idx + 1} is complete+++++++")
                self.update_non_dominated_set(result)

            for p in task_list:
                p.join()

            # Interior
            J_max = self.getJMax()
            task_list = []
            queue = Queue()
            inner_tasks = []
            for i in range(self.topK):
                inner_tasks.append(
                    (self._train_inner, (TD7_inner.TD7_INNER, self.env, self.eval_env, self.num_objectives,
                                         self.lr, self.seed, self.Xi[2], self.Psi[2], self.steps,
                                         self.pareto_ascent_num,
                                         self.opposite_num, J_max[i], self.use_seed)))

            for idx, (func, args) in enumerate(inner_tasks):
                p = Process(target=run_process, args=(func, args, queue, idx))
                task_list.append(p)
                p.start()
            for _ in range(len(task_list)):
                idx, result = queue.get()
                print(f"+++++++++Interior {idx + 1} is complete+++++++")
                self.update_non_dominated_set(result)
            for p in task_list:
                p.join()
        else:
            full_tasks = [
                (self._train_edge, (TD7_edge.TD7_EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[0], self.Xi[0], self.Psi[0],
                                    self.edge_direction[0], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[0], self.use_seed)),
                (self._train_edge, (TD7_edge.TD7_EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[1], self.Xi[1], self.Psi[1],
                                    self.edge_direction[1], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[1], self.use_seed)),
                (self._train_inner, (TD7_inner.TD7_INNER, self.env, self.eval_env, self.num_objectives,
                                     self.lr, self.seed, self.Xi[2], self.Psi[2], self.steps, self.pareto_ascent_num,
                                     self.opposite_num, self.J_max[2], self.use_seed))
            ]
            if self.num_objectives == 3:
                full_tasks.append(
                    (self._train_edge, (TD7_edge.TD7_EDGE, self.env, self.eval_env, self.num_objectives,
                                        self.lr, self.seed, self.Warmup_steps[2], self.Xi[3], self.Psi[3],
                                        self.edge_direction[2], self.steps,
                                        self.pareto_ascent_num, self.opposite_num, self.J_max[3], self.use_seed)))
            for idx, (func, args) in enumerate(full_tasks):
                p = Process(target=run_process, args=(func, args, queue, idx))
                task_list.append(p)
                p.start()
            for _ in range(len(task_list)):
                idx, result = queue.get()
                print(f"+++++++++Task {idx} completed+++++++")
                self.update_non_dominated_set(result)
            for p in task_list:
                p.join()

        self.visualize_pareto_front()


# Run
if __name__ == "__main__":
    seed = 42
    steps = 2000
    use_seed = False
    mp.set_start_method('spawn')
    max_episode_steps = 1000
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant')
    args = parser.parse_args()
    env_name = args.env
    print(f"Run {env_name}")
    if env_name == "HalfCheetah":
        env = half_cheetah.HalfCheetah(max_episode_steps=max_episode_steps)
        eval_env = half_cheetah.HalfCheetah(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Hopper-2":
        env = hopper_2.Hopper_2(max_episode_steps=max_episode_steps)
        eval_env = hopper_2.Hopper_2(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Swimmer":
        env = swimmer.Swimmer(max_episode_steps=max_episode_steps)
        eval_env = swimmer.Swimmer(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Ant":
        env = ant.Ant(max_episode_steps=max_episode_steps)
        eval_env = ant.Ant(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Walker2d":
        env = walker2d.Walker2D(max_episode_steps=max_episode_steps)
        eval_env = walker2d.Walker2D(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Humanoid":
        env = humanoid.Humanoid(max_episode_steps=max_episode_steps)
        eval_env = humanoid.Humanoid(max_episode_steps=max_episode_steps)
        obj_num = 2
    else:
        env = hopper_3.Hopper_3(max_episode_steps=max_episode_steps)
        eval_env = hopper_3.Hopper_3(max_episode_steps=max_episode_steps)
        obj_num = 3

    morl = mpft_motd7(env, eval_env, obj_num, lr=3e-4, seed=seed, steps=steps, use_seed=use_seed)
    morl.initialize_environment()
    print(f"Training on device: {device}")
    morl.train()

    # Save Results
    # print("\nSaving trained policies...")
    # morl.save_policies()
    # print("Generating visualizations...")
    # morl.visualize_training()
    # morl.visualize_pareto_front()
    print("All done!")

# Hopper-3  steps = 2000
# self.Warmup_steps = [200, 50, 300]
# self.Xi = [800, 500, 1000, 400]
# self.Psi = [1200, 1500, 1500, 1400]
# self.J_max = [np.array([4000, 3500, 3000]), np.array([3000, 4200, 4000]), np.array([3000, 2900, 3200]),
#                       np.array([3000., 3000., 3100])]

# Swimmer  steps = 2000
# self.Warmup_steps = [0, 80]
# self.Xi = [100, 100, 100]
# self.Psi = [200, 200, 400]
# self.J_max = [np.array([1800, 1500]), np.array([1500, 1800]), np.array([1650, 1650])]


# HalfCheetah  steps = 2000
# self.Warmup_steps = [80, 80]
# self.Xi = [100, 100, 100]
# self.Psi = [300, 300, 500]
# self.J_max = [np.array([1500, 1500]), np.array([1500, 1500]), np.array([1500, 1700])]


# Walk2D  steps = 2000
# self.Warmup_steps = [70, 0]
# self.Xi = [500, 100, 500]
# self.Psi = [500, 500, 500]
# self.J_max = [np.array([5000., 3000.]), np.array([3000., 4000.]), np.array([3250., 3750.])]


# Hopper-2  steps = 2000
# self.Warmup_steps = [500, 300]
# self.Xi = [1500, 1500, 1000]
# self.Psi = [800, 800, 1000]
# self.J_max = [np.array([5000, 4000]), np.array([4000, 5600]), np.array([4000, 5000])]


# Ant  steps = 2000
# self.Warmup_steps = [600, 600]
# self.Xi = [800, 800, 800]
# self.Psi = [800, 800, 800]
# self.J_max = [np.array([2000, 1200]), np.array([1200, 2000]), np.array([1800, 2600])]


# Humanoid  steps = 2000
# self.Warmup_steps = [200, 0]
# self.Xi = [2500, 600, 1400]
# self.Psi = [1200, 1700, 1400]
# self.J_max = [np.array([8000, 5000]), np.array([6000, 8000]), np.array([5800., 7200.])]
