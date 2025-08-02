import numpy as np
from scipy.optimize import minimize


class ParetoAscentDirection:
    def __init__(self):
        pass

    def objective(self, alpha, grads):
        """
        Objective function: minimize 1/2 * |sum(alpha_i * grad_i)|^2
        alpha: weight vector
        grads: list of gradient vectors
        """
        grad_sum = np.sum(alpha[:, np.newaxis] * grads, axis=0)
        return 0.5 * np.linalg.norm(grad_sum) ** 2

    def constraint1(self, alpha):
        """
        Constraint: sum(alpha_i) = 1
        """
        return np.sum(alpha) - 1

    def solve(self, grads):
        """
        Solve the optimization problem.
        grads: Gradient matrix, dimension (obj_num, feature_dim)
        Return: Optimal alpha, dimension (obj_num)
        """
        obj_num, feature_dim = grads.shape
        # 初始猜测
        alpha_initial = np.ones(obj_num) / obj_num
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': self.constraint1}
        ]
        # 非负约束
        bounds = [(0, None) for _ in range(obj_num)]
        # 优化问题
        result = minimize(
            fun=self.objective,
            x0=alpha_initial,
            args=(grads,),
            constraints=constraints,
            bounds=bounds
        )

        if result.success:
            return result.x
        else:
            return np.ones(obj_num) / obj_num


if __name__ == '__main__':
    # test
    error = 0
    for _ in range(1000):
        grads = np.random.rand(2, 163874) + 100 * np.random.random() - 120 * np.random.random()
        PA = ParetoAscentDirection()
        alpha = PA.solve(grads)
        grad1, grad2 = grads
        numerator = np.dot(grad2 - grad1, grad2)
        denominator = np.linalg.norm(grad1 - grad2) ** 2 + 1e-8
        alpha2 = max(0.0, min(1.0, numerator / denominator))
        alpha_star = np.array([alpha2, 1 - alpha2])
        error += np.sum((alpha - alpha_star) ** 2)
    print(f"error: {error / 1000}")
