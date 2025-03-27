import os
import numpy as np
import random
from pyswarms.single import GlobalBestPSO


# -------------------------------
# PSO 算法核心实现（修正版）
# -------------------------------
from TestSolutions.NEH_evaluation import neh_algorithm


class DiscretePSO:
    def __init__(self, processing_times, n_particles=20, iterations=100):
        self.processing_times = processing_times
        self.n_jobs = len(processing_times)
        self.n_particles = n_particles
        self.iterations = iterations

        # PSO参数（调整后参数）
        self.options = {'c1': 0.7, 'c2': 0.4, 'w': 0.8}  # 修正参数范围
        self.bounds = (np.zeros(self.n_jobs), np.ones(self.n_jobs) * self.n_jobs)

    def _decode(self, position):
        """将连续位置向量转换为作业排列（添加归一化）"""
        return list(np.argsort(position / np.linalg.norm(position)))

    def evaluate(self, positions, **kwargs):
        """兼容参数传递（添加**kwargs）"""
        makespans = []
        for pos in positions:
            permutation = self._decode(pos)
            makespan = compute_makespan(permutation, self.processing_times)
            makespans.append(makespan)
        return np.array(makespans)

    def optimize(self):
        """修正初始化方式"""
        # 初始化粒子群
        initial_pos = np.random.uniform(0, self.n_jobs, (self.n_particles, self.n_jobs))

        # 注入NEH初始解
        neh_sol = neh_algorithm(self.processing_times)
        initial_pos[0] = np.argsort(neh_sol) * (self.n_jobs / len(neh_sol))  # 归一化

        # 创建优化器（修正初始化方式）
        optimizer = GlobalBestPSO(n_particles=self.n_particles,
                                  dimensions=self.n_jobs,
                                  options=self.options,
                                  bounds=self.bounds,
                                  init_pos=initial_pos)  # 正确参数名

        # 执行优化（移除init_pos参数）
        cost, pos = optimizer.optimize(self.evaluate,
                                       iters=self.iterations)
        return self._decode(pos), cost


# ...（保留其他不变部分）...
def compute_makespan(perm, processing_times):
    """（与NEH代码完全一致）"""
    nb_machines = len(processing_times[0])
    timeline = [0] * nb_machines
    for job in perm:
        timeline[0] += processing_times[job][0]
        for m in range(1, nb_machines):
            timeline[m] = max(timeline[m], timeline[m - 1]) + processing_times[job][m]
    return timeline[-1]


class FlowShopProblem:
    """（与NEH代码完全一致）"""

    def __init__(self, instance_name):
        self.instance = instance_name
        self.num_jobs, self.num_machines, self.processing_matrix = self.load_data()

    def load_data(self):
        """加载Taillard或VRF格式数据"""
        if self.instance.startswith('t'):
            filepath = os.path.join("./taillard", f"{self.instance}.dat")
        elif self.instance.startswith('V'):
            filepath = os.path.join("./vrf", f"{self.instance}_Gap.txt")
        else:
            raise ValueError("Unsupported instance type")

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('EOF')]

            # 解析第一行获取作业数和机器数
            num_jobs, num_machines = map(int, lines[0].split())

            # 初始化处理时间矩阵
            pt_matrix = []

            if self.instance.startswith('t'):
                # Taillard格式：每行对应一台机器
                for line in lines[1:num_machines + 1]:
                    pt_matrix.append(list(map(int, line.split())))
            else:
                # VRF格式：每行对应一个作业，偶数列为处理时间
                pt_matrix = [[0] * num_jobs for _ in range(num_machines)]
                for job_idx, line in enumerate(lines[1:num_jobs + 1]):
                    values = list(map(int, line.split()))
                    for m in range(num_machines):
                        pt_matrix[m][job_idx] = values[2 * m + 1]

        return num_jobs, num_machines, pt_matrix

    def evaluate(self, sequence):
        """验证函数（与之前完全一致）"""
        timeline = [0] * self.num_machines
        for job in sequence:
            timeline[0] += self.processing_matrix[0][job]
            for m in range(1, self.num_machines):
                timeline[m] = max(timeline[m], timeline[m - 1]) + self.processing_matrix[m][job]
        return timeline[-1]


# ...（与之前NEH代码相同）...

# -------------------------------
# 主求解流程
# -------------------------------
def solve_with_pso(instance_name):
    # 加载问题实例
    problem = FlowShopProblem(instance_name)
    processing_data = [
        [problem.processing_matrix[m][j] for m in range(problem.num_machines)]
        for j in range(problem.num_jobs)
    ]

    # 配置PSO参数（可根据问题规模调整）
    n_particles = min(50, problem.num_jobs * 2)  # 粒子数量与问题规模相关
    iterations = 100 if problem.num_jobs < 50 else 200

    # 执行PSO算法
    pso = DiscretePSO(processing_data, n_particles=n_particles, iterations=iterations)
    best_sequence, best_makespan = pso.optimize()

    # 结果验证
    validated_makespan = problem.evaluate(best_sequence)

    # 输出结果
    print(f"\nPSO算法求解结果 - 实例 {instance_name}")
    print("=" * 40)
    print(f"最优排列: {best_sequence}")
    print(f"算法计算完工时间: {best_makespan}")
    print(f"验证完工时间: {validated_makespan}")
    print("验证状态: " + ("成功" if best_makespan == validated_makespan else "失败"))

    return best_sequence
# -------------------------------
# 执行主程序
# -------------------------------
if __name__ == "__main__":
    # 测试Taillard实例
    instance = "ta111"
    solve_with_pso(instance)