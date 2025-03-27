import os
import random
import numpy as np
from deap import base, creator, tools, algorithms

# -------------------------------
# 遗传算法参数配置
# -------------------------------
GA_POP_SIZE = 100  # 种群大小
GA_GEN_NUM = 200  # 进化代数
CX_PROB = 0.8  # 交叉概率
MUT_PROB = 0.2  # 变异概率
TOURNAMENT_SIZE = 3  # 锦标赛选择大小


# -------------------------------
# 遗传算法核心实现
# -------------------------------
def setup_ga(processing_times):
    # 定义适应度函数（最小化makespan）
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 初始化工具箱
    toolbox = base.Toolbox()
    nb_jobs = len(processing_times)

    # 定义个体生成器
    toolbox.register("indices", random.sample, range(nb_jobs), nb_jobs)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算子
    toolbox.register("evaluate", lambda ind: (compute_makespan(ind, processing_times),))
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", tools.cxOrdered)  # 顺序交叉
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / nb_jobs)  # 随机交换

    return toolbox


def ga_algorithm(processing_times):
    toolbox = setup_ga(processing_times)
    pop = toolbox.population(n=GA_POP_SIZE)

    # 统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 进化主循环
    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=GA_GEN_NUM,
        stats=stats,
        verbose=True
    )

    # 提取最优解
    best_ind = tools.selBest(pop, k=1)[0]
    return best_ind


# -------------------------------
# 公共函数（与NEH代码共享）
# -------------------------------
def compute_makespan(perm, processing_times):
    """计算排列的makespan（与NEH代码一致）"""
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
def solve_with_ga(instance_name):
    # 加载问题实例
    problem = FlowShopProblem(instance_name)
    processing_data = [
        [problem.processing_matrix[m][j] for m in range(problem.num_machines)]
        for j in range(problem.num_jobs)
    ]

    # 执行GA算法
    best_sequence = ga_algorithm(processing_data)

    # 结果验证
    ga_makespan = problem.evaluate(best_sequence)
    calculated_makespan = compute_makespan(best_sequence, processing_data)

    # 输出结果
    print(f"\nGA算法求解结果 - 实例 {instance_name}")
    print("=" * 40)
    print(f"最优排列: {best_sequence}")
    print(f"计算完工时间: {ga_makespan}")
    print(f"验证完工时间: {calculated_makespan}")
    print("验证状态: " + ("成功" if ga_makespan == calculated_makespan else "失败"))

    return best_sequence


# -------------------------------
# 执行主程序
# -------------------------------
if __name__ == "__main__":
    # 测试Taillard实例
    instance = "ta081"
    solve_with_ga(instance)