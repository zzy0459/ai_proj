import os
from collections import defaultdict


# -------------------------------
# NEH 算法核心实现
# -------------------------------
def neh_algorithm(processing_times):
    """NEH 算法主函数"""
    nb_jobs = len(processing_times)
    nb_machines = len(processing_times[0])

    # 步骤1: 计算作业总处理时间
    total_times = [sum(job) for job in processing_times]

    # 步骤2: 降序排列作业索引
    job_order = sorted(range(nb_jobs), key=lambda x: -total_times[x])

    # 步骤3: 逐步插入构建最优排列
    current_sequence = [job_order[0]]
    for idx in range(1, nb_jobs):
        candidate = job_order[idx]
        best_makespan = float('inf')
        best_pos = 0

        # 遍历所有可能插入位置
        for pos in range(len(current_sequence) + 1):
            temp_seq = current_sequence[:pos] + [candidate] + current_sequence[pos:]
            ms = calculate_makespan(temp_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos

        current_sequence = current_sequence[:best_pos] + [candidate] + current_sequence[best_pos:]

    return current_sequence


def calculate_makespan(sequence, processing_times):
    """计算给定排列的总完工时间"""
    nb_machines = len(processing_times[0])
    timeline = [0] * nb_machines

    for job in sequence:
        # 第一台机器的累计时间
        timeline[0] += processing_times[job][0]

        # 后续机器的累计时间
        for m in range(1, nb_machines):
            timeline[m] = max(timeline[m], timeline[m - 1]) + processing_times[job][m]

    return timeline[-1]


# -------------------------------
# 数据加载与验证类（保持不变）
# -------------------------------
class FlowShopProblem:
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


# -------------------------------
# 主求解流程
# -------------------------------
def solve_with_neh(instance_name):
    # 加载问题实例
    problem = FlowShopProblem(instance_name)

    # 转换处理时间格式为作业优先
    processing_data = [
        [problem.processing_matrix[m][j] for m in range(problem.num_machines)]
        for j in range(problem.num_jobs)
    ]

    # 执行NEH算法
    best_sequence = neh_algorithm(processing_data)

    # 结果验证
    neh_makespan = problem.evaluate(best_sequence)
    calculated_makespan = calculate_makespan(best_sequence, processing_data)

    # 输出结果
    print(f"NEH算法求解结果 - 实例 {instance_name}")
    print("=" * 40)
    print(f"最优排列: {best_sequence}")
    print(f"计算完工时间: {neh_makespan}")
    print(f"验证完工时间: {calculated_makespan}")
    print("验证状态: " + ("成功" if neh_makespan == calculated_makespan else "失败"))

    return best_sequence


# -------------------------------
# 执行主程序
# -------------------------------
if __name__ == "__main__":
    # 测试Taillard实例
    instance = "ta092"
    solve_with_neh(instance)