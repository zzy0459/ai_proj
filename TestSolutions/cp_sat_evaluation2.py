import os
from ortools.sat.python import cp_model
from collections import defaultdict


# -------------------------------
# NEH 算法实现
# -------------------------------
def neh_algorithm(processing_times):
    nbJob = len(processing_times)
    nbMach = len(processing_times[0])

    # Step 1: 计算每个作业的总处理时间
    total_times = [sum(job) for job in processing_times]

    # Step 2: 按总处理时间降序排列作业索引
    job_indices = sorted(range(nbJob), key=lambda x: -total_times[x])

    # Step 3: 逐步构建排列
    current_perm = [job_indices[0]]
    for i in range(1, nbJob):
        candidate = job_indices[i]
        best_makespan = float('inf')
        best_position = 0
        # 尝试插入所有可能的位置
        for pos in range(len(current_perm) + 1):
            temp_perm = current_perm[:pos] + [candidate] + current_perm[pos:]
            makespan = compute_makespan(temp_perm, processing_times)
            if makespan < best_makespan:
                best_makespan = makespan
                best_position = pos
        current_perm = current_perm[:best_position] + [candidate] + current_perm[best_position:]
    return current_perm


def compute_makespan(perm, processing_times):
    nbMach = len(processing_times[0])
    machine_times = [0] * nbMach
    for job in perm:
        pt = processing_times[job]
        machine_times[0] += pt[0]
        for m in range(1, nbMach):
            machine_times[m] = max(machine_times[m], machine_times[m - 1]) + pt[m]
    return machine_times[-1]


def compute_start_times(perm, processing_times):
    nbJob = len(processing_times)
    nbMach = len(processing_times[0])
    start_times = [[0] * nbMach for _ in range(nbJob)]
    prev_end = [0] * nbMach
    for job in perm:
        for m in range(nbMach):
            if m == 0:
                start = prev_end[m]
            else:
                start = max(prev_end[m], prev_end[m - 1])
            end = start + processing_times[job][m]
            start_times[job][m] = start
            prev_end[m] = end
    return start_times


# -------------------------------
# fsp 类（优化数据加载）
# -------------------------------
class fsp:
    def __init__(self, instance):
        self.instname = instance
        self.nbJob, self.nbMach, self.PTM = self.read_input(instance)
        self.eval = self.make_eval()

    def read_input(self, instname):
        if instname[0] == 't':
            filename = os.path.join("./taillard", instname + ".dat")
        elif instname[0] == 'V':
            filename = os.path.join("./vrf", instname + "_Gap.txt")
        else:
            raise ValueError("Unsupported instance type")

        with open(filename, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('EOF')]
            nbJob, nbMach = map(int, lines[0].split())
            PTM = []

            if instname[0] == 't':
                for line in lines[1:nbMach + 1]:
                    PTM.append(list(map(int, line.split())))
            elif instname[0] == 'V':
                PTM = [[0] * nbJob for _ in range(nbMach)]
                for j, line in enumerate(lines[1:nbJob + 1]):
                    parts = list(map(int, line.split()))
                    for m in range(nbMach):
                        PTM[m][j] = parts[2 * m + 1]
            return nbJob, nbMach, PTM

    def make_eval(self):
        def evaluate(perm):
            tmp = [0] * self.nbMach
            for job in perm:
                tmp[0] += self.PTM[0][job]
                for m in range(1, self.nbMach):
                    tmp[m] = max(tmp[m], tmp[m - 1]) + self.PTM[m][job]
            return tmp[-1]

        return evaluate


# -------------------------------
# CP-SAT 求解（优化版）
# -------------------------------
def solve_flowshop(instance_name):
    # 读取实例数据
    fsp_instance = fsp(instance_name)
    processing_times = [[fsp_instance.PTM[m][j] for m in range(fsp_instance.nbMach)]
                        for j in range(fsp_instance.nbJob)]
    nbJob, nbMach = fsp_instance.nbJob, fsp_instance.nbMach
    horizon = sum(sum(job) for job in processing_times)

    # 创建模型
    model = cp_model.CpModel()

    # 定义变量（优化变量存储结构）
    start_vars = {}
    for j in range(nbJob):
        for m in range(nbMach):
            start_vars[(j, m)] = model.NewIntVar(0, horizon, f's_{j}_{m}')

    # 添加NEH初始解提示
    neh_perm = neh_algorithm(processing_times)
    start_times = compute_start_times(neh_perm, processing_times)
    for j in range(nbJob):
        for m in range(nbMach):
            model.AddHint(start_vars[(j, m)], start_times[j][m])

    # 作业内部顺序约束
    for j in range(nbJob):
        for m in range(nbMach - 1):
            model.Add(start_vars[(j, m + 1)] >= start_vars[(j, m)] + processing_times[j][m])

    # 全局顺序约束（优化版）
    for m in range(nbMach):
        intervals = []
        for j in range(nbJob):
            interval = model.NewIntervalVar(
                start_vars[(j, m)],
                processing_times[j][m],
                start_vars[(j, m)] + processing_times[j][m],
                f'interval_{j}_{m}'
            )
            intervals.append(interval)
        model.AddNoOverlap(intervals)  # 同一机器不重叠

    # 强制所有机器顺序一致
    for m in range(1, nbMach):
        for j1 in range(nbJob):
            for j2 in range(j1 + 1, nbJob):
                b = model.NewBoolVar(f'order_{j1}_{j2}')
                model.Add(start_vars[(j1, 0)] < start_vars[(j2, 0)]).OnlyEnforceIf(b)
                model.Add(start_vars[(j1, 0)] >= start_vars[(j2, 0)]).OnlyEnforceIf(b.Not())
                # 应用顺序到其他机器
                model.Add(start_vars[(j1, m)] < start_vars[(j2, m)]).OnlyEnforceIf(b)
                model.Add(start_vars[(j1, m)] >= start_vars[(j2, m)]).OnlyEnforceIf(b.Not())

    # 目标函数
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [start_vars[(j, nbMach - 1)] + processing_times[j][nbMach - 1]
                                    for j in range(nbJob)])
    model.Minimize(makespan)

    # 配置求解器参数
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8  # 使用8线程
    solver.parameters.max_time_in_seconds = 300  # 5分钟超时
    solver.parameters.log_search_progress = True  # 显示日志
    solver.parameters.linearization_level = 1  # 平衡模型大小与速度

    # 求解
    status = solver.Solve(model)

    # 结果处理
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"Makespan: {solver.Value(makespan)}")
        perm = sorted(range(nbJob), key=lambda j: solver.Value(start_vars[(j, 0)]))
        print("Validated Makespan:", compute_makespan(perm, processing_times))
        return perm
    else:
        print("No solution found.")
        return None


# -------------------------------
# 主程序
# -------------------------------
if __name__ == '__main__':
    instance_name = 'ta001'  # 替换为实际实例名
    solve_flowshop(instance_name)
