import os
from ortools.sat.python import cp_model


# fsp 类，负责读取实例文件以及生成评价函数
class fsp:
    def __init__(self, instance):
        self.instname = instance
        self.nbJob, self.nbMach = self.read_nbJob_nbMachine(self.instname)
        # 这里我们只需要读取数据，因此不必使用 make_eval

    def read_nbJob_nbMachine(self, instname):
        if instname[0] == 't':
            filename = os.path.join("./taillard", instname + ".dat")
        elif instname[0] == 'V':
            filename = os.path.join("./vrf", instname + "_Gap.txt")
        else:
            raise ValueError("不支持的实例类型")

        with open(filename, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            nbJob = int(lines[0].split()[0])
            nbMach = int(lines[0].split()[1])
            return nbJob, nbMach

    # 读取加工时间矩阵，返回格式为 (PTM, nbJob, nbMach)
    def read_input(self, instname):
        if instname[0] == 't':
            filename = os.path.join("./taillard", instname + ".dat")
        elif instname[0] == 'V':
            filename = os.path.join("./vrf", instname + "_Gap.txt")
        else:
            raise ValueError("不支持的实例类型")

        with open(filename, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            nbJob = int(lines[0].split()[0])
            nbMach = int(lines[0].split()[1])
            # 初始化 PTM：存储每台机器的加工时间，尺寸 nbMach x nbJob
            PTM = [[0] * nbJob for _ in range(nbMach)]

            if instname[0] == 't':
                # 每行对应一台机器的所有作业的加工时间
                machine_index = 0
                for l in lines[1:]:
                    if not l.strip():
                        continue
                    if l.startswith('EOF'):
                        break
                    # 假设文件中机器顺序和行数一致
                    PTM[machine_index] = [int(x) for x in l.split()]
                    machine_index += 1
            elif instname[0] == 'V':
                # 每行对应一个作业，在每行中每个加工时间前有一个无关数值（例如：作业号或其他标识）
                job_index = 0
                for l in lines[1:]:
                    if not l.strip():
                        continue
                    if l.startswith('EOF'):
                        break
                    # 对于 vrf 实例，假设每个作业一行，其中每个机器加工时间前面有一个标识，故取偶数位置的数据
                    parts = l.split()
                    for i in range(nbMach):
                        # 数据位置为 2*i+1
                        PTM[i][job_index] = int(parts[2 * i + 1])
                    job_index += 1
            return PTM, nbJob, nbMach


# 根据 fsp 类读取的 PTM 数据，需要将数据转置为按作业组织的格式（nbJob×nbMach）
def transpose_ptm(PTM, nbJob, nbMach):
    # 原始 PTM: nbMach行，每行nbJob个加工时间；转置后为 nbJob 行，每行 nbMach 个加工时间
    processing_times = [[PTM[m][j] for m in range(nbMach)] for j in range(nbJob)]
    return processing_times


# 建立 CP-SAT 模型求解 Flow Shop Scheduling 问题
def solve_flowshop(instance_name):
    # 读取实例
    instance = fsp(instance_name)
    PTM, nbJob, nbMach = instance.read_input(instance_name)
    processing_times = transpose_ptm(PTM, nbJob, nbMach)

    # 计算一个上界（所有作业所有机器加工时间之和）
    horizon = sum(sum(job) for job in processing_times)

    model = cp_model.CpModel()

    # 建立变量字典
    interval_vars = {}
    start_vars = {}
    end_vars = {}

    # 为每个作业的每个机器建立变量
    for j in range(nbJob):
        for m in range(nbMach):
            duration = processing_times[j][m]
            start = model.NewIntVar(0, horizon, f'start_{j}_{m}')
            end = model.NewIntVar(0, horizon, f'end_{j}_{m}')
            interval = model.NewIntervalVar(start, duration, end, f'interval_{j}_{m}')
            start_vars[(j, m)] = start
            end_vars[(j, m)] = end
            interval_vars[(j, m)] = interval

    # 同一作业内，后一道工序必须在前一道工序完成之后开始
    for j in range(nbJob):
        for m in range(nbMach - 1):
            model.Add(start_vars[(j, m + 1)] >= end_vars[(j, m)])

    # 每台机器上，不同作业任务之间不允许重叠
    for m in range(nbMach):
        intervals = [interval_vars[(j, m)] for j in range(nbJob)]
        model.AddNoOverlap(intervals)

    # 目标：最小化所有作业在最后一台机器上的完工时间（makespan）
    makespan = model.NewIntVar(0, horizon, 'makespan')
    last_machine_ends = [end_vars[(j, nbMach - 1)] for j in range(nbJob)]
    model.AddMaxEquality(makespan, last_machine_ends)
    model.Minimize(makespan)

    # 求解模型
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("找到可行解：")
        print("最小完工时间 (makespan) =", solver.Value(makespan))
        for j in range(nbJob):
            print(f"作业 {j}:")
            for m in range(nbMach):
                start_val = solver.Value(start_vars[(j, m)])
                end_val = solver.Value(end_vars[(j, m)])
                print(f"  机器 {m}: 开始 = {start_val}, 结束 = {end_val} (加工时间: {processing_times[j][m]})")
    else:
        print("未能找到可行解。")


# 示例调用
if __name__ == '__main__':
    # 请根据实际情况填写实例名称，如 't1' 或 'V1'，要求文件存放在指定目录下
    instance_name = 'ta001'  # 示例：taillard 格式实例
    solve_flowshop(instance_name)
