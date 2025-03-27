import os
from ortools.sat.python import cp_model


# -------------------------------
# fsp 类，用于读取实例数据和评价调度方案
# -------------------------------
class fsp:
    def __init__(self, instance):
        self.instname = instance
        self.nbJob, self.nbMach = self.read_nbJob_nbMachine(self.instname)
        self.eval = self.make_eval(self.instname)

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

    # 读取实例中的加工时间矩阵
    # 对于 taillard 格式，每一行存储一台机器的所有作业加工时间；
    # 对于 vrf 格式，每一行存储一个作业的各台机器加工时间（加工时间在偶数位置）
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
            # 初始化 PTM：按机器组织，尺寸 nbMach x nbJob
            PTM = [[0] * nbJob for _ in range(nbMach)]
            if instname[0] == 't':
                # 每行代表一台机器的加工时间
                machine_index = 0
                for l in lines[1:]:
                    if not l.strip():
                        continue
                    if l.startswith('EOF'):
                        break
                    PTM[machine_index] = [int(x) for x in l.split()]
                    machine_index += 1
            elif instname[0] == 'V':
                # 每行代表一个作业，取偶数位置的加工时间数据
                job_index = 0
                for l in lines[1:]:
                    if not l.strip():
                        continue
                    if l.startswith('EOF'):
                        break
                    parts = l.split()
                    for i in range(nbMach):
                        PTM[i][job_index] = int(parts[2 * i + 1])
                    job_index += 1
            return PTM, nbJob, nbMach

    # 生成评价函数，输入一个作业排列 perm，返回 makespan
    def make_eval(self, instance):
        PTM, nJob, nMach = self.read_input(instance)

        def evaluate(perm) -> int:
            assert len(perm) == nJob
            tmp = [0] * nMach
            for i in range(nJob):
                jb = perm[i]
                tmp[0] += PTM[0][jb]
                for j in range(1, nMach):
                    tmp[j] = max(tmp[j], tmp[j - 1]) + PTM[j][jb]
            return tmp[nMach - 1]

        return evaluate


# -------------------------------
# CP-SAT 模型求解（调整后为排列流水车间）
# -------------------------------
def solve_flowshop(instance_name):
    # 读取实例数据
    fsp_instance = fsp(instance_name)
    PTM, nbJob, nbMach = fsp_instance.read_input(instance_name)
    # 将加工时间矩阵转置为：nbJob 行，每行 nbMach 个加工时间
    processing_times = [[PTM[m][j] for m in range(nbMach)] for j in range(nbJob)]

    # 上界：所有作业各机器加工时间之和
    horizon = sum(sum(job) for job in processing_times)

    model = cp_model.CpModel()

    # 定义作业在各机器上的开始、结束、区间变量
    start_vars = {}
    end_vars = {}
    interval_vars = {}
    for j in range(nbJob):
        for m in range(nbMach):
            duration = processing_times[j][m]
            start = model.NewIntVar(0, horizon, f'start_{j}_{m}')
            end = model.NewIntVar(0, horizon, f'end_{j}_{m}')
            interval = model.NewIntervalVar(start, duration, end, f'interval_{j}_{m}')
            start_vars[(j, m)] = start
            end_vars[(j, m)] = end
            interval_vars[(j, m)] = interval

    # 作业内部顺序：同一作业在相邻机器上依次加工
    for j in range(nbJob):
        for m in range(nbMach - 1):
            model.Add(start_vars[(j, m + 1)] >= end_vars[(j, m)])

    # —— 强制排列流水车间：各机器上必须有相同的作业顺序 ——
    # 为每对作业 (i, j) (i < j) 引入二元变量 order[(i,j)]，
    # 若 order[(i,j)] == True 则表示作业 i 在所有机器上均排在作业 j 之前。
    order_vars = {}
    for i in range(nbJob):
        for j in range(i + 1, nbJob):
            b = model.NewBoolVar(f"order_{i}_{j}")
            order_vars[(i, j)] = b
            # 对于每台机器 m，都加上相应的顺序约束
            for m in range(nbMach):
                # 若 b 为真，则 i 在 m 上必须排在 j 之前
                model.Add(start_vars[(i, m)] + processing_times[i][m] <= start_vars[(j, m)]).OnlyEnforceIf(b)
                # 若 b 为假，则 j 在 m 上必须排在 i 之前
                model.Add(start_vars[(j, m)] + processing_times[j][m] <= start_vars[(i, m)]).OnlyEnforceIf(b.Not())

    # 目标：最小化所有作业在最后一台机器上的完工时间（makespan）
    makespan = model.NewIntVar(0, horizon, 'makespan')
    last_machine_ends = [end_vars[(j, nbMach - 1)] for j in range(nbJob)]
    model.AddMaxEquality(makespan, last_machine_ends)
    model.Minimize(makespan)

    # 求解模型
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        cp_sat_makespan = solver.Value(makespan)
        print("找到可行解：")
        print("最小完工时间 (makespan) =", cp_sat_makespan)
        for j in range(nbJob):
            print(f"作业 {j}:")
            for m in range(nbMach):
                start_val = solver.Value(start_vars[(j, m)])
                end_val = solver.Value(end_vars[(j, m)])
                print(f"  机器 {m}: 开始 = {start_val}, 结束 = {end_val} (加工时间: {processing_times[j][m]})")

        # 提取作业顺序：由于所有机器顺序一致，可以仅基于机器0排序
        perm = sorted(range(nbJob), key=lambda j: solver.Value(start_vars[(j, 0)]))
        print("\n提取的作业顺序（基于机器0的开始时间）:", perm)

        # 用 fsp 中的评价函数验证 makespan
        eval_fn = fsp_instance.make_eval(instance_name)
        eval_makespan = eval_fn(perm)
        print("Evaluate 函数计算的 makespan =", eval_makespan)

        if eval_makespan == cp_sat_makespan:
            print("验证成功：两者一致！")
        else:
            print("验证失败：两者不一致，请检查调度顺序或模型约束。")
    else:
        print("未能找到可行解。")


# -------------------------------
# 主函数
# -------------------------------
if __name__ == '__main__':
    # 假设实际文件名为 "ta001.dat"，实例名称设置为 "ta001"
    instance_name = 'ta001'
    solve_flowshop(instance_name)