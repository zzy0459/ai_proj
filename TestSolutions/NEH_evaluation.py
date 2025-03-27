import openai
import os
import time
from collections import defaultdict

# ========================
# 配置参数部分
# ========================
OPENAI_API_KEY = 'sk-xxx'  # 替换为您的实际密钥
DATA_PATH = './taillard'  # 数据集路径
MAX_ITERATIONS = 10  # 最大迭代次数
POPULATION_SIZE = 5  # 种群规模
TIMEOUT = 30  # API超时时间（秒）


# ========================
# 数据加载与处理模块
# ========================
class FlowShopProblem:
    def __init__(self, instance_name):
        self.instance = instance_name
        self.num_jobs, self.num_machines, self.processing_matrix = self.load_data()
        self.processing_times = [
            [self.processing_matrix[m][j] for m in range(self.num_machines)]
            for j in range(self.num_jobs)
        ]

    def load_data(self):
        """加载Taillard格式数据"""
        filepath = os.path.join(DATA_PATH, f"{self.instance}.dat")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"实例文件 {filepath} 未找到")

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

            # 解析基础信息
            num_jobs, num_machines = map(int, lines[0].split())

            # 初始化处理时间矩阵
            pt_matrix = []
            current_line = 1

            # 读取机器数据
            for _ in range(num_machines):
                while current_line < len(lines):
                    if lines[current_line].isdigit():
                        pt_matrix.append(list(map(int, lines[current_line].split())))
                        current_line += 1
                        break
                    current_line += 1

        return num_jobs, num_machines, pt_matrix


# ========================
# NEH算法模块
# ========================
def neh_algorithm(processing_times):
    """NEH算法核心实现"""
    num_jobs = len(processing_times)
    total_times = [sum(job) for job in processing_times]
    job_order = sorted(range(num_jobs), key=lambda x: -total_times[x])

    current_seq = [job_order[0]]
    for idx in range(1, num_jobs):
        candidate = job_order[idx]
        best_pos, best_ms = 0, float('inf')

        for pos in range(len(current_seq) + 1):
            temp_seq = current_seq[:pos] + [candidate] + current_seq[pos:]
            ms = calculate_makespan(temp_seq, processing_times)
            if ms < best_ms:
                best_ms, best_pos = ms, pos

        current_seq = current_seq[:best_pos] + [candidate] + current_seq[best_pos:]

    return current_seq


def calculate_makespan(sequence, processing_times):
    """计算完工时间"""
    num_machines = len(processing_times[0])
    timeline = [0] * num_machines

    for job in sequence:
        timeline[0] += processing_times[job][0]
        for m in range(1, num_machines):
            timeline[m] = max(timeline[m], timeline[m - 1]) + processing_times[job][m]

    return timeline[-1]


# ========================
# 智能优化模块
# ========================
class ChatGPTLLM:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=TIMEOUT
        )

    def generate(self, prompt, max_retries=3):
        """带重试机制的生成函数"""
        for _ in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "您需要生成优化调度算法的Python代码"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )
                return response.choices[0].message.content
            except openai.APITimeoutError:
                print("API请求超时，正在重试...")
                time.sleep(5)
            except Exception as e:
                print(f"API错误: {e}")
                return None
        return None


class FunSearch:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.llm = ChatGPTLLM()
        self.population = self.initialize_population()

    def initialize_population(self):
        """使用NEH算法初始化种群"""
        neh_sequence = neh_algorithm(self.problem.processing_times)
        initial_code = self.sequence_to_code(neh_sequence)
        return [(initial_code, self.evaluate(initial_code))]

    def sequence_to_code(self, sequence):
        """将调度序列转换为可执行代码"""
        code = f"""def scheduler(instance):
    return {sequence}
"""
        return code

    def evaluate(self, program_code):
        """增强型评估函数"""
        try:
            exec_globals = {}
            exec(program_code, exec_globals)

            if 'scheduler' not in exec_globals:
                return float('inf')

            scheduler = exec_globals['scheduler']
            schedule = scheduler(None)

            # 增强验证逻辑
            if not isinstance(schedule, list):
                return float('inf')

            job_ids = set(range(self.problem.num_jobs))
            for job, op in schedule:
                if not (0 <= job < self.problem.num_jobs):
                    return float('inf')
                if not (0 <= op < self.problem.num_machines):
                    return float('inf')

            return calculate_makespan(schedule, self.problem.processing_times)
        except:
            return float('inf')

    def evolve(self):
        """进化主循环"""
        for iter in range(MAX_ITERATIONS):
            print(f"\n=== 迭代 {iter + 1}/{MAX_ITERATIONS} ===")

            # 生成提示
            prompt = self.build_prompt()

            # 生成新代码
            new_code = self.llm.generate(prompt)
            if not new_code:
                continue

            # 评估新代码
            score = self.evaluate(new_code)
            print(f"新方案得分: {score}")

            # 更新种群
            self.population.append((new_code, score))
            self.population = sorted(self.population, key=lambda x: x[1])[:POPULATION_SIZE]

    def build_prompt(self):
        """构建优化提示"""
        base_prompt = f"""需要优化流水车间调度算法，当前问题参数：
- 作业数量：{self.problem.num_jobs}
- 机器数量：{self.problem.num_machines}
- 当前最佳方案：{self.population[0][1]}

请根据以下优秀方案生成改进代码：
"""
        for i, (code, score) in enumerate(self.population[:3]):
            base_prompt += f"\n方案{i + 1} (得分: {score}):\n{code}\n"

        base_prompt += "\n请生成新的优化方案代码，确保：\n- 包含完整的scheduler函数\n- 合理利用问题特征\n- 包含必要的性能优化"
        return base_prompt

    def get_best_solution(self):
        return min(self.population, key=lambda x: x[1])


# ========================
# 主程序模块
# ========================
def main():
    # 初始化问题实例
    problem = FlowShopProblem("ta101")

    # 运行NEH算法基准
    neh_sequence = neh_algorithm(problem.processing_times)
    neh_ms = calculate_makespan(neh_sequence, problem.processing_times)
    print(f"\n基准NEH算法结果：")
    print(f"最优排列：{neh_sequence}")
    print(f"完工时间：{neh_ms}")

    # 运行FunSearch优化
    print("\n启动FunSearch优化...")
    optimizer = FunSearch(problem)
    optimizer.evolve()

    # 输出最终结果
    best_code, best_score = optimizer.get_best_solution()
    print("\n优化最终结果：")
    print(f"最佳完工时间：{best_score}")
    print(f"对比NEH改进：{neh_ms - best_score}")
    print("\n最佳方案代码：")
    print(best_code)


if __name__ == "__main__":
    main()