import os
import random
import time
import re
from typing import List, Tuple, Dict
from openai import OpenAI
from func_timeout import func_timeout, FunctionTimedOut

# 配置参数
TAILLARD_DIR = 'TestSolutions/taillard'
MAX_ITERATIONS = 10
POPULATION_SIZE = 3
MAX_RETRIES = 5
BASE_DELAY = 1.5

def load_taillard_instance(file_path: str) -> Dict:
    """
    解析流水车间(Flow Shop)格式的Taillard数据。
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            num_jobs, num_machines = map(int, lines[0].split())
            processing_times = []
            for m in range(num_machines):
                data = list(map(int, lines[m + 1].split()))
                if len(data) != num_jobs:
                    print(f"机器 {m + 1} 数据不完整，期望 {num_jobs} 个作业，实际 {len(data)} 个")
                    return None
                processing_times.append(data)
            job_processing = [
                [processing_times[m][j] for m in range(num_machines)]
                for j in range(num_jobs)
            ]
            machine_orders = [[m for m in range(num_machines)] for _ in range(num_jobs)]
            return {
                'num_jobs': num_jobs,
                'num_machines': num_machines,
                'processing_times': job_processing,
                'machine_orders': machine_orders
            }
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def calculate_makespan(schedule: List[Tuple[int, int]], instance: Dict) -> int:
    """
    计算调度方案的 makespan。
    使用 job_finish 记录作业完成时间，machine_times 记录机器完成时间，
    检查每个作业的工序是否严格按照顺序执行。
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    processing_times = instance['processing_times']
    machine_orders = instance['machine_orders']

    machine_times = [0] * num_machines
    job_finish = [0] * num_jobs
    next_op = [0] * num_jobs

    for step in schedule:
        if not (isinstance(step, tuple) and len(step) == 2):
            print(f"无效调度步骤: {step}")
            return float('inf')
        job, op = step
        if not (0 <= job < num_jobs) or not (0 <= op < num_machines):
            print(f"非法作业ID或工序号: {step}")
            return float('inf')
        if op != next_op[job]:
            print(f"工序顺序错误: 作业 {job} 预期 {next_op[job]}，实际 {op}")
            return float('inf')
        try:
            machine = machine_orders[job][op]
            proc_time = processing_times[job][op]
        except IndexError as e:
            print(f"数据访问错误: {e}")
            return float('inf')
        start_time = max(job_finish[job], machine_times[machine])
        finish_time = start_time + proc_time
        machine_times[machine] = finish_time
        job_finish[job] = finish_time
        next_op[job] += 1

    if any(next_op[j] != num_machines for j in range(num_jobs)):
        print("部分作业未完成所有工序")
        return float('inf')
    return max(machine_times)

def evaluate_program(program_code: str, instance: Dict) -> Tuple[int, str]:
    """
    对生成的代码进行评估，自动删除所有 Markdown 格式标记（```等），
    并执行代码检查调度方案是否满足工序顺序要求。
    """
    try:
        start_idx = program_code.find("def scheduler(instance):")
        if start_idx == -1:
            return float('inf'), "未找到 scheduler 函数定义"
        start_idx += len("def scheduler(instance):")

        end_idx = program_code.find("return schedule", start_idx)
        if end_idx == -1:
            return float('inf'), "未找到 return schedule 语句"

        inner_code = program_code[start_idx:end_idx].strip()
        new_program_code = f"""
def scheduler(instance):
    {inner_code}
    return schedule
"""
        # 设置执行环境，允许 min 和 max
        restricted_globals = {
            "__builtins__": {
                "range": range, "int": int, "list": list, "len": len,
                "min": min, "max": max
            },
            "instance": instance,
            "__name__": "__main__"
        }
        exec(new_program_code, restricted_globals)
        if 'scheduler' not in restricted_globals:
            print("未找到 scheduler 函数")
            return float('inf'), "未找到 scheduler 函数"

        def safe_exec():
            scheduler = restricted_globals['scheduler']
            schedule = scheduler(instance)
            if not isinstance(schedule, list):
                raise TypeError("调度结果必须为列表类型")
            op_progress = [0] * instance['num_jobs']
            for step in schedule:
                if not (isinstance(step, tuple) and len(step) == 2):
                    raise ValueError(f"无效步骤格式: {step}")
                job, op = step
                if op != op_progress[job]:
                    raise ValueError(f"工序顺序错误: 作业 {job} 预期 {op_progress[job]}，实际 {op}")
                op_progress[job] += 1
            if any(op_progress[j] != instance['num_machines'] for j in range(instance['num_jobs'])):
                raise ValueError("部分作业未完成所有工序")
            return schedule

        try:
            schedule = func_timeout(5, safe_exec)
        except FunctionTimedOut:
            return float('inf'), "代码执行超时（超过5秒）"
        except Exception as e:
            error_message = f"执行错误: {str(e)}"
            if "is not defined" in str(e):
                error_message += " - 请检查是否使用了未允许的函数，仅限 range, len, list, int, min, max"
            elif "index out of range" in str(e):
                error_message += " - 请检查对 instance['processing_times'] 的访问是否越界"
            return float('inf'), error_message

        makespan = calculate_makespan(schedule, instance)
        return makespan, None

    except Exception as e:
        error_message = f"评估异常: {e}"
        print(error_message)
        return float('inf'), error_message

class DashScopeLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "qwen-coder-plus-latest"
        self.temperature = 0.5  # 调整为0.5
        self.max_tokens = 1500  # 增加到1500

    def generate(self, prompt: str) -> str:
        """
        调用大模型生成代码，支持指数退避重试机制
        """
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "你是一个专注于调度算法优化的AI助手。请基于给定代码和错误反馈做微调，只修改必要部分，使得调度方案中每个作业的工序严格按顺序执行。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30
                )
                latency = time.time() - start_time
                print(f"API响应时间: {latency:.2f}秒")
                if (not response.choices or not response.choices[0].message or not response.choices[0].message.content):
                    raise ValueError("无效的API响应格式")
                print("API回复内容：")
                print(response.choices[0].message.content)
                return response.choices[0].message.content
            except Exception as e:
                delay = BASE_DELAY ** attempt
                print(f"API请求失败（第{attempt + 1}次重试）: {e} | {delay:.1f}秒后重试")
                time.sleep(delay)
        print(f"连续{MAX_RETRIES}次请求失败")
        return None

class FunSearch:
    def __init__(self, instance: Dict):
        self.instance = instance
        self.llm = DashScopeLLM()
        self.programs_db = []
        self.error_history = []
        # 初始代码作为起点
        initial_program = self._create_initial_program()
        initial_score, error_message = evaluate_program(initial_program, instance)
        if error_message:
            self.error_history.append(error_message)
        self.programs_db.append((initial_program, initial_score))

    def _create_initial_program(self) -> str:
        return f"""
def scheduler(instance):
    num_jobs = {self.instance['num_jobs']}
    num_machines = {self.instance['num_machines']}
    # 初始调度：每个作业的工序按顺序加工
    schedule = []
    for j in range(num_jobs):
        for op in range(num_machines):
            schedule.append((j, op))
    return schedule
"""

    def evolve(self):
        for iter in range(MAX_ITERATIONS):
            print(f"\n=== 迭代 {iter + 1}/{MAX_ITERATIONS} ===")
            best_score = min(score for _, score in self.programs_db)
            print(f"当前最佳分数: {best_score if best_score != float('inf') else '无解'}")
            try:
                prompt = self._build_prompt()
                print("提示长度:", len(prompt), "字符")
                new_code = self.llm.generate(prompt)
                if not new_code:
                    continue
                if "def scheduler" not in new_code:
                    print("生成代码缺少函数定义")
                    continue
                score, error_message = evaluate_program(new_code, self.instance)
                if error_message:
                    self.error_history.append(error_message)
                print(f"评估结果: {score if score != float('inf') else '无效解'}")
                self._update_population(new_code, score)
            except Exception as e:
                print(f"迭代异常: {e}")
                continue

    def _build_prompt(self) -> str:
        error_str = "\n".join(self.error_history) if self.error_history else "无"
        current_best = min(score for _, score in self.programs_db)
        return f"""你是一个专注于流水车间（Flow Shop）调度优化的AI助手。请基于以下信息对调度算法进行微调，目标是最小化makespan。

# 调度问题描述：
- 作业数量：{self.instance['num_jobs']} 个
- 机器数量：{self.instance['num_machines']} 台
- 每个作业有 {self.instance['num_machines']} 个工序，必须严格按照 0, 1, 2, ..., {self.instance['num_machines']-1} 的顺序在对应机器上执行。
- 加工时间存储在 instance['processing_times'] 中，其中 instance['processing_times'][j][m] 表示作业 j 在机器 m 上的工序加工时间。
- 调度方案是一个列表，元素为元组 (job, operation)，表示作业 job 的第 operation 个工序。

# 当前代码：
{self.programs_db[0][0]}

# 当前评估结果：
makespan = {current_best}

# 近期错误反馈：
{error_str}

# 要求：
1. 优化调度方案以降低makespan，考虑使用启发式方法（如NEH算法、Johnson's rule的扩展）或贪心策略。
2. 严格保证每个作业的工序按 0, 1, 2, ... 顺序执行。
3. 仅使用以下内置函数：range, len, list, int, min, max，避免使用其他未提供的函数。
4. 使用 instance['processing_times'] 获取加工时间，不要假设固定时间。
5. 函数必须以 def scheduler(instance): 开头，以 return schedule 结尾，仅输出函数定义，无额外代码或注释。

请生成优化后的调度代码。
"""

    def _update_population(self, new_code: str, score: float):
        existing_codes = [code for code, _ in self.programs_db]
        if new_code in existing_codes:
            print("发现重复程序，跳过")
            return
        self.programs_db.append((new_code, score))
        self.programs_db.sort(key=lambda x: x[1])
        self.programs_db = self.programs_db[:POPULATION_SIZE]

def main():
    print("当前API密钥:", os.getenv("DASHSCOPE_API_KEY"))
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    instance_path = os.path.join(TAILLARD_DIR, 'ta092.dat')
    instance = load_taillard_instance(instance_path)
    if not instance:
        return
    funsearch = FunSearch(instance)
    funsearch.evolve()
    best_program, best_score = min(funsearch.programs_db, key=lambda x: x[1])
    print(f"\n最佳 makespan: {best_score}")
    print("最佳程序代码:")
    print(best_program)

if __name__ == "__main__":
    main()