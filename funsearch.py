import os
import random
import time
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
    """正确解析流水车间(Flow Shop)格式的Taillard数据"""
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

            num_jobs, num_machines = map(int, lines[0].split())
            processing_times = []
            for m in range(num_machines):
                data = list(map(int, lines[m + 1].split()))
                if len(data) != num_jobs:
                    print(f"机器{m + 1}数据不完整，期望{num_jobs}个作业，实际{len(data)}个")
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
        print(f"数据加载失败: {str(e)}")
        return None


def calculate_makespan(schedule: List[Tuple[int, int]], instance: Dict) -> int:
    """带安全检查的makespan计算"""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    processing_times = instance['processing_times']
    machine_orders = instance['machine_orders']

    machine_times = [0] * num_machines
    job_progress = [0] * num_jobs

    for step in schedule:
        if len(step) != 2:
            print(f"无效调度步骤: {step}")
            return float('inf')

        job, op = step

        if not (0 <= job < num_jobs):
            print(f"非法作业ID: {job}")
            return float('inf')
        if not (0 <= op < num_machines):
            print(f"非法工序号: {op}")
            return float('inf')

        if op != job_progress[job]:
            print(f"工序顺序错误: 作业{job}预期工序{job_progress[job]}，实际{op}")
            return float('inf')

        try:
            machine = machine_orders[job][op]
            proc_time = processing_times[job][op]
        except IndexError as e:
            print(f"数据访问错误: {str(e)}")
            return float('inf')

        start_time = max(machine_times[machine], job_progress[job])
        machine_times[machine] = start_time + proc_time
        job_progress[job] = start_time + proc_time

    return max(machine_times)


def evaluate_program(program_code: str, instance: Dict) -> float:
    """带超时机制和安全检查的评估函数"""
    try:
        restricted_globals = {
            "__builtins__": {"range": range, "int": int, "list": list},
            "instance": instance
        }

        exec(program_code, restricted_globals)

        if 'scheduler' not in restricted_globals:
            print("未找到scheduler函数")
            return float('inf')

        def safe_exec():
            scheduler = restricted_globals['scheduler']
            schedule = scheduler(instance)

            if not isinstance(schedule, list):
                raise TypeError("调度结果必须为列表类型")

            for step in schedule:
                if not (isinstance(step, tuple) and len(step) == 2):
                    raise ValueError(f"无效步骤格式: {step}")
                job, op = step
                if not (0 <= job < instance['num_jobs']):
                    raise ValueError(f"作业ID越界: {job}")
                if not (0 <= op < instance['num_machines']):
                    raise ValueError(f"工序号越界: {op}")
            return schedule

        try:
            schedule = func_timeout(5, safe_exec)
        except FunctionTimedOut:
            print("代码执行超时")
            return float('inf')
        except Exception as e:
            print(f"执行错误: {str(e)}")
            return float('inf')

        return calculate_makespan(schedule, instance)

    except Exception as e:
        print(f"评估异常: {str(e)}")
        return float('inf')


class DashScopeLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "deepseek-r1"
        self.temperature = 0.7

    def generate(self, prompt: str) -> str:
        """带指数退避的重试机制"""
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专注于生成安全、高效调度算法的AI助手"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                    timeout=30
                )

                latency = time.time() - start_time
                print(f"API响应时间: {latency:.2f}秒")

                if (not response.choices or
                        not response.choices[0].message or
                        not response.choices[0].message.content):
                    raise ValueError("无效的API响应格式")

                return response.choices[0].message.content

            except Exception as e:
                delay = BASE_DELAY ** attempt
                print(f"API请求失败（第{attempt + 1}次重试）: {str(e)} | {delay:.1f}秒后重试")
                time.sleep(delay)

        print(f"连续{MAX_RETRIES}次请求失败")
        return None


class FunSearch:
    def __init__(self, instance: Dict):
        self.instance = instance
        self.llm = DashScopeLLM()
        self.programs_db = []
        initial_program = self._create_initial_program()
        initial_score = evaluate_program(initial_program, instance)
        self.programs_db.append((initial_program, initial_score))

    def _create_initial_program(self) -> str:
        return f'''
def scheduler(instance):
    num_jobs = {self.instance['num_jobs']}
    num_machines = {self.instance['num_machines']}
    return [(j, o) for j in range(num_jobs) for o in range(num_machines)]
'''

    def evolve(self):
        for iter in range(MAX_ITERATIONS):
            print(f"\n=== 迭代 {iter + 1}/{MAX_ITERATIONS} ===")
            best_score = min([s for _, s in self.programs_db])
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

                score = evaluate_program(new_code, self.instance)
                print(f"评估结果: {score if score != float('inf') else '无效解'}")

                self._update_population(new_code, score)

            except Exception as e:
                print(f"迭代异常: {str(e)}")
                continue

    def _build_prompt(self) -> str:
        return f'''请生成符合以下要求的调度算法代码：
# 问题描述
- 处理{self.instance['num_jobs']}个作业和{self.instance['num_machines']}台机器
- 作业工序必须按0->1->2->...顺序执行


def scheduler(instance):
    # 算法实现
    return schedule  # 格式：[(作业ID, 工序号), ...]

# 当前最佳参考（makespan: {min([s for _, s in self.programs_db])}）
{self.programs_db[0][0]}'''

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
        print("请先设置DASHSCOPE_API_KEY环境变量")
        return

    instance_path = os.path.join(TAILLARD_DIR, 'ta001.dat')
    instance = load_taillard_instance(instance_path)
    if not instance:
        return

    funsearch = FunSearch(instance)
    funsearch.evolve()

    best_program, best_score = min(funsearch.programs_db, key=lambda x: x[1])
    print(f"\n最佳makespan: {best_score}")
    print("最佳程序代码:")
    print(best_program)


if __name__ == "__main__":
    main()