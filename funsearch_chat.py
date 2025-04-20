import os
import time
import re
import textwrap
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from func_timeout import func_timeout, FunctionTimedOut

# 配置参数
TAILLARD_DIR = 'TestSolutions/taillard'
MAX_ITERATIONS = 10
POPULATION_SIZE = 3
MAX_RETRIES = 5
BASE_DELAY = 1.5
# 强制输出格式的正则
SCHEDULER_REGEX = r'^def scheduler\(instance\):[\s\S]*?return schedule$'


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


def calculate_makespan(schedule: List[Any], instance: Dict) -> int:
    """
    计算调度方案的 makespan。
    验证每个作业的工序严格按顺序执行，并返回总完工时间。
    """
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    processing_times = instance['processing_times']
    machine_orders = instance['machine_orders']

    machine_times = [0] * num_machines
    job_finish = [0] * num_jobs
    next_op = [0] * num_jobs

    for idx, step in enumerate(schedule):
        if not (hasattr(step, '__getitem__') and len(step) >= 2):
            print(f"索引 {idx} 处的调度步骤格式错误: {step}")
            return float('inf')
        job, op = step[0], step[1]
        if not (0 <= job < num_jobs) or not (0 <= op < num_machines):
            print(f"步骤 {idx} 非法作业ID或工序号: {step}")
            return float('inf')
        if op != next_op[job]:
            print(f"步骤 {idx} 工序顺序错误: 作业 {job} 预期 {next_op[job]}，实际 {op}")
            return float('inf')
        machine = machine_orders[job][op]
        proc_time = processing_times[job][op]
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
    清理代码、执行并验证调度方案，返回 makespan 或错误信息。
    """
    # 去除代码块标记行，保留内容
    lines = program_code.splitlines()
    clean_lines = [l for l in lines if not l.strip().startswith('```')]
    clean_code = '\n'.join(clean_lines)

    # 提取完整函数定义
    func_match = re.search(r'(def scheduler\s*\([^)]*\):[\s\S]*?\breturn\s+schedule)', clean_code)
    if not func_match:
        return float('inf'), "未找到完整的 scheduler 定义或 return schedule"
    func_def = func_match.group(1)

    # 确保正确缩进
    body_match = re.search(r'def scheduler\s*\([^)]*\):(.*)', func_def, re.S)
    if not body_match:
        return float('inf'), "无法解析函数体结构"
    body = textwrap.dedent(body_match.group(1))
    indented = ''.join('    ' + line + '\n' for line in body.splitlines())
    new_code = f"def scheduler(data):\n{indented}    return schedule\n"

    restricted_globals = {
        '__builtins__': {'range': range, 'len': len, 'int': int, 'list': list},
        'instance': instance['processing_times'],
        '__name__': '__main__'
    }
    try:
        exec(new_code, restricted_globals)
    except Exception as e:
        return float('inf'), f"编译错误: {e}"

    def safe_exec():
        scheduler = restricted_globals['scheduler']
        schedule = scheduler(restricted_globals['instance'])
        if not isinstance(schedule, list):
            raise TypeError("调度结果必须为列表类型")
        return schedule

    try:
        schedule = func_timeout(5, safe_exec)
    except FunctionTimedOut:
        return float('inf'), '代码执行超时'
    except Exception as e:
        return float('inf'), f"执行错误: {e}"

    # 验证与计算 makespan
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            makespan = calculate_makespan(schedule, instance)
    except Exception as e:
        debug = buf.getvalue().strip()
        reason = f"验证时异常: {e}"
        if debug:
            reason += f" | 输出: {debug}"
        return float('inf'), reason
    debug_output = buf.getvalue().strip()
    if makespan == float('inf'):
        reason = debug_output or '未知验证错误'
        return makespan, reason
    return makespan, None


class DashScopeLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "qwen-max-latest"
        self.temperature = 0.7

    def generate(self, prompt: str) -> str:
        # 在提示中加入正则要求
        prompt += f"\n\n# 请仅输出满足以下正则的函数定义，不要包含注释或额外的代码： {SCHEDULER_REGEX}"
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "你是调度算法优化助手，基于给定错误反馈和代码，改进 scheduler 函数。"
                        )},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                    timeout=30
                )
                code = response.choices[0].message.content.strip()
                # 验证正则
                if not re.match(SCHEDULER_REGEX, code, re.MULTILINE):
                    print("生成代码未匹配正则格式，跳过此输出")
                    return None
                return code
            except Exception as e:
                delay = BASE_DELAY ** attempt
                print(f"API请求失败: {e}，{delay:.1f}s后重试")
                time.sleep(delay)
        return None


class FunSearch:
    def __init__(self, instance: Dict):
        self.instance = instance
        self.llm = DashScopeLLM()
        self.programs_db: List[Tuple[str, float]] = []
        self.error_history: List[str] = []
        initial_program = self._create_initial_program()
        score, err = evaluate_program(initial_program, instance)
        if err:
            print(f"初始程序评估错误: {err}")
            self.error_history.append(err)
        self.programs_db.append((initial_program, score))

    def _create_initial_program(self) -> str:
        return f"""
def scheduler(instance):
    num_jobs = {self.instance['num_jobs']}
    num_machines = {self.instance['num_machines']}
    schedule = []
    for j in range(num_jobs):
        for op in range(num_machines):
            schedule.append((j, op))
    return schedule
"""

    def _build_prompt(self) -> str:
        error_str = ""
        if self.error_history:
            error_str = "\n\n# 之前出现的错误信息\n" + "\n".join(self.error_history)
        current_best = min(score for _, score in self.programs_db)
        instance_note = (
            "# 注意: 传入 instance 为二维列表，instance[j][op] 返回加工时间。"
        )
        return f"""请基于以下代码和错误反馈对调度算法做微调：
{instance_note}
# 当前代码：
{self.programs_db[0][0]}

# 当前评估结果: makespan = {current_best}
# 近期错误反馈：
{error_str}

请优化 scheduler 函数以降低 makespan，仅输出完整函数定义，不要包含额外说明。"""

    def _update_population(self, code: str, score: float):
        if code in (c for c, _ in self.programs_db):
            print("重复程序，跳过")
            return
        self.programs_db.append((code, score))
        self.programs_db.sort(key=lambda x: x[1])
        self.programs_db = self.programs_db[:POPULATION_SIZE]

    def evolve(self):
        for i in range(MAX_ITERATIONS):
            print(f"=== 迭代 {i+1}/{MAX_ITERATIONS} ===")
            best = min(s for _, s in self.programs_db)
            print(f"当前最佳 makespan: {best}")
            prompt = self._build_prompt()
            new_code = self.llm.generate(prompt)
            if not new_code or 'def scheduler' not in new_code:
                print("无效的生成内容，跳过")
                continue
            score, err = evaluate_program(new_code, self.instance)
            if err:
                print(f"执行错误: {err}")
                self.error_history.append(err)
            else:
                print(f"评估结果: {score}")
            self._update_population(new_code, score)


def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return
    path = os.path.join(TAILLARD_DIR, 'ta092.dat')
    inst = load_taillard_instance(path)
    if not inst:
        return
    fs = FunSearch(inst)
    fs.evolve()
    best_code, best_score = min(fs.programs_db, key=lambda x: x[1])
    print(f"\n最佳 makespan: {best_score}")
    print("最佳程序代码:")
    print(best_code)

if __name__ == "__main__":
    main()


