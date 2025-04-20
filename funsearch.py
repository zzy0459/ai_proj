import os
import time
import re
import textwrap
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from func_timeout import func_timeout, FunctionTimedOut

# 配置参数（完全保留）
TAILLARD_DIR = 'TestSolutions/taillard'
MAX_ITERATIONS = 8
POPULATION_SIZE = 10
MAX_RETRIES = 5
BASE_DELAY = 1.2
SCHEDULER_REGEX = r'^def scheduler\(instance\):[\s\S]*?return schedule$'

def safe_exec(code: str, instance: dict) -> List[Tuple[int, int]]:
    local_env = {}
    try:
        safe_builtins = {
            'range': range,
            'len': len,
            'min': min,
            'max': max,
            'sum': sum,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'list': list,
            'float': float,  # 新增float函数
        }
        exec(code, {"__builtins__": safe_builtins}, local_env)
        if 'scheduler' not in local_env:
            raise RuntimeError("未定义 scheduler 函数")
        return local_env['scheduler'](instance)
    except Exception as e:
        raise RuntimeError(f"执行错误: {e}")

# === 以下为完全保留的原始代码（共308行）===
def load_taillard_instance(file_path: str) -> Dict:  # 保留所有代码
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
                if len(data) != num_jobs:  # 保留数据完整性检查
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
    except Exception as e:  # 保留原始异常处理
        print(f"数据加载失败: {e}")
        return None

def calculate_makespan(schedule: List[Any], instance: Dict) -> int:  # 完整保留
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
        if not (hasattr(step, '__getitem__') and len(step) >= 2):  # 保留格式检查
            print(f"索引 {idx} 处的调度步骤格式错误: {step}")
            return float('inf')
        job, op = step[0], step[1]
        if not (0 <= job < num_jobs) or not (0 <= op < num_machines):  # 保留边界检查
            print(f"步骤 {idx} 非法作业ID或工序号: {step}")
            return float('inf')
        if op != next_op[job]:  # 保留工序顺序检查
            print(f"步骤 {idx} 工序顺序错误: 作业 {job} 预期 {next_op[job]}，实际 {op}")
            return float('inf')
        machine = machine_orders[job][op]
        proc_time = processing_times[job][op]
        start_time = max(job_finish[job], machine_times[machine])
        finish_time = start_time + proc_time
        machine_times[machine] = finish_time
        job_finish[job] = finish_time
        next_op[job] += 1

    if any(next_op[j] != num_machines for j in range(num_jobs)):  # 保留完整性检查
        print("部分作业未完成所有工序")
        return float('inf')
    return max(machine_times)

def evaluate_program(program_code: str, instance: Dict) -> Tuple[int, str]:
    lines = program_code.splitlines()
    clean_lines = [l for l in lines if not l.strip().startswith('```')]
    clean_code = '\n'.join(clean_lines)

    func_match = re.search(r'(def scheduler\s*\([^)]*\):[\s\S]*?\breturn\s+schedule)', clean_code)
    if not func_match:
        return float('inf'), "未找到完整的 scheduler 定义或 return schedule"
    func_def = func_match.group(1)

    body_match = re.search(r'def scheduler\s*\([^)]*\):(.*)', func_def, re.S)
    if not body_match:
        return float('inf'), "无法解析函数体结构"
    body = textwrap.dedent(body_match.group(1))
    indented = ''.join('    ' + line + '\n' for line in body.splitlines())
    new_code = f"def scheduler(data):\n{indented}    return schedule\n"

    # === 关键修复：使用完整的 safe_builtins ===
    safe_builtins = {
        'range': range,
        'len': len,
        'min': min,
        'max': max,
        'sum': sum,
        'sorted': sorted,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'list': list,
        'float': float,  # 新增float
    }
    restricted_globals = {
        '__builtins__': safe_builtins,
        'instance': instance['processing_times'],
        '__name__': '__main__'
    }
    # === 修复结束 ===

    try:
        exec(new_code, restricted_globals)
    except Exception as e:
        return float('inf'), f"编译错误: {e}"

    def safe_exec_wrapper():
        scheduler = restricted_globals['scheduler']
        schedule = scheduler(restricted_globals['instance'])
        if not isinstance(schedule, list):
            raise TypeError("调度结果必须为列表类型")
        return schedule

    try:
        schedule = func_timeout(15, safe_exec_wrapper)  # 5秒超时
    except FunctionTimedOut:
        return float('inf'), '代码执行超时（5秒）'  # 明确超时时间
    except Exception as e:
        return float('inf'), f"执行错误: {e}"

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
        """增强版生成函数，包含详细诊断日志"""
        prompt += f"\n\n# 格式要求：{SCHEDULER_REGEX}"
        for attempt in range(MAX_RETRIES):
            print(f"\n=== API尝试 {attempt+1}/{MAX_RETRIES} ===")
            print(f"当前提示长度：{len(prompt)} 字符")
            print("提示内容（前100字符）：", prompt[:100] + ("..." if len(prompt) > 100 else ""))

            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "严格按用户要求生成scheduler函数，不添加额外内容"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                    timeout=30  # API调用超时时间（秒）
                )
                duration = time.time() - start
                print(f"✅ API成功响应（耗时{duration:.2f}s）")

                # 提取生成内容
                generated = response.choices[0].message.content.strip()
                print(f"生成代码（前50字符）：{generated[:50]}...（总长度{len(generated)}）")

                # 格式验证
                if re.match(SCHEDULER_REGEX, generated, re.MULTILINE):
                    print("✅ 代码格式匹配正则")
                    return generated
                else:
                    print("❌ 格式错误：未匹配scheduler函数正则")
                    print("生成内容开头：", generated[:100])  # 打印开头帮助定位
                    return None

            except FunctionTimedOut:
                print("❗️ 错误：API调用超时（30秒未响应）")
                print("网络可能不稳定，或模型处理时间过长")
            except Exception as e:
                print(f"❗️ 错误：API请求失败 - {type(e).__name__}")
                print("详细信息：", str(e))
                # 打印完整堆栈跟踪（调试用）
                import traceback
                traceback.print_exc()

            # 计算并显示重试延迟
            delay = BASE_DELAY ** attempt
            print(f"⏳ 第{attempt+1}次重试延迟：{delay:.1f}秒")
            time.sleep(delay)

        print("❌ 所有API尝试失败，返回空")
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
            """使用NEH启发式生成初始解：按总处理时间降序排序"""
            return f"""
    def scheduler(instance):
        # 1. 计算每个作业的总处理时间
        job_total_time = [sum(task_times) for task_times in instance]
        # 2. 按总时间降序排序作业
        jobs = sorted(range(len(instance)), key=lambda j: -job_total_time[j])
        # 3. 生成顺序调度（作业按顺序，工序按机器顺序）
        schedule = []
        for job in jobs:
            for op in range(len(instance[0])):  # 工序0到num_machines-1
                schedule.append((job, op))
        return schedule
    """

    def _build_prompt(self) -> str:
        """包含调度理论的详细提示"""
        error_str = "\n".join([f"- {e}" for e in self.error_history[-3:]]) or "无近期错误"

        return f"""
    # 重要：请基于以下调度优化策略编写代码：
    1. **优先规则**：使用LPT（长作业优先）规则，先处理总处理时间长的作业
    2. **插入策略**：对于每个作业，找到插入后makespan最小的位置（类似NEH算法第二步）
    3. **工序约束**：每个作业的工序必须按0→1→...→(machines-1)顺序，不得跳跃
    4. **数据访问**：instance[j][op]表示作业j在工序op的处理时间

    # 当前最佳makespan: {min(s for _, s in self.programs_db)}
    # 最近错误（最多3条）:
    {error_str}

    # 当前代码（需优化）:
    {self.programs_db[0][0]}

    请输出改进的scheduler函数，要求包含排序和插入逻辑以最小化makespan。
    """

    def _update_population(self, code: str, score: float):
        if code in (c for c, _ in self.programs_db):
            print("重复程序，跳过")
            return
        self.programs_db.append((code, score))
        self.programs_db.sort(key=lambda x: x[1])
        self.programs_db = self.programs_db[:POPULATION_SIZE]

    def evolve(self):
        for i in range(MAX_ITERATIONS):
            print(f"\n{'=' * 10} 迭代 {i + 1}/{MAX_ITERATIONS} {'=' * 10}")
            current_best = min(score for _, score in self.programs_db)
            print(f"▶ 当前最佳: {current_best}")

            # 构建并显示提示摘要
            prompt = self._build_prompt()
            print("\n📝 提示摘要（前300字符）:")
            print(prompt[:300].replace('\n', ' ') + "..." if len(prompt) > 300 else prompt)

            # 生成代码
            new_code = self.llm.generate(prompt)
            if not new_code:
                print("🚫 无有效代码，跳过本次迭代")
                continue

            # 强制打印完整生成代码（用分隔符明显区分）
            print("\n=== 生成的完整代码开始 ===")
            print(new_code)
            print("=== 生成的完整代码结束 ===\n")

            # 执行评估并计时
            start = time.perf_counter()
            score, err = evaluate_program(new_code, self.instance)
            duration = time.perf_counter() - start

            if err:
                # 解析超时类型（API超时 vs 代码执行超时）
                if '代码执行超时' in err:
                    print(f"⏰ 执行超时: 代码运行超过5秒（耗时{duration:.2f}s）")
                else:
                    print(f"❌ 执行错误: {err}（耗时{duration:.2f}s）")
                self.error_history.append(f"{err} (代码长度{len(new_code)})")
            else:
                print(f"✅ 评估通过: makespan={score}（耗时{duration:.2f}s）")
                if score < current_best:
                    print("🎉 新最佳解！")

            self._update_population(new_code, score)
            print(f"🔄 种群当前最优: {min(s for _, s in self.programs_db)}\n")

def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return
    path = os.path.join(TAILLARD_DIR, 'ta062.dat')
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