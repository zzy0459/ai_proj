import openai
import random
import os
from typing import List, Tuple, Dict

# 配置参数
OPENAI_API_KEY = 'sk-71U9TYaINmhsBgTy21705cC937A1476a9870855922F9F8Ee'  # 替换为有效API密钥
TAILLARD_DIR = 'TestSolutions/taillard'  # 数据集目录
MAX_ITERATIONS = 10
POPULATION_SIZE = 3


def load_taillard_instance(file_path: str) -> Dict:
    """正确解析流水车间(Flow Shop)格式的Taillard数据"""
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

            # 解析作业数和机器数
            num_jobs, num_machines = map(int, lines[0].split())

            # 读取机器处理时间矩阵
            processing_times = []
            for m in range(num_machines):
                data = list(map(int, lines[m + 1].split()))
                if len(data) != num_jobs:
                    print(f"机器{m + 1}数据不完整，期望{num_jobs}个作业，实际{len(data)}个")
                    return None
                processing_times.append(data)

            # 转换为作业优先格式 [作业][机器]
            job_processing = [
                [processing_times[m][j] for m in range(num_machines)]
                for j in range(num_jobs)
            ]

            # 流水车间不需要机器顺序（固定顺序0,1,2,...）
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


# 强化版makespan计算
def calculate_makespan(schedule: List[Tuple[int, int]], instance: Dict) -> int:
    """带安全检查的makespan计算"""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    processing_times = instance['processing_times']
    machine_orders = instance['machine_orders']

    # 初始化跟踪器
    machine_times = [0] * num_machines
    job_progress = [0] * num_jobs  # 每个作业的当前工序

    for step in schedule:
        # 安全检查
        if len(step) != 2:
            print(f"无效调度步骤: {step}")
            return float('inf')

        job, op = step

        # 范围检查
        if not (0 <= job < num_jobs):
            print(f"非法作业ID: {job}")
            return float('inf')
        if not (0 <= op < num_machines):
            print(f"非法工序号: {op}")
            return float('inf')

        # 工序顺序验证
        if op != job_progress[job]:
            print(f"工序顺序错误: 作业{job}预期工序{job_progress[job]}，实际{op}")
            return float('inf')

        # 获取机器和处理时间
        try:
            machine = machine_orders[job][op]
            proc_time = processing_times[job][op]
        except IndexError as e:
            print(f"数据访问错误: {str(e)}")
            return float('inf')

        # 计算开始时间
        start_time = max(machine_times[machine], job_progress[job])

        # 更新时间
        machine_times[machine] = start_time + proc_time
        job_progress[job] = start_time + proc_time

    return max(machine_times)


# 强化版评估函数
def evaluate_program(program_code: str, instance: Dict) -> float:
    """带多维验证的评估函数"""
    try:
        exec_env = {}
        exec(program_code, exec_env)

        # 函数存在性检查
        if 'scheduler' not in exec_env:
            print("未找到scheduler函数")
            return float('inf')

        scheduler = exec_env['scheduler']
        schedule = scheduler(instance)

        # 类型检查
        if not isinstance(schedule, list):
            print("调度结果非列表类型")
            return float('inf')

        # 元素格式检查
        valid = True
        for step in schedule:
            if not (isinstance(step, tuple) and len(step) == 2):
                print(f"无效步骤格式: {step}")
                valid = False
            job, op = step
            if not (0 <= job < instance['num_jobs']):
                print(f"作业ID越界: {job}")
                valid = False
            if not (0 <= op < instance['num_machines']):
                print(f"工序号越界: {op}")
                valid = False

        if not valid:
            return float('inf')

        return calculate_makespan(schedule, instance)

    except Exception as e:
        print(f"评估异常: {str(e)}")
        return float('inf')


# 改进版LLM类
class ChatGPTLLM:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.7

    def generate(self, prompt: str) -> str:
        """带重试机制的生成函数"""
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专注于生成安全、高效调度算法的AI助手"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except openai.APITimeoutError:
                print("API请求超时，正在重试...")
            except Exception as e:
                print(f"API错误: {str(e)}")
                return None
        return None


# 强化版FunSearch
class FunSearch:
    def __init__(self, instance: Dict):
        self.instance = instance
        self.llm = ChatGPTLLM()
        self.programs_db = []

        # 初始化种群
        initial_program = self._create_initial_program()
        initial_score = evaluate_program(initial_program, instance)
        self.programs_db.append((initial_program, initial_score))

    def _create_initial_program(self) -> str:
        """生成安全初始程序"""
        return f'''
def scheduler(instance):
    num_jobs = {self.instance['num_jobs']}
    num_machines = {self.instance['num_machines']}
    return [(j, o) for j in range(num_jobs) for o in range(num_machines)]
'''

    def evolve(self):
        """带安全约束的进化过程"""
        for iter in range(MAX_ITERATIONS):
            print(f"\n=== 迭代 {iter + 1}/{MAX_ITERATIONS} ===")

            # 生成提示
            prompt = self._build_prompt()

            # 生成新程序
            new_code = self.llm.generate(prompt)
            if not new_code:
                continue

            # 评估新程序
            score = evaluate_program(new_code, self.instance)
            print(f"评估得分: {score}")

            # 更新种群
            self._update_population(new_code, score)

    def _build_prompt(self) -> str:
        """构建安全提示"""
        prompt = f'''请生成改进的调度程序，需满足：
- 处理{self.instance['num_jobs']}个作业和{self.instance['num_machines']}台机器
- 每个作业必须按顺序完成所有工序
- 必须包含完整的安全检查

参考代码：
{self.programs_db[0][0]}

请生成新的安全优化代码：'''
        return prompt

    def _update_population(self, new_code: str, score: float):
        """带多样性保护的种群更新"""
        # 去除重复程序
        existing_codes = [code for code, _ in self.programs_db]
        if new_code in existing_codes:
            print("发现重复程序，跳过")
            return

        self.programs_db.append((new_code, score))
        # 按分数排序并保留最优
        self.programs_db.sort(key=lambda x: x[1])
        self.programs_db = self.programs_db[:POPULATION_SIZE]


# 主流程
def main():
    # 加载实例
    instance_path = os.path.join(TAILLARD_DIR, 'ta001.dat')
    instance = load_taillard_instance(instance_path)
    if not instance:
        return

    # 初始化FunSearch
    funsearch = FunSearch(instance)

    # 运行进化
    funsearch.evolve()

    # 输出结果
    best_program, best_score = min(funsearch.programs_db, key=lambda x: x[1])
    print(f"\n最佳makespan: {best_score}")
    print("最佳程序代码:")
    print(best_program)


if __name__ == "__main__":
    main()