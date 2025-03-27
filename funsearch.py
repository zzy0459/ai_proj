import openai
import random
import os

# 直接在代码中设置 OpenAI API 密钥（请替换为您的实际密钥）
openai.api_key = 'sk-71U9TYaINmhsBgTy21705cC937A1476a9870855922F9F8Ee'


# 加载 Taillard 数据集
def load_taillard_instance(file_path):
    """加载 Taillard 数据集中的一个实例。"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            num_jobs, num_machines = map(int, lines[0].split())
            processing_times = []
            machine_orders = []
            for line in lines[1:num_jobs + 1]:
                data = list(map(int, line.split()))
                times = data[::2]
                machines = data[1::2]
                processing_times.append(times)
                machine_orders.append(machines)
        return {
            'num_jobs': num_jobs,
            'num_machines': num_machines,
            'processing_times': processing_times,
            'machine_orders': machine_orders
        }
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


# 计算 makespan 的辅助函数
def calculate_makespan(schedule, instance):
    """根据调度方案计算 makespan。"""
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    processing_times = instance['processing_times']
    machine_orders = instance['machine_orders']

    machine_times = [0] * num_machines
    job_times = [0] * num_jobs

    for job, op in schedule:
        machine = machine_orders[job][op]
        proc_time = processing_times[job][op]
        start_time = max(machine_times[machine], job_times[job])
        machine_times[machine] = start_time + proc_time
        job_times[job] = start_time + proc_time

    return max(machine_times)


# 初始调度程序
initial_program = """
def scheduler(instance):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    schedule = []
    for job in range(num_jobs):
        for op in range(num_machines):
            schedule.append((job, op))
    return schedule
"""


# 评估函数
def evaluate_program(program_code, instance):
    """评估调度程序的 makespan。"""
    try:
        exec_globals = {}
        exec(program_code, exec_globals)
        if 'scheduler' not in exec_globals:
            print("未找到 'scheduler' 函数")
            return float('inf')
        scheduler = exec_globals['scheduler']
        schedule = scheduler(instance)
        # 修正后的验证逻辑
        if not isinstance(schedule, list) or not all( (isinstance(item, tuple) and len(item) == 2) for item in schedule ):
            print("调度格式错误")
            return float('inf')
        makespan = calculate_makespan(schedule, instance)
        return makespan
    except Exception as e:
        print(f"评估程序时出错: {e}")
        return float('inf')


# ChatGPT LLM 类（适配 openai 1.68.2）
class ChatGPTLLM:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        self.client = openai.OpenAI(api_key=openai.api_key)  # 使用新客户端
        self.model = model
        self.temperature = temperature

    def generate(self, prompt):
        """使用 ChatGPT API 生成新的调度程序代码。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that generates Python code for scheduling algorithms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                n=1
            )
            generated_code = response.choices[0].message.content
            return generated_code
        except Exception as e:
            print(f"调用 ChatGPT API 时出错: {e}")
            return None


# FunSearch 类
class FunSearch:
    def __init__(self, evaluate, initial_program, num_iterations=100, population_size=10):
        """初始化 FunSearch。"""
        self.evaluate = evaluate
        self.programs_db = [(initial_program, self.evaluate(initial_program))]
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.llm = ChatGPTLLM(model="gpt-3.5-turbo", temperature=0.7)

    def sample_programs(self):
        """从程序数据库中采样高分程序并生成提示。"""
        sorted_db = sorted(self.programs_db, key=lambda x: x[1])
        best_programs = [prog for prog, _ in sorted_db[:self.population_size]]
        prompt = (
            "你需要改进 Flow Job Shop Scheduling 问题的调度算法，目标是最小化 makespan。\n"
            "以下是一些当前表现最好的调度程序代码，请根据它们生成一个新的、改进的调度函数。\n\n"
        )
        for i, prog in enumerate(best_programs, 1):
            prompt += f"程序 {i}:\n{prog}\n\n"
        prompt += (
            "请生成一个新的调度函数，可能比上述程序更优。\n"
            "确保函数名为 'scheduler'，接受一个 'instance' 字典作为输入，返回一个 (job, operation) 元组的列表。"
        )
        return prompt

    def evolve(self):
        """进化程序。"""
        for i in range(self.num_iterations):
            print(f"迭代 {i + 1}/{self.num_iterations}")
            prompt = self.sample_programs()
            new_program = self.llm.generate(prompt)
            if new_program:
                score = self.evaluate(new_program)
                if score < float('inf'):
                    self.programs_db.append((new_program, score))
                    self.programs_db = sorted(self.programs_db, key=lambda x: x[1])[:self.population_size]

    def get_best_program(self):
        """返回得分最低（makespan 最小）的程序。"""
        return min(self.programs_db, key=lambda x: x[1])


# 主函数
def main():
    # 加载 Taillard 实例（请替换为实际文件路径）
    instance = load_taillard_instance('TestSolutions/taillard/ta001.dat')
    if instance is None:
        return

    # 定义评估函数
    def evaluate(program_code):
        return evaluate_program(program_code, instance)

    # 初始化 FunSearch
    funsearch = FunSearch(
        evaluate=evaluate,
        initial_program=initial_program,
        num_iterations=10,
        population_size=3
    )

    # 运行 FunSearch
    funsearch.evolve()

    # 获取最佳程序
    best_program, best_score = funsearch.get_best_program()
    print(f"最佳 makespan: {best_score}")
    print("最佳程序:")
    print(best_program)


if __name__ == "__main__":
    main()