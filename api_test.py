import os
from openai import OpenAI


def test_api_connection():
    """API连接性测试函数"""
    print("=== 开始API连接测试 ===")

    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print(f"1. 环境变量检查: {'通过' if api_key else '失败'}")
    if not api_key:
        print("   请设置环境变量: export DASHSCOPE_API_KEY='your_api_key'")
        return False

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 测试请求
    test_prompt = "请回答9.11和9.9哪个数字更大，只需回答数字本身"
    print(f"2. 发送测试请求（模型: deepseek-r1）")
    print(f"   测试问题: {test_prompt}")

    try:
        # 带超时设置（10秒）
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            timeout=10
        )
    except Exception as e:
        print(f"3. 连接失败: {str(e)}")
        print("   可能原因：")
        print("   - 无效的API密钥")
        print("   - 网络连接问题（请检查是否能访问阿里云服务）")
        print("   - 地域限制（确保使用国内网络环境）")
        return False

    # 验证响应结构
    if not response.choices:
        print("3. 连接异常：响应格式不正确")
        return False

    # 输出完整响应
    print("3. 连接成功！服务器返回：")
    print("   思考过程：")
    print(response.choices[0].message.reasoning_content)
    print("   最终答案：")
    print(response.choices[0].message.content.strip())
    return True


if __name__ == "__main__":
    if test_api_connection():
        print("\n✅ 测试通过！API连接正常")
    else:
        print("\n❌ 测试失败，请根据提示检查配置")