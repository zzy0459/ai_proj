# api_test_advanced.py
import os
import socket
import httpx
from openai import OpenAI


def check_network():
    """深度网络诊断"""
    try:
        # 测试DNS解析
        print("DNS解析结果:", socket.gethostbyname('dashscope.aliyuncs.com'))

        # 测试TCP连接
        with socket.create_connection(('dashscope.aliyuncs.com', 443), timeout=5) as sock:
            print("TCP连接成功")

        # HTTPS证书验证
        response = httpx.get('https://dashscope.aliyuncs.com', verify=True)
        print(f"HTTPS证书状态: {response.status_code}")

        return True
    except Exception as e:
        print(f"网络诊断失败: {str(e)}")
        return False


def test_api_connection():
    print("=== 深度网络诊断 ===")
    if not check_network():
        return False

    print("\n=== API连接测试 ===")
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=30,
        http_client=httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=3),
            transport=httpx.HTTPTransport(retries=3)
        )
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[{"role": "user", "content": "9.11和9.9谁大？直接回答数字"}],
            timeout=30
        )
        print("测试成功！响应摘要：")
        print(response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"最终失败: {str(e)}")
        return False


if __name__ == "__main__":
    if test_api_connection():
        print("\n✅ 所有检查通过")
    else:
        print("\n❌ 请根据提示检查网络配置")