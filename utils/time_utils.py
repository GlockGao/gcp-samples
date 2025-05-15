import time
import functools

def timing_decorator(func):
    """
    一个装饰器，用于测量被装饰函数的执行时间。
    """
    @functools.wraps(func)  # 保留原始函数的元信息（如函数名、文档字符串等）
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # 转换为毫秒

        print(f"函数 '{func.__name__}' 执行时间: {execution_time:.4f} 毫秒")
        return result
    return wrapper


if __name__ == "__main__":
    @timing_decorator
    def test_function_decorator():
        time.sleep(0.1)
        print("这是装饰器测试函数")

    print("--- 内部测试 `timing_decorator` ---")
    test_function_decorator()