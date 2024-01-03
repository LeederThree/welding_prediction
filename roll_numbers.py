import random


def generate_random_numbers(n):
    random_numbers = [round(random.uniform(94, 99), 2) for _ in range(n)]
    return random_numbers


# 生成 10 个数值的例子
n = 18
random_values = generate_random_numbers(n)

print(sorted(random_values))
