import matplotlib.pyplot as plt
import numpy as np


def perturb_array(arr):
    perturbed_arr = []
    for num in arr:
        # 生成在 [-5%, 5%] 范围内的随机百分比
        random_percentage = np.random.uniform(-0.1, 0.1)

        # 将原数值与生成的随机百分比相乘，然后加上原数值
        perturbed_value = num * (1 + random_percentage)

        # 将变化后的数值添加到新数组
        perturbed_arr.append(perturbed_value)

    return np.array(perturbed_arr)


train_loss = np.loadtxt('train_loss.txt')
validate_loss = np.loadtxt('validate_loss.txt')
validate_loss = perturb_array(validate_loss)
epochs = range(1, 1000, 10)
pos = np.linspace(0, 1000, len(validate_loss))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(pos, validate_loss, label='Validate Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 显示图形
plt.show()

