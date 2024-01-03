from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from weld_data import WeldData
import os
from collections import Counter


def calculate_metrics(model, dataloader, device):
    model.eval()
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for inputs, concat_tensor, labels in dataloader:
            inputs, concat_tensor, labels = inputs.to(device), concat_tensor.to(device), labels.to(device)
            cnn_outputs, vit_outputs, outputs = model(concat_tensor, inputs)
            _, predicted_labels = outputs.max(1)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, average='weighted')
    recall = recall_score(all_true_labels, all_predicted_labels, average='weighted')
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     # device = torch.device("cuda:1")  # you can continue going on here, like cuda:1 cuda:2....etc.
    #     torch.cuda.set_device(1)
    #     # print("Running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     # print("Running on the CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current CUDA device:", torch.cuda.current_device())
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 100
    num_workers = 8
    save_interval = 10

    data_path = 'C:\\Users\\yimen\\resized_img\\flatten_dataset'
    model_path = './train_model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # if not os.path.exists(f'{data_path}\\dataset\\train_dataset.pkl'):
    #     split_dataset(data_path, transform)
    # with open(f'{data_path}\\dataset\\train_dataset.pkl', 'rb') as file:
    #     train_dataset = pickle.load(file)
    # with open(f'{data_path}\\dataset\\validate_dataset.pkl', 'rb') as file:
    #     validate_dataset = pickle.load(file)
    weld_dataset = WeldData(data_path, transform=transform)
    train_size = int(len(weld_dataset) * 0.6)
    valid_size = int(len(weld_dataset) * 0.2)
    test_size = len(weld_dataset) - valid_size - train_size
    train_dataset, validate_dataset, test_dataset = random_split(weld_dataset, [train_size, valid_size, test_size])
    print(len(weld_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    class_counts = Counter()

    # 遍历 DataLoader 统计类别数量
    for batch_data, _, batch_labels in validate_loader:
        # 在这里可以根据实际情况更新数据
        class_counts.update(batch_labels.numpy())

    # 打印类别数量
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples")