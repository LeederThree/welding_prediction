import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, random_split
from make_dataset import split_dataset, WeldData
from vit_pytorch import ViT
from multiprocessing import cpu_count
import os
import pickle
from resnet_layer import resnet18

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    # print("Running on the GPU")
else:
    device = torch.device("cpu")
    # print("Running on the CPU")

batch_size = 2560
learning_rate = 1e-3
num_epochs = 100
num_workers = 8

data_path = 'C:\\Users\\yimen\\resized_img\\flatten_dataset'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
if __name__ == '__main__':
    # if os.path.exists('dataset/train_dataset.pkl'):
    #     split_dataset(data_path, transform)
    # with open('dataset/train_dataset.pkl', 'rb') as file:
    #     train_dataset = pickle.load(file)
    # with open('dataset/validate_dataset.pkl', 'rb') as file:
    #     validate_dataset = pickle.load(file)
    weld_dataset = WeldData(data_path, transform=transform)
    train_size = int(len(weld_dataset) * 0.6)
    valid_size = int(len(weld_dataset) * 0.2)
    test_size = len(weld_dataset) - valid_size - train_size
    train_dataset, validate_dataset, test_dataset = random_split(weld_dataset, [train_size, valid_size, test_size])
    print(len(weld_dataset.img_path))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=6,
        dim=128,
        depth=1,
        heads=1,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    model = resnet18.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        model.train()
        count = 0
        iter_start = time.time()
        for images, voltage, current, sound, labels in train_loader:
            images = images.to(device)
            voltage = voltage.to(device)
            current = current.to(device)
            sound = sound.to(device)
            labels = labels.to(device)
            print(f"voltage tensor: {voltage.shape}")
            print(f"current tensor: {current.shape}")
            print(f"sound tensor: {sound.shape}")
            # load_time = time.time()
            # optimizer.zero_grad()
            # outputs = model(images)
            # refer_time = time.time()
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # iter_end = time.time()
            # count += 1
            # print(f"iter {count}: load time: {load_time-iter_start}, refer time: {refer_time - load_time}, whole iter: {iter_end - iter_start}")
            # iter_start = iter_end
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # with torch.no_grad():
        #     for inputs, labels in validate_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()
        #         _, pred = outputs.max(1)
        #         total += labels.size(0)
        #         correct += pred.eq(labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Validation Loss: {val_loss/len(validate_loader):.4f}, '
              f'Validation Accuracy: {100.*correct/total:.2f}%')
