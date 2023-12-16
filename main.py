# import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from make_dataset import split_dataset
from weld_data import WeldData
from fusion_model import Simple1DCNN, ViTModel, FusionModel, VGG19, ViTB16Model
# from multiprocessing import cpu_count
import os
import pickle
# from resnet_layer import resnet18
import matplotlib.pyplot as plt


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
    batch_size = 16
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    cnn_model = VGG19(in_channel=12, classes=18).to(device)
    # cnn_model = Simple1DCNN(input_size=75, num_channels=12, num_classes=6).to(device)
    # pic_model = ViTModel(image_size=224, num_classes=6).to(device)
    pic_model = ViTB16Model(num_classes=6).to(device)
    # print(pic_model)
    model = FusionModel(cnn_model, pic_model, features=24, num_classes=6).to(device)
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(f'{model_path}/vit_vgg_fusion_30.pth'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss().to(device)

    train_loss = []
    validate_loss = []
    for epoch in range(num_epochs):
        pic_model.train()
        count = 1
        # iter_start = time.time()
        epoch_loss = 0.0
        for images, concat_tensor, labels in tqdm(train_loader, desc=f"Training epoch {epoch} in {num_epochs}", leave=False):
            images = images.to(device)
            concat_tensor = concat_tensor.to(device)
            labels = labels.to(device)
            # print(f"concat tensor: {concat_tensor.shape}")
            # print(f"voltage tensor: {voltage.shape}")
            # print(f"current tensor: {current.shape}")
            # print(f"sound tensor: {sound.shape}")
            # load_time = time.time()
            optimizer.zero_grad()
            outputs = pic_model(images)
            # refer_time = time.time()
            loss = criterion(outputs, labels.long())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # iter_end = time.time()
            # count += 1
            # print(f"iter {count}: load time: {load_time-iter_start}, refer time: {refer_time - load_time}, whole iter: {iter_end - iter_start}")
            # iter_start = iter_end
        train_loss.append(round(epoch_loss/len(train_loader), 4))

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, concat_tensor, labels in tqdm(validate_loader, desc=f"Validating epoch {epoch} in {num_epochs}", leave=False):
                inputs, concat_tensor, labels = inputs.to(device), concat_tensor.to(device), labels.to(device)
                outputs = model(concat_tensor, inputs)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        validate_loss.append(round(val_loss/len(validate_loader), 4))
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Validation Loss: {val_loss/len(validate_loader):.4f}, '
              f'Validation Accuracy: {100.*correct/total:.2f}%')
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"{model_path}/vit_vgg_fusion_{epoch}.pth")
            print(f"Model saved at epoch {epoch}")
    with open('train_loss.txt', 'w') as file:
        for loss in train_loss:
            file.write(f"{loss}\n")
    with open('validate_loss.txt', 'w') as file:
        for loss in validate_loss:
            file.write(f"{loss}\n")
    epochs = range(1, 1000, 10)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, validate_loss, label='Validate Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 显示图形
    plt.show()
