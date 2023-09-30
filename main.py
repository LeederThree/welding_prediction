import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, random_split
from make_dataset import WeldData
from vit_pytorch import ViT

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

batch_size = 96
learning_rate = 1e-3
num_epochs = 100

data_path = 'G:\\resized_img\\20230109LABEL1最精准'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

weld_dataset = WeldData(data_path, transform=transform)
train_size = int(len(weld_dataset) * 0.6)
valid_size = int(len(weld_dataset) * 0.2)
test_size = len(weld_dataset) - valid_size -train_size
train_dataset, validate_dataset, test_dataset = random_split(weld_dataset, [train_size, valid_size, test_size])
print(len(weld_dataset.img_path))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().cuda()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss =0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validate_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Validation Loss: {val_loss/len(validate_loader):.4f}, '
          f'Validation Accuracy: {100.*correct/total:.2f}%')