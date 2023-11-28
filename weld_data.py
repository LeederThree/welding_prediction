from torch.utils.data import Dataset
from data_process import traversal_files
import cv2
import torch
import os


# 自定义的Dataset类别，用来加载数据集
class WeldData(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.img_path = traversal_files(root_dir + '\\images')[0]
        self.transform = transform
        self.cache = dict()
        # print(self.img_path)

    def __getitem__(self, index):
        img_item_path = self.img_path[index]
        img = cv2.cvtColor(cv2.imread(img_item_path), cv2.COLOR_BGR2RGB)
        # img = Image.open(img_item_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_name = os.path.basename(img_item_path).rstrip('.png')
        img_index, img_offset = img_name.rsplit('_', 1)
        img_offset = int(img_offset)
        label = int(img_item_path.split("\\")[-2])
        if img_index not in self.cache.keys():
            voltage_tensor = torch.load(self.root_dir + f'\\tensors\\Voltage\\{str(label)}\\{img_index}.pt')
            current_tensor = torch.load(self.root_dir + f'\\tensors\\Current\\{str(label)}\\{img_index}.pt')
            sound_tensor = torch.load(self.root_dir + f'\\tensors\\Sound\\{str(label)}\\{img_index}.pt')
            self.cache[img_index] = [voltage_tensor, current_tensor, sound_tensor]
        else:
            voltage_tensor = self.cache[img_index][0]
            current_tensor = self.cache[img_index][1]
            sound_tensor = self.cache[img_index][2]
        try:
            voltage = voltage_tensor[img_offset]
            current = current_tensor[img_offset]
            sound = sound_tensor[img_offset]
            sound = sound.view(voltage_tensor.shape[1], -1)
            concat_tensor = torch.zeros(voltage.shape[0], 12)
            concat_tensor[:, 0] = voltage
            concat_tensor[:, 1] = current
            concat_tensor[:, 2:] = sound
            # print(voltage_tensor.shape)
        except IndexError:
            print(img_offset)
            print(voltage_tensor.shape)
            return None
        # print(label)
        return img, concat_tensor.T, label

    def __len__(self):
        return len(self.img_path)
