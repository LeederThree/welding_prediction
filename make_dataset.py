from torch.utils.data import Dataset, TensorDataset
from PIL import Image
from data_process import traversal_files
from random import shuffle
import torch
from tqdm import tqdm
import pickle
import os
import re


label_dict = {
    "正常": 0,
    "焊偏": 1,
    "卡丝": 2,
    "气孔": 3,
    "烧穿": 4,
    "未焊透": 5
}


def split_dataset(root_dir, transform):
    img_paths = traversal_files(root_dir)[0]
    shuffle(img_paths)
    total_len = len(img_paths)
    train_len = int(0.6 * total_len)
    valid_len = int(0.2 * total_len)

    train_paths = img_paths[:train_len]
    valid_paths = img_paths[train_len:train_len+valid_len]
    test_paths = img_paths[train_len+valid_len:]
    
    train_data, train_label = traversal_set(train_paths, train_len, transform)
    train_dataset = TensorDataset(train_data, train_label)
    with open('dataset/train_dataset.pkl', 'wb') as file:
        pickle.dump(train_dataset, file)
    del train_data, train_label, train_dataset
    valid_data, valid_label = traversal_set(valid_paths, valid_len, transform)
    valid_dataset = TensorDataset(valid_data, valid_label)
    with open('dataset/validate_dataset.pkl', 'wb') as file:
        pickle.dump(valid_dataset, file)
    del valid_data, valid_label, valid_dataset
    test_data, test_label = traversal_set(test_paths, len(test_paths), transform)
    test_dataset = TensorDataset(test_data, test_label)
    with open('dataset/test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)
    del test_data, test_label, test_dataset
    

def traversal_set(paths, length, transform):
    tensor_shape = (length, 3, 224, 224)
    img_tensor = torch.zeros(tensor_shape)
    label_tensor = torch.zeros(length, dtype=torch.int64)

    for i in tqdm(range(length), desc="Loading Images"):
        img = Image.open(paths[i]).convert('RGB')
        img_tensor[i] = transform(img)
        label_tensor[i] = label_dict[paths[i].split("\\")[-4]]
    return img_tensor, label_tensor


def split_voltage_current(voltage_paths):
    for voltage_file in voltage_paths:
        voltage_dir = os.path.dirname(voltage_file)
        pic_number = get_pic_numbers(voltage_dir)
        seq_len = len(pic_number)
        with open(voltage_file, 'r') as file:
            voltages = file.readlines()
            voltage_list = [float(line.strip()) for line in voltages]
        original_dir = voltage_dir.replace('20230109LABEL1最精准', '20230109 - 制备')
        original_pics = get_pic_numbers(original_dir)
        original_len = len(original_pics)
        offset = min(pic_number) - min(original_pics)
        seg_len = len(voltage_list) // original_len
        voltage_tensor = torch.zeros(seq_len, seg_len)
        for i in range(seq_len):
            start = i * seg_len + offset
            end = (i + 1) * seg_len + offset
            voltage_tensor[i] = torch.tensor(voltage_list[start:end])
        print(voltage_tensor.shape)
        save_tensors(voltage_tensor, voltage_file)
        

        
def get_pic_numbers(dir):
    dir_files = os.listdir(dir)
    pic_numbers = []
    for file in dir_files:
        file_name, file_ext = os.path.splitext(file)
        if re.search('\(1\)', file_name):
            continue
        if file_ext == '.png':
            pic_numbers.append(int(file_name))
    return pic_numbers


def save_tensors(tensor, tensor_path):
    root_path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1-VoltCurSoundTensors'
    dir_list = tensor_path.split('\\')
    tensor_type = dir_list[-1].split('.')[0]
    label = str(label_dict[dir_list[-4]])
    tensor_name = tensor_type + '-' + '-'.join(dir_list[-3:-1]) + '.pt'
    save_dir = '\\'.join([root_path, label])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(tensor, '\\'.join([save_dir, tensor_name]))
    print(tensor_name)



class WeldData(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.img_path = traversal_files(root_dir)[0]
        self.transform = transform
        # print(self.img_path)
    
    def __getitem__(self, index):
        img_item_path = self.img_path[index]
        img = Image.open(img_item_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = label_dict[img_item_path.split("\\")[-4]]
        # print(label)
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    # data_path = 'G:\\resized_img\\20230109LABEL1最精准'
    # train_dataset = WeldData(data_path)
    # train_dataset.__getitem__(0)
    # print(train_dataset.__len__())
    path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1最精准'
    _, voltage_paths, current_paths, sound_paths = traversal_files(path)
    voltage_tensor = split_voltage_current(voltage_paths)
    current_tensor = split_voltage_current(current_paths)
    sound_tensor = split_voltage_current(sound_paths)