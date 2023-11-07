from torch.utils.data import Dataset, TensorDataset
from PIL import Image
import cv2
from data_process import traversal_files
from random import shuffle
import torch
from tqdm import tqdm
import pickle
import os
import shutil
import re
import json

label_dict = {
    "正常": 0,
    "焊偏": 1,
    "卡丝": 2,
    "气孔": 3,
    "烧穿": 4,
    "未焊透": 5
}


# 直接将全部图片转换为Tensor加载到内存中
def split_dataset(root_dir, transform):
    img_paths = traversal_files(root_dir)[0]
    shuffle(img_paths)
    total_len = len(img_paths)
    train_len = int(0.6 * total_len)
    valid_len = int(0.2 * total_len)

    train_paths = img_paths[:train_len]
    valid_paths = img_paths[train_len:train_len + valid_len]
    test_paths = img_paths[train_len + valid_len:]

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


# 遍历图片目录
def traversal_set(paths, length, transform):
    tensor_shape = (length, 3, 224, 224)
    img_tensor = torch.zeros(tensor_shape)
    label_tensor = torch.zeros((length, 2), dtype=torch.int64)
    dir_start_num = dict()
    data_path_dict = dict()
    for i in tqdm(range(length), desc="Loading Images"):
        img = Image.open(paths[i]).convert('RGB')
        img_dir = os.path.dirname(paths[i])
        img_file_name = os.path.basename(paths[i])
        img_name = os.path.splitext(img_file_name)[0]
        if img_dir not in dir_start_num:
            pic_number = get_pic_numbers(img_dir)
            start_num = min(pic_number)
            dir_start_num[img_dir] = start_num
        offset = int(img_name) - dir_start_num[img_dir]
        label = label_dict[paths[i].split("\\")[-4]]
        img_tensor[i] = transform(img)
        label_tensor[i][0] = label
        label_tensor[i][1] = offset
        if img_dir not in data_path_dict:
            voltage_tensor_path = '/'.join(['20230109LABEL1-VoltCurSoundTensors', str(label),
                                            'Voltage' + '-' + '-'.join(paths[i].split("\\")[-3:-1]) + '.pt'])
            current_tensor_path = '/'.join(['20230109LABEL1-VoltCurSoundTensors', str(label),
                                            'Current' + '-' + '-'.join(paths[i].split("\\")[-3:-1]) + '.pt'])
            sound_tensor_path = '/'.join(['20230109LABEL1-VoltCurSoundTensors', str(label),
                                          'Sound' + '-' + '-'.join(paths[i].split("\\")[-3:-1]) + '.pt'])
            data_path_dict[img_dir] = [voltage_tensor_path, current_tensor_path, sound_tensor_path]
    with open("pic_data_map.json", "w") as json_file:
        json.dump(data_path_dict, json_file)
    return img_tensor, label_tensor


# 将电流电压声音的tensor文件转移到对应目录下
def move_tensor_file(origin_dir, new_dir):
    paths = os.walk(origin_dir)
    tensor_paths = []
    for path, dir_list, file_list in paths:
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if re.search(r".pt", file_path):
                tensor_paths.append(file_path)
    reverse_label_dict = dict()
    for key, value in label_dict.items():
        reverse_label_dict[str(value)] = key
    for tensor_path in tensor_paths:
        sub_dir_list = os.path.splitext(os.path.basename(tensor_path))[0].split('-')
        new_tensor_path = '\\'.join(
            [new_dir, reverse_label_dict[tensor_path.split('\\')[-2]], sub_dir_list[1], sub_dir_list[2],
             sub_dir_list[0] + '.pt'])
        shutil.copy(tensor_path, new_tensor_path)
        print('tensor saved at: ', new_tensor_path)


# 将图片数据集和电流电压声音的tensor按标签存放，并根据原来对应的焊道编号重命名
def img_set_flatten(root_dir):
    img_paths, _, _, _, tensor_paths = traversal_files(root_dir)
    dir_cache = dict()
    for img_path in img_paths:
        img_dir = os.path.dirname(img_path)
        dir_list = img_dir.split("\\")[-2:]
        img_file_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_file_name)[0]
        if img_dir not in dir_cache:
            pic_number = get_pic_numbers(img_dir)
            dir_cache[img_dir] = pic_number
        index = dir_cache[img_dir].index(int(img_name))
        dir_list.append(str(index))
        label = label_dict[img_path.split("\\")[-4]]
        # if label == 4 and index == 8000:
        #     pass
        new_dir = f"C:\\Users\\yimen\\resized_img\\flatten_dataset\\images\\{str(label)}\\"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_img_path = new_dir + '_'.join(dir_list) + '.png'
        shutil.copy(img_path, new_img_path)
        print(f"img saved at {new_img_path}")
    for tensor_path in tensor_paths:
        label = label_dict[tensor_path.split('\\')[-4]]
        tensor_name = os.path.splitext(os.path.basename(tensor_path))[0]
        # tensor_type = tensor_name.split('-')[0]
        new_dir = f"C:\\Users\\yimen\\resized_img\\flatten_dataset\\tensors\\{tensor_name}\\{str(label)}"
        dir_list = tensor_path.split('\\')[-3:-1]
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_tensor_path = new_dir + '\\' + '_'.join(dir_list) + '.pt'
        shutil.copy(tensor_path, new_tensor_path)
        print(f"tensor saved at {new_tensor_path}")


# 根据图片数据集切分电压电流声音数据并转为tensor文件
def split_voltage_current(voltage_paths):
    pic_numbers_map = dict()
    original_number_map = dict()
    for voltage_file in voltage_paths:
        voltage_dir = os.path.dirname(voltage_file)
        if voltage_dir not in pic_numbers_map:
            pic_number = get_pic_numbers(voltage_dir)
            pic_numbers_map[voltage_dir] = pic_number
        else:
            pic_number = pic_numbers_map[voltage_dir]
        seq_len = len(pic_number)
        original_dir = voltage_dir.replace('20230109LABEL1', '20230109 - 制备')
        if original_dir not in original_number_map:
            original_pics = get_pic_numbers(original_dir)
            original_number_map[original_dir] = original_pics
        else:
            original_pics = original_number_map[original_dir]
        original_len = max(original_pics) - min(original_pics) + 1
        origin = min(original_pics)
        if origin:
            pic_number = [num - origin for num in pic_number]
        with open(voltage_file, 'r') as file:
            voltages = file.readlines()
            voltage_list = [float(line.strip()) for line in voltages]
        seg_len = len(voltage_list) // original_len
        original_tensor = torch.tensor(voltage_list)
        print(original_tensor.shape)
        reshaped_tensor = original_tensor[:seg_len * original_len].view(original_len, seg_len)
        # reshaped_tensor = original_tensor.reshape(original_len, -1)
        voltage_tensor = torch.zeros(seq_len, seg_len)
        for i in range(len(pic_number)):
            try:
                voltage_tensor[i] = reshaped_tensor[pic_number[i]]
            except IndexError:
                print(voltage_tensor.shape)
                print(reshaped_tensor.shape)
                print(pic_number[i])
        print(f"{voltage_file}: {voltage_tensor.shape}")
        save_tensors(voltage_tensor, voltage_file)


# 遍历目录得到图片的编号列表
def get_pic_numbers(pic_dir):
    dir_files = os.listdir(pic_dir)
    pic_numbers = []
    for file in dir_files:
        file_name, file_ext = os.path.splitext(file)
        if re.search(r'\(1\)', file_name):
            print(file_name)
            continue
        if file_ext == '.png':
            pic_numbers.append(int(file_name))
    return sorted(pic_numbers)


# 将转换得到的电压电流声音tensor保存在对应的目录下
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


# 自定义的Dataset类别，用来加载数据集
class WeldData(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.img_path = traversal_files(root_dir + '\\images')[0]
        self.transform = transform
        # print(self.img_path)

    def __getitem__(self, index):
        img_item_path = self.img_path[index]
        img = cv2.cvtColor(cv2.imread(img_item_path), cv2.COLOR_BGR2RGB)
        # img = Image.open(img_item_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_name = os.path.basename(img_item_path).rstrip('.png')
        img_offset = int(img_name[8:])
        img_index = img_name[:7]
        label = int(img_item_path.split("\\")[-2])
        voltage_tensor = torch.load(self.root_dir + f'\\tensors\\Voltage\\{str(label)}\\{img_index}.pt')
        current_tensor = torch.load(self.root_dir + f'\\tensors\\Current\\{str(label)}\\{img_index}.pt')
        sound_tensor = torch.load(self.root_dir + f'\\tensors\\Sound\\{str(label)}\\{img_index}.pt')
        try:
            voltage = voltage_tensor[img_offset]
            current = current_tensor[img_offset]
            sound = sound_tensor[img_offset]
        except Exception:
            print(img_offset)
            print(voltage_tensor.shape)
        # print(label)
        return img, voltage, current, sound, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    # data_path = 'G:\\resized_img\\20230109LABEL1最精准'
    # train_dataset = WeldData(data_path)
    # train_dataset.__getitem__(0)
    # print(train_dataset.__len__())
    # path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1'
    # # _, voltage_paths, current_paths, sound_paths, _ = traversal_files(path)
    # # split_voltage_current(voltage_paths)
    # # split_voltage_current(current_paths)
    # # split_voltage_current(sound_paths)
    # origin_path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1-VoltCurSoundTensors'
    new_path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1'
    img_set_flatten(new_path)
    # move_tensor_file(origin_path, new_path)
