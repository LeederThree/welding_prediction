from torch.utils.data import Dataset, TensorDataset
from PIL import Image
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
            voltage_tensor_path = '/'.join(['20230109LABEL1-VoltCurSoundTensors', str(label), 'Voltage' + '-' + '-'.join(paths[i].split("\\")[-3:-1]) + '.pt'])
            current_tensor_path = '/'.join(['20230109LABEL1-VoltCurSoundTensors', str(label), 'Current' + '-' + '-'.join(paths[i].split("\\")[-3:-1]) + '.pt'])
            sound_tensor_path = '/'.join(['20230109LABEL1-VoltCurSoundTensors', str(label), 'Sound' + '-' + '-'.join(paths[i].split("\\")[-3:-1]) + '.pt'])
            data_path_dict[img_dir] = [voltage_tensor_path, current_tensor_path, sound_tensor_path]
    with open("pic_data_map.json", "w") as json_file:
        json.dump(data_path_dict, json_file)
    return img_tensor, label_tensor


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
        new_tensor_path = '\\'.join([new_dir, reverse_label_dict[tensor_path.split('\\')[-2]], sub_dir_list[1], sub_dir_list[2], sub_dir_list[0]+'.pt'])
        shutil.copy(tensor_path, new_tensor_path)
        print('tensor saved at: ', new_tensor_path)


def img_set_flatten(root_dir):
    img_paths, _, _, _, tensor_paths = traversal_files(root_dir)
    dir_cache = dict()
    # for img_path in img_paths:
    #     img_dir = os.path.dirname(img_path)
    #     dir_list = img_dir.split("\\")[-2:]
    #     img_file_name = os.path.basename(img_path)
    #     img_name = os.path.splitext(img_file_name)[0]
    #     if img_dir not in dir_cache:
    #         pic_number = get_pic_numbers(img_dir)
    #         start_num = min(pic_number)
    #         dir_cache[img_dir] = start_num
    #     offset = int(img_name) - dir_cache[img_dir]
    #     dir_list.append(str(offset))
    #     label = label_dict[img_path.split("\\")[-4]]
    #     new_dir = f"C:\\Users\\yimen\\resized_img\\flatten_dataset\\images\\{str(label)}\\"
    #     if not os.path.exists(new_dir):
    #         os.makedirs(new_dir)
    #     new_img_path = new_dir + '_'.join(dir_list) + '.png'
    #     shutil.copy(img_path, new_img_path)
    #     print(f"img saved at {new_img_path}")
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
        with open(voltage_file, 'r') as file:
            voltages = file.readlines()
            voltage_list = [float(line.strip()) for line in voltages]
        original_dir = voltage_dir.replace('20230109LABEL1最精准', '20230109 - 制备')
        if original_dir not in original_number_map:
            original_pics = get_pic_numbers(original_dir)
            original_number_map[original_dir] = original_pics
        else:
            original_pics = original_number_map[original_dir]
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
            print(file_name)
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
    # path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1最精准'
    # _, voltage_paths, current_paths, sound_paths = traversal_files(path)
    # split_voltage_current(voltage_paths)
    # split_voltage_current(current_paths)
    # split_voltage_current(sound_paths)
    origin_path = 'C:\\Users\\yimen\\welding_prediction\\20230109LABEL1-VoltCurSoundTensors'
    new_path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1最精准'
    img_set_flatten(new_path)
