from torch.utils.data import Dataset, TensorDataset
from torch.nn.functional import interpolate
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


def tensor_preprocess(root_path, target_width, process_pic=False):
    tensor_files = traversal_files(root_path)[-1]
    for tensor_file in tensor_files:
        tensor = torch.load(tensor_file)
        width = tensor.shape[1]
        if target_width > width >= target_width * 0.8:
            tensor = tensor_interpolate(tensor, target_width)
            torch.save(tensor, tensor_file)
        elif target_width < width <= target_width * 1.2:
            tensor = tensor[:, :target_width]
            torch.save(tensor, tensor_file)
        elif width < target_width * 0.8:
            if process_pic:
                remove_pic(tensor_file)
            os.remove(tensor_file)
        elif width > target_width * 1.2:
            if process_pic:
                tensor_flatten(tensor_file)
            tensor_index = os.path.basename(tensor_file).rstrip('.pt')
            target_col = len(list(pic_traversal(get_pic_dir(tensor_file), index=tensor_index)))
            tensor = tensor.view(-1)[:target_col*target_width].view(target_col, target_width)
            torch.save(tensor, tensor_file)


def tensor_flatten(tensor_path):
    tensor = torch.load(tensor_path)
    pic_current = tensor.shape[0]
    pic_target = tensor.shape[0] * tensor.shape[1] // 75
    stride = pic_target // pic_current
    tail = pic_target % pic_current
    pic_dir = get_pic_dir(tensor_path)
    tensor_index = os.path.basename(tensor_path).rstrip('.pt')
    pic_list = list(pic_traversal(pic_dir, index=tensor_index))
    pic_list = sorted(pic_list, key=lambda x: int(os.path.splitext(x)[0].rsplit('_', 1)[1]))
    for pic_path in pic_list:
        iter_stride = stride if tail else stride - 1
        pic_copy(pic_path, iter_stride)
        if tail:
            tail -= 1
    new_pic_series = list(pic_traversal(pic_dir, index=tensor_index))
    new_pic_series = sorted(new_pic_series, key=lambda x: tuple(map(int, os.path.splitext(x)[0].rsplit('_', 2)[1:])))
    for i in range(len(new_pic_series)):
        base_name = os.path.splitext(new_pic_series[i])[0].rsplit('_', 2)[0]
        new_name = f'{base_name}_{i}.png'
        os.rename(new_pic_series[i], new_name)


def pic_copy(pic_path, stride):
    origin_name, _ = os.path.splitext(pic_path)
    count = stride
    new_name = f"{origin_name}_0.png"
    os.rename(pic_path, new_name)
    for i in range(count):
        copy_path = f"{origin_name}_{str(i+1)}.png"
        shutil.copy(new_name, copy_path)


def tensor_interpolate(tensor, target_width):
    origin_tensor = tensor.view(-1)
    interpolated_tensor = interpolate(origin_tensor.unsqueeze(0).unsqueeze(0), size=tensor.shape[0]*target_width, mode='linear')
    return interpolated_tensor.view(tensor.shape[0], target_width)


def pic_traversal(pic_dir, index=None):
    for path, _, pic_files in os.walk(pic_dir):
        for pic_file in pic_files:
            pic_path = os.path.join(path, pic_file)
            if index:
                if re.search(f"{index}", pic_path):
                    yield pic_path
            else:
                yield pic_path


def get_pic_dir(tensor_path):
    dir_list = tensor_path.split('\\')
    tensor_label = dir_list[-2]
    base_dir = '\\'.join(dir_list[:-4])
    pic_dir = f"{base_dir}\\images\\{tensor_label}"
    return pic_dir


def remove_pic(tensor_path):
    pic_dir = get_pic_dir(tensor_path)
    tensor_index = os.path.basename(tensor_path).rstrip('.pt')
    pic_iter = pic_traversal(pic_dir, tensor_index)
    for pic_path in pic_iter:
        os.remove(pic_path)


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


if __name__ == '__main__':
    # data_path = 'G:\\resized_img\\20230109LABEL1最精准'
    # train_dataset = WeldData(data_path)
    # train_dataset.__getitem__(0)
    # print(train_dataset.__len__())
    path = 'C:\\Users\\yimen\\resized_img\\flatten_dataset\\tensors\\Current'
    # _, voltage_paths, current_paths, sound_paths, _ = traversal_files(path)
    # split_voltage_current(voltage_paths)
    # split_voltage_current(current_paths)
    # split_voltage_current(sound_paths)
    # origin_path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1-VoltCurSoundTensors'
    new_path = 'C:\\Users\\yimen\\resized_img\\20230109LABEL1'
    # img_set_flatten(new_path)
    # move_tensor_file(origin_path, new_path)
    # tensor_files = traversal_files(path)[-1]
    # tensor_width = dict()
    # tensor_origin = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    # for tensor_file in tensor_files:
    #     tensor = torch.load(tensor_file)
    #     label = tensor_file.split('\\')[-2]
    #     tensor_count = tensor.shape[1]
    #     tensor_origin[label] += tensor.shape[0]
    #     if 80 > tensor_count > 60:
    #         print(tensor_file, tensor.shape[1])
    #         if label not in tensor_width.keys():
    #             tensor_width[label] = tensor.shape[0]
    #         else:
    #             tensor_width[label] += tensor.shape[0]
    # print(tensor_width)
    # print(tensor_origin)
    root_path = 'C:\\Users\\yimen\\resized_img\\flatten_dateset\\tensors\\'
    # tensor_preprocess(root_path+'Current', target_width=75, process_pic=True)
    # tensor_preprocess(root_path+'Voltage', target_width=75, process_pic=False)
    # tensor_preprocess(root_path+'Sound', target_width=750, process_pic=False)
