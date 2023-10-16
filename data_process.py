import os
import re
import torchvision.transforms as F
from PIL import Image
import shutil
def traversal_files(path):
    paths = os.walk(path)
    pic_path_list = list()
    voltage_path_list = list()
    current_path_list = list()
    sound_path_list = list()
    for path, dir_list, file_list in paths:
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if re.search(r".png", file_path):
                pic_path_list.append(file_path)
            elif re.search(r"Voltage", file_path):
                voltage_path_list.append(file_path)
            elif re.search(r"Current", file_path):
                current_path_list.append(file_path)
            elif re.search(r"Sound", file_path):
                sound_path_list.append(file_path)            
    return pic_path_list, voltage_path_list, current_path_list, sound_path_list


def resize_pic(path_list):
    for pic_path in path_list:
        source_pic = Image.open(pic_path)
        target_size = (224, 224)
        resizer = F.Resize(target_size)
        try:
            resized_pic = resizer(source_pic)
        except OSError as e:
            print(f"Error at {pic_path}: ", e)
        resized_path = re.sub("SY_DATA", "resized_img", pic_path)
        save_dir = os.path.dirname(resized_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(resized_path)
        resized_pic.save(resized_path)
        

def voltage_process(voltage_paths):
    for voltage_file in voltage_paths:
        resized_path = re.sub(r"Z:\\SY_DATA", r"C:\\Users\\yimen\\resized_img", voltage_file)
        save_dir = os.path.dirname(resized_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(resized_path)
        shutil.copy(voltage_file, save_dir)


            

if __name__ == "__main__":
    pic_path_list, voltage_path_list, current_path_list, sound_path_list = traversal_files('Z:\\SY_DATA\\20230109LABEL1最精准')
    print(current_path_list)
    voltage_process(current_path_list)