from torch.utils.data import Dataset
from PIL import Image
from data_process import traversal_files

label_dict = {
    "正常": 0,
    "焊偏": 1,
    "卡丝": 2,
    "气孔": 3,
    "烧穿": 4,
    "未焊透": 5
}

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
        label = label_dict[img_item_path.split("\\")[3]]
        # print(label)
        return img, label

    def __len__(self):
        return len(self.img_path)

# data_path = 'G:\\resized_img\\20230109LABEL1最精准'
# train_dataset = WeldData(data_path)
# train_dataset.__getitem__(0)
# print(train_dataset.__len__())