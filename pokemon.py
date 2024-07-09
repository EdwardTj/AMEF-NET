import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import SimpleITK as sitk
import numpy as np


class Pokemon(Dataset):

    def __init__(self, root, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print("self.name2label::::::::::",self.name2label) #{'NoPD': 0, 'PD': 1}
        # image, label
        # print(self.name2label)
        # self.images, self.labels = self.load_csv('images.csv')
        # print(len(self.images),len(self.labels))
        # print(self.images, self.labels)
        # a, b = self.get_class_counts()
        # print(a,b)
        if mode == 'train':  # 80%
            self.images,self.labels = self.load_csv('images_train.csv')
            # self.images = self.images[:int(0.8 * len(self.images))]
            # self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'val':  # 20%
            self.images, self.labels = self.load_csv('images_val.csv')
            # self.images = self.images[int(0.8 * len(self.images)):]
            # self.labels = self.labels[int(0.8 * len(self.labels)):]

    def get_class_counts(self, ):
        self.train_data = self.load_csv('images.csv')
        target = []
        # print(self.train_data[1]) #[1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
        class_counts = torch.zeros(2)  # 共有2个类别
        for targets in self.train_data[1]:
            class_counts[targets] += 1
            target.append(targets)
        # print("class_counts:",class_counts) #class_counts: tensor([4., 9.])
        return class_counts, torch.tensor(target)

    def load_csv(self, filename):
        # print("start load csv.............")
        # print("os.path.join(self.root, filename))::::",os.path.join(self.root, filename))
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                data_path = "pokemon"

                # print("self.name2label==================>",self.name2label)

                # for name in self.name2label.keys():

                images += glob.glob(os.path.join(data_path, name, '*.gz'))
                # images_img += glob.glob(os.path.join(data_path, j,'*.png'))
                # print("images:",images)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    # print("name:----",name)
                    label = self.name2label[name]
                    # print("label:",label)
                    writer.writerow([img, label])
                # print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # print("row:---",row) # ['.\\\\pokemon_MRI1\\\\PD\\10\\100012_Anon_20230109090150_5001.nii.gz', '1']
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        # print(" read from csv file over")
        return images, labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon_MRI1\\NoPD1\\2080.jpg'
        # label: 0
        print(idx)
        img, label = self.images[idx], self.labels[idx]
        # print(img,label)

        # print(img)
        img = sitk.ReadImage(img)
        img = sitk.GetArrayFromImage(img)
        # 将影像像素值归一化到[0, 1]的范围 计算图像数组中的最小值和最大值：
        # np.min(image_array): 这个表达式计算图像数组的最小值。
        # np.max(image_array): 这个表达式计算图像数组的最大值。
        # 归一化操作：
        # (image_array - np.min(image_array)): 这个表达式将图像数组中的每个像素值减去最小值，目的是使最小值归零。
        # (np.max(image_array) - np.min(image_array)): 这个表达式计算最大值和最小值的差，即动态范围。
        # (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)): 这个表达式对上述两个结果进行除法操作，得到的结果是在[0, 1]范围内的归一化值。它通过对原始像素值进行线性变换，将最小值映射到0，最大值映射到1，而其他值则按照其相对位置进行比例映射。
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = torch.from_numpy(img).float()
        img = img.resize_(182, 182, 182)
        img = img.unsqueeze(0)

        # # print("tensor_img前：",img.shape)
        # tensor_img = transforms.ToTensor()
        # img = tensor_img(img)
        # img = img.resize_(182, 182, 182)
        # # print("img_resize:",img.shape)
        # img = torch.unsqueeze(img, dim=0)
        # # img = torch.unsqueeze(img, dim=0)
        print("最终image_shape：", img.shape)
        label = torch.tensor(label)
        return img, label


def main():
    db = Pokemon('pokemon', 'train')
    print("len(db):::::::::::::", len(db))
    val = Pokemon('pokemon', 'val')
    print("len(val):::::::::::::", len(val))
    # get_class_counts= Pokemon.get_class_counts()
    x, y, = next(iter(db))
    print('sample:', x.shape, "--", y.shape, "--", y)


if __name__ == '__main__':
    main()