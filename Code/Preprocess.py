import torch
import os
import glob
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Anime(Dataset):

    def __init__(self, root, resize, mode):
        # root is the directory of the dataset
        # resize the output size of the picture
        # mode is train or test or validation

        super(Anime, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)

        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train':
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.8 * len(self.images)):int(0.9 * len(self.images))]
            self.labels = self.labels[int(0.8 * len(self.labels)):int(0.9 * len(self.labels))]
        elif mode == 'display':
            self.images = self.images
            self.labels = self.labels
        else:
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, "*.png"))
                images += glob.glob(os.path.join(self.root, name, "*.jpg"))

            # 305 images
            # print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print("write into csv file: ", filename)

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x

    def __getitem__(self, index):
        # index~[0~len(images)]
        # self.images, self.labels
        img, label = self.images[index], self.labels[index]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label


def main():
    # import visdom
    # import time
    #
    # viz = visdom.Visdom()

    db = Anime('..\\Images', 64, 'train')

    x, y = next(iter(db))
    # print('sample:', x.shape, y.shape, y)

    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    # loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
    print(db.name2label)

    # for x, y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)


if __name__ == '__main__':
    main()
