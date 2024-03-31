import pandas as pd
import torch
import torchvision.transforms.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("L")
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_mean_and_std(dataset):
        tensors = []
        for img, _ in dataset:
            if not torch.is_tensor(img):
                img = torchvision.transforms.functional.to_tensor(img)
            tensors.append(img)
        all_tensors = torch.stack(tensors)
        mean = torch.mean(all_tensors)
        std = torch.std(all_tensors)

        return mean, std


if __name__ == "__main__":
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.2484], [0.2341])
    ])

    dataset = BrainTumorDataset(csv_file='PATH_TO_CSV',
                                root_dir='PATH_TO_BRAIN_DATASET',
                                transform=image_transform)

    # mean, std = BrainTumorDataset.get_mean_and_std(dataset)
    # print(mean, std)

    def tensor_to_img(tensor):
        to_img = transforms.ToPILImage()
        img = to_img(tensor)
        return img

    img_tensor, label = dataset[0]
    img = tensor_to_img(img_tensor)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
