import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import csv
from PIL import Image
import requests
from io import BytesIO

class GANHandDataset(Dataset):

    def __init__(self, csv_path="dataset_noObj.csv", transform="None"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_path (string): Directory with all the images url.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.target = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append(row[0])
                self.target.append(row[1])
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        response = requests.get(self.data[idx], stream=True)
        img = Image.open(BytesIO(response.content))
        image = torch.from_numpy(np.array(img))

        response_joints = requests.get(self.target[idx])
        joint_pos = response_joints.content.decode().split(",")

        return image, joint_pos


# Testing
# Dataset = GANHandDataset()
# print(len(Dataset.__getitem__(0)))