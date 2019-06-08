import csv
import os
from io import BytesIO

import numpy as np
import requests
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GANHandDataset(Dataset):
    def __init__(self, csv_path="./dataset_noObj.csv", transform="None"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_path (string): Directory with all the images url.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.target = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append(row[0])
                self.target.append(row[1])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        response = requests.get(self.data[idx], stream=True)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        print(image)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = torch.from_numpy(image)

        response_joints = requests.get(self.target[idx])
        joint_pos = response_joints.content.decode().split(",")
        joint_pos = [float(x) for x in joint_pos]
        joint_pos = torch.from_numpy(np.array(joint_pos))

        sample = {'image': image, 'joint_pos': joint_pos}
        
        return sample



class EgoHandDataset(Dataset):
    def __init__(self, dataset_dir="./EgoHands_dataset/", transform="None"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_path (string): Directory with all the images url.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = dataset_dir
        self.frame_count = frame_count
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):

        video = cv2.VideoCapture(os.listdir(self.dataset_path)[idx])
        imgs = []
        while(cap.isOpened()):
            _, frame = video.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        cap.release()
        cv2.destroyAllWindows()

        tensor = torch.from_numpy(np.array(imgs))

        return tensor


class FPABDataset(Dataset):
    def __init__(self, csv_path="./Subject_6.csv", transform="None"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_path (string): Directory with all the images url.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        self.target = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append([row[0], row[1]])
                self.target.append(row[2])

        self.transform = transform

    def joint_arranger(self, idx):

        joint_pos = [float(x) for x in self.target[idx].split(" ")]
target
        joint_pos = torch.from_numpy(np.array(joint_pos))

        return joint_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        rgb_img = torch.from_numpy(np.array(Image.open(self.data[idx][0])))
        depth_img = torch.from_numpy(np.array(Image.open(self.data[idx][1])))

        joint_pos = self.joint_arranger(idx)

        # sample = {'RGB_image': rgb_img, 'Depth_image': depth_img, 'joint_pos': joint_pos}
        
        return rgb_img, depth_img, joint_pos


# Testing
dataset = FPABDataset()
loader = DataLoader(dataset)

