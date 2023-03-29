import os
import numpy as np
import re
import torch
from tqdm.auto import tqdm

from conf import *

class DataLoader:
    def __init__(self, directory: str, ratio :float = 0.2) -> None:
        self.classes2index = CLASSES2INDEX
        self.index2classes = INDEX2CLASSES
        self.classes = CLASSES
        self.dir = directory
        self.data = []
        self.BLANK = len(self.classes)
        self.ratio = ratio
        self.data_loader()

    def line2float(self, line: str):
        return [float(i) for i in line[:-1].split(" ")]
    
    def read_file(self, file:str):
        with open(file, "r") as f:
            lines = f.readlines()
        x = [[]]
        for i in range(len(lines)):
            if ";" in lines[i]:
                x.append([])
            else:
                x[-1].append(self.line2float(lines[i]))
        if x[-1] == []:
            return x[:-1]
        return x

    def file2data(self, file: str):
        X = self.read_file(file=file)
        y = file.split("/")[-1].split(".")[0] if "/" in file else file.split(".")[0]
        y = re.findall('[A-Z][^A-Z]*', y)
        for x in X:
            self.data.append([np.array(x), [self.classes2index[move] for move in y]])
    
    def data_loader(self):
        files = os.listdir(self.dir)
        for file in tqdm(files):
            if file.endswith("txt"):
                self.file2data(file=self.dir+file)
    
    def train_val_split(self, ith):
        split_size = int(self.ratio * len(self.data))
        point1 = ith * split_size
        point2 = point1 + split_size
        self.val = self.data[point1:point2]
        self.train = self.data[:point1] + self.data[point2:]
    
    def pad_x(self, x, length):
        return np.pad(x, ((0, length - x.shape[0]), (0, 0)), mode='constant', constant_values = 0.)

    def pad_y(self, y, length):
        out = [i for i in y]
        for i in range(length - len(y)):
            out.append(self.BLANK)
        return out

    def extract_batch(self, data, index, bs, device):
        last_index = min(len(data), (index + 1) * bs)
        batch = data[index * bs: last_index]
        max_x = max([d[0].shape[0] for d in batch])
        max_y = max([len(d[1]) for d in batch])
        X = []
        y = []
        target_lengths = []
        for d in batch:
            X.append(self.pad_x(d[0], max_x))
            y.append(self.pad_y(d[1], max_y))
            target_lengths.append(len(d[1]))
        y = np.array(y)
        X = np.array(X)
        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).int().to(device)
        target_lengths = torch.from_numpy(np.array(target_lengths)).int().to(device)
        return X, y, target_lengths, len(batch)
