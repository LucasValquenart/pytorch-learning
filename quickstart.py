import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.io import decode_image
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


"""def download_data():
    :return: traindataset et testdataset
    X_train = datasets.FashionMNIST(root='data',
                                      train=True,
                                      download=True,
                                      transform=ToTensor())

    X_test = datasets.FashionMNIST(root='data',
                                  train=False,
                                  download=True,
                                  transform=ToTensor())

    return X_train, X_test

X_train, X_test = download_data()

#print(X_train)
#print(X_test)

batch_size = 64
X_train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
X_test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

#for X, y in X_test_loader:
#    print(f"Shape of X[N, C, H, W]: {X.shape}]")
#   print(f"Shape of y[N, C, H, W]: {y.shape}")"""

class CustomDataset(Dataset):
    def __init__(self,annotation_dir, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotation_dir, usecols=['label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        filename = str(self.img_labels.label[index]).zfill(2)+ "_"+ str(index%230+1).zfill(3)+ ".jpg"
        img_path = os.path.join(self.img_dir , filename)
        img = decode_image(img_path)
        label = self.img_labels.iloc[index].values[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

ds = CustomDataset(annotation_dir='./data/data/CFEE_au.csv', img_dir='./data/data/cfee', transform = None, target_transform = None)

indices = list(range(len(ds)))

train_val_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=0.2,
    random_state=42
)

train_dataset = Subset(ds, train_indices)
val_dataset = Subset(ds, val_indices)
test_dataset = Subset(ds, test_indices)

batch_size = 64

dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dl_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""counter = 0
for epoch in [1,2]:
    for batch, label in dl_train:
        if counter == 0:
            print("batch label",batch, label)
            counter += 1
        print("batch.size label.size",batch.size(), label.size())"""

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = Net().to(device)

for name, param in model.named_parameters():
    print(f"Layer : {name} / Size: {param.size()}/ Shape: {param[:2]}\n")
