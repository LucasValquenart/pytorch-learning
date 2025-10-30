# PyTorch Learning - Basics

Un projet d'apprentissage personnel pour maîtriser les fondamentaux de PyTorch et du deep learning.

## Contenu du projet

Ce repository documente mon parcours d'apprentissage de PyTorch à travers différents concepts essentiels :

### 0. Utilisation GPU

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
```


### 1. Gestion des datasets
- Création de classes personnalisées pour gérer des datasets d'images
- Implémentation de transformations et prétraitement des données
- Manipulation et chargement efficace d'images

```python
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
```

### 2. DataLoaders
- Utilisation des DataLoaders PyTorch pour un chargement optimisé des données
- Configuration du batching, shuffling
- Compréhension de l'impact des DataLoaders sur les performances

```python
dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dl_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

counter = 0
for epoch in [1,2]:
    for batch, label in dl_train:
        if counter == 0:
            print("batch label",batch, label)
            counter += 1
        print("batch.size label.size",batch.size(), label.size())
```

### 3. Réseaux de neurones
- Construction de mon premier réseau de neurones pour la classification d'images
- Architecture, couches et fonctions d'activation

```python
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
```

