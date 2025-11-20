import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, id_list_file, image_size=(224, 224)):
        self.root_dir = root_dir
        self.image_size = image_size
        with open(id_list_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines()]

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                                 std=[0.229, 0.224, 0.225])
        ])

        # Resize per la label, con interpolazione NEAREST per evitare valori intermedi
        self.label_resize = transforms.Resize(image_size, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_path = os.path.join(self.root_dir, sample_id, 'rgb.jpg')
        label_path = os.path.join(self.root_dir, sample_id, 'labels.png')

        # Carica immagine RGB e label (come ID numerici)
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)  # Non convertire in 'RGB'!

        # Applica trasformazioni
        image = self.image_transform(image)
        label = self.label_resize(label)

        # Converte in array numpy e poi tensor long (class ID)
        label = torch.from_numpy(np.array(label)).long()

        return image, label
