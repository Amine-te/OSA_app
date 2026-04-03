import torch
from torch.utils.data import Dataset
from PIL import Image

class SubclassDataset(Dataset):
    """Custom dataset for training the CNN classifier on subclasses"""

    def __init__(self, image_paths, labels, transform=None, class_names=None):
        print(f"Initializing dataset with {len(image_paths)} images...")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or [f"class_{i}" for i in range(len(set(labels)))]

        # Quick validation - don't check every image to avoid hanging
        print("Dataset initialized successfully!")
        print(f"Total images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image and label as fallback
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label
