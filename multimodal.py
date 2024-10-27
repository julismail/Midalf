import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=2):
        super(ImageClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.load_state_dict(torch.load('/home/plaiground/Documents/resnet-18/resnet18-f37072fd.pth'))
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Use identity to output features instead of classification
        self.in_features = in_features  # Save the in_features attribute
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.model(x)
        return features, self.classifier(features)

class SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectrogramClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.load_state_dict(torch.load('/home/plaiground/Documents/resnet-18/resnet18-f37072fd.pth'))
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Use identity to output features instead of classification
        self.in_features = in_features  # Save the in_features attribute
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.model(x)
        return features, self.classifier(features)

class ImageClassifierWrapper(nn.Module):
    def __init__(self, pretrained_model_path, num_classes=2):
        super(ImageClassifierWrapper, self).__init__()
        self.model = ImageClassifier(num_classes)
        self.load_pretrained_weights(pretrained_model_path)

    def load_pretrained_weights(self, pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

class SpectrogramClassifierWrapper(nn.Module):
    def __init__(self, pretrained_model_path, num_classes=2):
        super(SpectrogramClassifierWrapper, self).__init__()
        self.model = SpectrogramClassifier(num_classes)
        self.load_pretrained_weights(pretrained_model_path)

    def load_pretrained_weights(self, pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

class MultimodalLateFusion(nn.Module):
    def __init__(self, image_model_path, spectrogram_model_path):
        super(MultimodalLateFusion, self).__init__()
        self.image_model = ImageClassifierWrapper(image_model_path)
        self.spectrogram_model = SpectrogramClassifierWrapper(spectrogram_model_path)
        
        # Get the feature sizes
        image_feature_dim = self.image_model.model.in_features
        spectrogram_feature_dim = self.spectrogram_model.model.in_features
        
        combined_feature_size = image_feature_dim + spectrogram_feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, image, spectrogram):
        image_features, _ = self.image_model(image)
        spectrogram_features, _ = self.spectrogram_model(spectrogram)
        
        combined_features = torch.cat((image_features, spectrogram_features), dim=1)
        output = self.fusion_layer(combined_features)
        return output

class PairedDataset(Dataset):
    def __init__(self, image_dataset, spectrogram_dataset):
        self.image_dataset = image_dataset
        self.spectrogram_dataset = spectrogram_dataset

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        spectrogram, _ = self.spectrogram_dataset[idx]
        return image, spectrogram, label

# Define transforms for the datasets, including normalization
data_transforms = {
    'images': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'spectrograms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

image_dataset = ImageFolder(root='/home/plaiground/Documents/dataset/bodimgsplit/test/', transform=data_transforms['images'])
spectrogram_dataset = ImageFolder(root='/home/plaiground/Documents/dataset/bodimgspectrosplit/test/', transform=data_transforms['spectrograms'])

paired_dataset = PairedDataset(image_dataset, spectrogram_dataset)
test_loader = DataLoader(paired_dataset, batch_size=32, shuffle=False)

image_model_path = "/home/plaiground/Documents/multimodal/imgcnnbod.pth"
spectrogram_model_path = "/home/plaiground/Documents/multimodal/spectroCNNbodmas.pth"

fusion_model = MultimodalLateFusion(image_model_path, spectrogram_model_path)

def evaluate(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, spectrograms, labels in data_loader:
            images = images.to(device)
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            outputs = model(images, spectrograms)
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fusion_model.to(device)

accuracy = evaluate(fusion_model, test_loader)
print(f'Test Accuracy: {accuracy:.4f}')
