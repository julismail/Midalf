import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define a simple CNN model
class SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Custom dataset class
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.img_list = self.generate_img_list()

    def generate_img_list(self):
        img_list = []
        for cls in self.classes:
            cls_folder = os.path.join(self.root, cls)
            for img_name in os.listdir(cls_folder):
                img_list.append((os.path.join(cls_folder, img_name), cls))
        return img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, cls = self.img_list[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_to_idx[cls]

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Set up datasets and dataloaders
root_folder = '/path/to/your/spectrogram/images'
dataset = SpectrogramDataset(root_folder, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
num_classes = len(dataset.classes)
model = SpectrogramClassifier(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), 'spectrogram_classifier.pth')
