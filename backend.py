"""
Resources used:
- https://pytorch.org/hub/pytorch_vision_alexnet/
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
"""

## Standard libraries
import numpy as np
import random
from tqdm import tqdm

## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

# Torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


class AlexNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetMini, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def preprocess_image(input_image, device):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    return input_batch


def visualize_prediction(prediction, categories):
    top5_prob, top5_catid = torch.topk(prediction, 5)

    for i in range(top5_prob.size(0)):
        print(f"Image of a {categories[top5_catid[i]]} (probability: {round(top5_prob[i].item() * 100, 2)}%)")

    prediction_np = prediction.cpu().numpy()
    heatmap_data = prediction_np.reshape(25, 40)
    top5_prob, top5_catid = torch.topk(prediction, 5)
    top5_coords = [np.unravel_index(catid, heatmap_data.shape) for catid in top5_catid.cpu().numpy()]
    top5_labels = [categories[catid] for catid in top5_catid.cpu().numpy()]

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=False, cbar=True)
    angles = np.random.uniform(0, 2 * np.pi, len(top5_coords))
    angles = [2, 6, 5.7, 2.3, 3.5]

    for i, ((y, x), label) in enumerate(zip(top5_coords, top5_labels)):
        angle = angles[i]
        offset_distance = 3
        text_x = x + 0.5 + offset_distance * np.cos(angle)
        text_y = y + 0.5 + offset_distance * np.sin(angle)

        line_end_x = x + 0.5 + (offset_distance - 0.8) * np.cos(angle)
        line_end_y = y + 0.5 + (offset_distance - 0.8) * np.sin(angle)

        text_x = np.clip(text_x, 0, heatmap_data.shape[1] - 1)
        text_y = np.clip(text_y, 0, heatmap_data.shape[0] - 1)

        plt.text(text_x, text_y, label, color='white', ha='center', va='center', fontsize=10, fontweight='bold')
        plt.plot([x + 0.5, line_end_x], [y + 0.5, line_end_y], color='white', linewidth=1)

    plt.show()


def get_cifar10_dataloaders(visualize_samples=False):
    set_seed(43)
    DATASET_PATH = "../data"

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_set = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    if visualize_samples:
        _, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            img, label = test_set[i]
            img = img * torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            img = img.clamp(0, 1)
            axs[i].imshow(img.permute(1, 2, 0))
            axs[i].set_title(train_set.classes[label])
            axs[i].axis('off')
        plt.show()

    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader


def load_trained_alexnet(device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights='AlexNet_Weights.IMAGENET1K_V1')
    model.eval()
    model.to(device)
    print("AlexNet model loaded successfully!")

    return model


def load_untrained_alexnetmini(device):
    model = AlexNetMini()
    model.eval()
    model.to(device)
    print("Untrained AlexNet Mini model loaded successfully!")

    return model


def layer_type_to_str(layer):
    if isinstance(layer, nn.Conv2d):
        return 'conv'
    elif isinstance(layer, nn.Linear):
        return 'linear'
    elif isinstance(layer, nn.ReLU):
        return '??'
    elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
        return '??'
    elif isinstance(layer, nn.Dropout):
        return '??'
    else:
        return 'other'


def print_model_architecture(model):
    features, classifier = model.features, model.classifier

    print("======== Convolutional part ========")
    for i, layer in enumerate(features):
        print(layer_type_to_str(layer), end='')

        if i != len(features) - 1:
            print(' => ', end='')
    print()

    print("======== Classifier part ========")
    for i, layer in enumerate(classifier):
        if i != 0:
           print(' => ', end='')
        print(layer_type_to_str(layer), end='')
    print()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate_model(model, test_loader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}")


def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=0.001):
    set_seed(43)
    # We re-load an untrained model on purpose to avoid confusion,
    # where the model continues training if the cell is run multiple times.
    model = load_untrained_alexnetmini(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"Before training\n-------------------------------")
    evaluate_model(model, test_loader, device, epoch=-1)
    print("\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        evaluate_model(model, test_loader, device, epoch)
        print("\n")

    return model


def visualize_layer_weights(model, layer_index):
    set_seed(43)
    conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    layer = conv_layers[layer_index]
    weights = layer.weight.data.cpu().detach().clone()

    # Normalize weights to [-1, 1]
    weights_min = weights.min()
    weights_max = weights.max()
    normalized_weights = 2 * (weights - weights_min) / (weights_max - weights_min) - 1

    # Scale to [0, 1] range where 0 is mapped to 0.5 (gray)
    scaled_weights = (normalized_weights + 1) / 2

    num_kernels = weights.shape[0]
    num_plots = 24

    selected_indices = random.sample(range(num_kernels), num_plots)

    fig, axes = plt.subplots(3, 8, figsize=(12, 4.5))

    for i, ax in enumerate(axes.flat):
        filter = scaled_weights[selected_indices[i]]
        filter = filter.permute(1, 2, 0)
        ax.imshow(filter)
        ax.axis('off')

    plt.show()
