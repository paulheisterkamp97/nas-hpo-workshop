import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Class Labels
classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# 1. Show Example Images
def plot_example_images(dataset, classes, num_images=10):
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(classes[label])
        plt.axis('off')
    plt.suptitle("FashionMNIST Example Images")
    plt.tight_layout()
    plt.show()


plot_example_images(train_dataset, classes)


# 2. Class Distribution
def plot_class_distribution(dataset, classes):
    labels = [label for _, label in dataset]
    label_counts = np.bincount(labels)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(range(len(classes))), y=label_counts, palette="viridis")
    plt.xticks(ticks=list(range(len(classes))), labels=classes, rotation=45)
    plt.title("FashionMNIST Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count Img")
    plt.show()


plot_class_distribution(train_dataset, classes)


# 3. Pixelintensitätsverteilung
def plot_pixel_distribution(dataset, num_samples=1000):
    images = np.stack([dataset[i][0].numpy().flatten() for i in range(num_samples)])
    plt.figure(figsize=(10, 6))
    sns.histplot(images.flatten(), bins=50, kde=True, color='blue')
    plt.title("Distribution of Pixel Intensities")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()


plot_pixel_distribution(train_dataset)
