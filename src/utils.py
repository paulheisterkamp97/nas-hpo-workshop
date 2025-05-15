import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

# Define a transformation pipeline for the CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and std for each channel
])

# List of classes in the CIFAR-10 dataset
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_dataset(batch_size: int = 32):
    """
    Load the CIFAR-10 dataset and create PyTorch DataLoaders for training and testing.

    Args:
        batch_size (int): Number of samples per batch. Default is 32.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for training data.
            - test_loader (DataLoader): DataLoader for testing data.
    """
    # Load CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    # Load CIFAR-10 testing dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    # Create DataLoader for training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Create DataLoader for testing dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_best_torch_device():
    """
    Determine the best available device for PyTorch computations (e.g., MPS, CUDA, or CPU).

    Returns:
        torch.device: The most suitable device for computations.
    """
    if torch.backends.mps.is_available():
        # Use Metal Performance Shaders (MPS) for Apple Silicon
        device = torch.device('mps')
        print("Using device: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        # Use CUDA if an NVIDIA GPU is available
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        # Fallback to CPU if no GPU is available
        device = torch.device('cpu')
        print("Using device: CPU")
    
    return device


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, verbose: bool = False):
    """
    Evaluate a PyTorch model on the test dataset and calculate performance metrics.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        verbose (bool): Whether to print detailed metrics (precision and recall per class). Default is False.

    Returns:
        float: The overall accuracy of the model on the test dataset.
    """
    # Get the best available device
    device = get_best_torch_device()

    # Number of classes in the CIFAR-10 dataset
    num_classes = len(CLASSES)
    # Set the model to evaluation mode
    model.eval()

    # Lists to store all predictions and true labels
    all_predictions = []
    all_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over the test dataset
        for images, labels in test_loader:
            # Move data to the selected device
            images, labels = images.to(device), labels.to(device)
            # Perform forward pass through the model
            outputs = model(images)
            # Get the predicted class for each image
            _, predicted = torch.max(outputs, 1)
            # Store predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_predictions)

    # If verbose is False, return only the overall accuracy
    if not verbose:
        return overall_accuracy
    
    # Calculate precision and recall for each class
    precision_per_class = precision_score(
        all_labels, all_predictions, average=None, labels=range(num_classes)
    )
    recall_per_class = recall_score(
        all_labels, all_predictions, average=None, labels=range(num_classes)
    )

    # Print detailed metrics if verbose is True
    print(f"Overall Accuracy: {100 * overall_accuracy:.2f}%\n")
    print("Precision and Recall per Class:")
    for idx, class_name in enumerate(CLASSES):
        print(
            f"Class {idx} ({class_name}): "
            f"Precision: {precision_per_class[idx]:.2f}, "
            f"Recall: {recall_per_class[idx]:.2f}"
        )

    return overall_accuracy
