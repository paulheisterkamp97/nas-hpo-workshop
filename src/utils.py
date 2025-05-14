import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Load training and test datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)



def get_best_torch_device():
    """
    Checks for the best available PyTorch backend (MPS, CUDA, or CPU)
    and returns a properly configured torch device.
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Use Metal Performance Shaders (MPS) for Apple Silicon
        print("Using device: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')  # Use CUDA for NVIDIA GPUs
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')  # Fallback to CPU
        print("Using device: CPU")
    
    return device

def evaluate_model_with_class_names(model: torch.nn.Module, test_loader: DataLoader):
    device = get_best_torch_device()
    class_names = train_dataset.classes
    num_classes = len(class_names)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate precision, recall, and accuracy per class (same as before)
    
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    precision_per_class = precision_score(all_labels, all_predictions, average=None, labels=range(num_classes))
    recall_per_class = recall_score(all_labels, all_predictions, average=None, labels=range(num_classes))

    print(f"Overall Accuracy: {100 * overall_accuracy:.2f}%\n")
    print("Precision and Recall per Class:")
    for idx, class_name in enumerate(class_names):
        print(f"Class {idx} ({class_name}): Precision: {precision_per_class[idx]:.2f}, Recall: {recall_per_class[idx]:.2f}")