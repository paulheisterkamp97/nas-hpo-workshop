"""
Utility functions for Neural Architecture Search (NAS)

This module provides utility functions for dataset loading, model training,
evaluation, and visualization of results.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import random

# Define a transformation pipeline for the CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and std for each channel
])

# List of classes in the CIFAR-10 dataset
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

def get_dataset(batch_size=64, num_workers=2, val_split=0.1, small_subset=False):
    """
    Load CIFAR-10 dataset and create data loaders
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation
        small_subset: If True, use a small subset of the data for quick testing
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create a small subset for quick testing if requested
    if small_subset:
        # For NAS, we use a small subset to speed up the search process
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)
        train_indices = train_indices[:1000]  # Use only 1000 training examples
        train_dataset = Subset(train_dataset, train_indices)
        
        test_indices = list(range(len(test_dataset)))
        random.shuffle(test_indices)
        test_indices = test_indices[:200]  # Use only 200 test examples
        test_dataset = Subset(test_dataset, test_indices)
    
    # Split training data into training and validation sets
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=5, early_stopping=True, patience=3, verbose=True):
    """
    Train a model and return training history
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        num_epochs: Number of epochs to train for
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0,
        'best_epoch': 0
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass, backward pass, optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {epoch_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history

def evaluate_model(model, test_loader, device=None, verbose=True):
    """
    Evaluate a model on the test set
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to use for evaluation (if None, use the model's current device)
        verbose: Whether to print results
        
    Returns:
        Dictionary containing evaluation metrics (accuracy, inference time)
    """
    if device is None:
        # Use the device that the model is currently on
        param_device = next(model.parameters()).device
        device = param_device
    
    model.to(device)
    model.eval()
    
    # Lists to store all predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate metrics
    accuracy = 100 * accuracy_score(all_labels, all_predictions)
    
    # Calculate per-image inference time
    total = len(all_predictions)
    images_per_second = total / inference_time
    ms_per_image = 1000 / images_per_second
    
    if verbose:
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Total inference time: {inference_time:.4f} seconds")
        print(f"Images per second: {images_per_second:.2f}")
        print(f"Milliseconds per image: {ms_per_image:.2f}")
        
        # Calculate precision and recall for each class if verbose
        num_classes = len(CLASSES)
        precision_per_class = precision_score(
            all_labels, all_predictions, average=None, labels=range(num_classes), zero_division=0
        )
        recall_per_class = recall_score(
            all_labels, all_predictions, average=None, labels=range(num_classes), zero_division=0
        )

        print("\nPrecision and Recall per Class:")
        for idx, class_name in enumerate(CLASSES):
            print(
                f"Class {idx} ({class_name}): "
                f"Precision: {precision_per_class[idx]:.2f}, "
                f"Recall: {recall_per_class[idx]:.2f}"
            )
    
    return {
        'accuracy': accuracy,
        'total_inference_time': inference_time,
        'images_per_second': images_per_second,
        'ms_per_image': ms_per_image
    }

def plot_comparison_results(results, metric='accuracy'):
    """
    Plot comparison of different NAS strategies
    
    Args:
        results: Dictionary mapping strategy names to metric values
        metric: Metric to plot ('accuracy', 'ms_per_image', etc.)
    """
    strategies = list(results.keys())
    values = []
    for strategy in strategies:
        values.append(results[strategy][metric])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, values)
    
    # Add value labels on top of bars
    for i in range(len(bars)):
        bar = bars[i]
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel('NAS Strategy')
    
    if metric == 'accuracy':
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of NAS Strategies - Accuracy')
    elif metric == 'ms_per_image':
        plt.ylabel('Inference Time (ms/image)')
        plt.title('Comparison of NAS Strategies - Inference Time')
    elif metric == 'search_time':
        plt.ylabel('Search Time (minutes)')
        plt.title('Comparison of NAS Strategies - Search Time')
    else:
        plt.ylabel(metric)
        plt.title(f'Comparison of NAS Strategies - {metric}')
    
    plt.tight_layout()
    plt.show()

def plot_pareto_front(results, x_metric='ms_per_image', y_metric='accuracy'):
    """
    Plot Pareto front of different NAS strategies
    
    Args:
        results: Dictionary mapping strategy names to metric dictionaries
        x_metric: Metric for x-axis (typically inference time)
        y_metric: Metric for y-axis (typically accuracy)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract x and y values for each strategy
    x_values = []
    y_values = []
    labels = []
    
    for strategy, metrics in results.items():
        x_values.append(metrics[x_metric])
        y_values.append(metrics[y_metric])
        labels.append(strategy)
    
    # Plot scatter points
    plt.scatter(x_values, y_values, s=100)
    
    # Add labels to points
    for i in range(len(labels)):
        plt.annotate(labels[i], (x_values[i], y_values[i]), 
                     xytext=(5, 5), textcoords='offset points')
    
    # Set axis labels and title
    if x_metric == 'ms_per_image':
        plt.xlabel('Inference Time (ms/image)')
    else:
        plt.xlabel(x_metric)
        
    if y_metric == 'accuracy':
        plt.ylabel('Accuracy (%)')
    else:
        plt.ylabel(y_metric)
    
    plt.title(f'Pareto Front: {y_metric} vs {x_metric}')
    
    # If plotting accuracy vs inference time, we want high accuracy and low inference time
    if x_metric == 'ms_per_image' and y_metric == 'accuracy':
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Draw Pareto front
        points = np.array(list(zip(x_values, y_values)))
        pareto_points = []
        
        # Find non-dominated points (Pareto front)
        for i in range(len(points)):
            point = points[i]
            dominated = False
            for j in range(len(points)):
                other_point = points[j]
                # For accuracy, higher is better; for inference time, lower is better
                if (other_point[0] <= point[0] and other_point[1] >= point[1] and 
                    (other_point[0] < point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(point)
        
        # Sort Pareto points by x-value
        pareto_points.sort(key=lambda p: p[0])
        
        if pareto_points:
            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]
            plt.plot(pareto_x, pareto_y, 'r--', label='Pareto Front')
            plt.legend()
    
    plt.tight_layout()
    plt.show()
