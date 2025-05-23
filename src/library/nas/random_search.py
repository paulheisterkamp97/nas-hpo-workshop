"""
Random Search for Neural Architecture Search (NAS)

This module implements an efficient random search algorithm for NAS,
which samples random architectures from the search space and evaluates them.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from .search_space import SearchSpace
from src.library.nas.utils_2 import get_dataset, train_model, evaluate_model, get_best_torch_device

class RandomSearch:
    def __init__(self, search_space, num_samples=50, epochs_per_model=5, batch_size=64, 
                 device=None, small_subset=True):
        """
        Initialize the Random Search
        
        Args:
            search_space: SearchSpace object defining the architecture search space
            num_samples: Number of random architectures to sample and evaluate
            epochs_per_model: Number of epochs to train each sampled model
            batch_size: Batch size for training
            device: Device to use for training (if None, use best available)
            small_subset: Whether to use a small subset of data for faster evaluation
        """
        self.search_space = search_space
        self.num_samples = num_samples
        self.epochs_per_model = epochs_per_model
        self.batch_size = batch_size
        self.small_subset = small_subset
        
        # Automatically select the best available device (GPU/CPU)
        if device is None:
            self.device = get_best_torch_device()
        else:
            self.device = device
            
        # Load datasets for training and evaluation
        self.train_loader, self.val_loader, self.test_loader = get_dataset(
            batch_size=batch_size, small_subset=small_subset
        )
        
        # Storage for search results
        self.results = []
        self.best_architecture = None
        self.best_accuracy = 0
        self.best_model = None
        
    def search(self, verbose=True, use_simple_sampling=False):
        """
        Perform random search by sampling and evaluating architectures
        
        Args:
            verbose: Whether to print progress
            use_simple_sampling: Whether to use the simpler sampling method
            
        Returns:
            Best architecture found and its corresponding model
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting Random Search with {self.num_samples} samples...")
            iterator = tqdm(range(self.num_samples))
        else:
            iterator = range(self.num_samples)
        
        # Sample and evaluate random architectures
        for i in iterator:
            # Create architectures with varying numbers of layers and structures
            architecture = self.search_space.sample_random_architecture()

            # Step 2: Build a PyTorch model from the architecture
            model = self.search_space.build_model_from_architecture(architecture)
            
            # Step 3: Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Step 4: Train the model for a epochs
            history = train_model(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                num_epochs=self.epochs_per_model,
                verbose=False
            )
            
            # Step 5: Evaluate the model on test data
            eval_results = evaluate_model(
                model=model,
                test_loader=self.test_loader,
                device=self.device,
                verbose=False
            )
            
            # Step 6: Store results
            result = {
                'architecture': architecture,
                'accuracy': eval_results['accuracy'],
                'ms_per_image': eval_results['ms_per_image'],
                'val_accuracy': history['best_val_acc'],
                'model': model
            }
            
            self.results.append(result)
            
            # Step 7: Update best architecture if needed
            if result['accuracy'] > self.best_accuracy:
                self.best_architecture = architecture
                self.best_accuracy = result['accuracy']
                self.best_model = model
                
                if verbose:
                    tqdm.write(f"New best architecture found! Accuracy: {self.best_accuracy:.2f}%")

        
        end_time = time.time()
        search_time = (end_time - start_time) / 60  # in minutes
        
        if verbose:
            print(f"Random Search completed in {search_time:.2f} minutes")
            print(f"Best architecture accuracy: {self.best_accuracy:.2f}%")
        
        return self.best_architecture, self.best_model
    
    def plot_search_results(self):
        """
        Plot the results of the random search, showing distributions of
        accuracy and inference time, as well as their relationship
        """
        # Extract accuracies and inference times
        accuracies = []
        inference_times = []
        
        for result in self.results:
            accuracies.append(result['accuracy'])
            inference_times.append(result['ms_per_image'])
        
        # plt.figure(figsize=(12, 5))
        
        # # Plot accuracy distribution
        # plt.subplot(1, 2, 1)
        # plt.hist(accuracies, bins=10, alpha=0.7)
        # plt.axvline(self.best_accuracy, color='r', linestyle='--', 
        #            label=f'Best: {self.best_accuracy:.2f}%')
        # plt.xlabel('Accuracy (%)')
        # plt.ylabel('Count')
        # plt.title('Distribution of Model Accuracies')
        # plt.legend()
        
        # # Plot inference time distribution
        # plt.subplot(1, 2, 2)
        # plt.hist(inference_times, bins=10, alpha=0.7)
        
        # Find the inference time of the best model
        best_time = None
        for r in self.results:
            if r['accuracy'] == self.best_accuracy:
                best_time = r['ms_per_image']
                break
                
        # plt.axvline(best_time, color='r', linestyle='--', 
        #            label=f'Best model: {best_time:.2f} ms')
        # plt.xlabel('Inference Time (ms/image)')
        # plt.ylabel('Count')
        # plt.title('Distribution of Inference Times')
        # plt.legend()
        
        # plt.tight_layout()
        # plt.show()
        
        # Plot accuracy vs. inference time scatter plot (Pareto front visualization)
        plt.figure(figsize=(10, 6))
        plt.scatter(inference_times, accuracies, alpha=0.7)
        plt.scatter([best_time], [self.best_accuracy], color='r', s=100, 
                   label='Best model')
        plt.xlabel('Inference Time (ms/image)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy vs. Inference Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
    
    def get_best_architecture_summary(self):
        """
        Summary of the best architecture found
        """
        arch = self.best_architecture

        summary = "Best Architecture Summary:\n"
        summary += f"Accuracy: {self.best_accuracy:.2f}%\n\n"

        summary += "Convolutional Layers:\n"
        layer_num = 1
        for layer in arch['conv_layers']:
            summary += f"Layer {layer_num}:\n"
            summary += f"  Operation: {layer['op']}\n"
            summary += f"  Channels: {layer['in_channels']} -> {layer['out_channels']}\n"
            summary += f"  Activation: {layer['activation']}\n"
            summary += f"  Pooling: {layer['pooling'] if layer['use_pooling'] else 'None'}\n"
            layer_num += 1

        summary += "\nFully Connected Layers:\n"
        layer_num = 1
        for layer in arch['fc_layers']:
            summary += f"Layer {layer_num}:\n"
            summary += f"  Size: {layer['in_features']} -> {layer['out_features']}\n"
            summary += f"  Activation: {layer['activation'] if layer['activation'] else 'None'}\n"
            summary += f"  Dropout: {layer['dropout_rate']}\n"
            layer_num += 1

        return summary
