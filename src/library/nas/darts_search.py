"""
Differentiable Architecture Search (DARTS) for Neural Architecture Search

This module implements DARTS, a gradient-based NAS approach that:
1. Relaxes the discrete architecture search space to be continuous
2. Optimizes architecture parameters using gradient descent

Reference: Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable Architecture Search.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

from .search_space import SearchSpace
from src.library.nas.utils_2 import get_dataset, train_model, evaluate_model, get_best_torch_device

class MixedOperation(nn.Module):
    """
    Mixed operation for DARTS, representing a weighted sum of candidate operations
    
    This module implements the continuous relaxation of the architecture search space
    by representing each layer as a weighted sum of all possible operations.
    """
    def __init__(self, operations, in_channels, out_channels):
        """
        Initialize the mixed operation
        
        Args:
            operations: List of (name, constructor) tuples for candidate operations
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(MixedOperation, self).__init__()
        self.ops = nn.ModuleList()
        for op_name, op_constructor in operations:
            # Create each candidate operation
            op = op_constructor(in_channels, out_channels)
            self.ops.append(op)
    
    def forward(self, x, weights):
        """
        Forward pass with architecture weights
        
        Args:
            x: Input tensor
            weights: Architecture weights for operations
            
        Returns:
            Weighted sum of operations applied to input
        """
        # Compute weighted sum of all operations
        result = 0
        for i in range(len(self.ops)):
            result = result + weights[i] * self.ops[i](x)
        return result
    
class DARTSCell(nn.Module):
    """
    Cell structure for DARTS
    
    A cell is a building block of the network that contains mixed operations.
    In the full DARTS implementation, cells would have multiple nodes and edges,
    but this simplified version uses a single mixed operation per cell.
    """
    def __init__(self, operations, in_channels, out_channels, reduction=False):
        """
        Initialize the DARTS cell
        
        Args:
            operations: List of (name, constructor) tuples for candidate operations
            in_channels: Number of input channels
            out_channels: Number of output channels
            reduction: Whether this is a reduction cell (reduces spatial dimensions)
        """
        super(DARTSCell, self).__init__()
        self.reduction = reduction
        self.n_ops = len(operations)
        
        # Create mixed operation
        self.op = MixedOperation(operations, in_channels, out_channels)
        
        # For simplicity, we're using a single mixed operation per cell
        # In the full DARTS implementation, there would be multiple nodes and edges
    
    def forward(self, x, weights):
        """
        Forward pass with architecture weights
        
        Args:
            x: Input tensor
            weights: Architecture weights for operations
            
        Returns:
            Output tensor
        """
        return self.op(x, weights)
    
class DARTSNetwork(nn.Module):
    """
    Network with DARTS cells for architecture search
    
    This network consists of a stack of DARTS cells, each containing mixed operations.
    The architecture parameters (alpha) determine the weights of operations in each cell.
    """
    def __init__(self, operations, channels, num_cells, num_classes=10):
        """
        Initialize the DARTS network
        
        Args:
            operations: List of (name, constructor) tuples for candidate operations
            channels: Initial number of channels
            num_cells: Number of cells in the network
            num_classes: Number of output classes
        """
        super(DARTSNetwork, self).__init__()
        self.operations = operations
        self.n_ops = len(operations)
        
        # Initial convolution to process input images
        self.stem = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        
        # Create cells
        self.cells = nn.ModuleList()
        in_channels = channels
        for i in range(num_cells):
            # Every third cell is a reduction cell (reduces spatial dimensions)
            reduction = (i % 3 == 2)
            cell = DARTSCell(operations, in_channels, channels, reduction=reduction)
            self.cells.append(cell)
            if reduction:
                channels *= 2  # Double channels after reduction
            in_channels = channels
        
        # Global pooling and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
        
        # Initialize architecture parameters (alpha)
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        """
        Initialize architecture parameters
        
        The alpha parameters determine the weights of operations in each cell.
        They are learned during the architecture search process.
        """
        # One set of alpha parameters for each cell, with one weight per operation
        self.alphas = nn.Parameter(torch.zeros(len(self.cells), self.n_ops))
        nn.init.normal_(self.alphas, mean=0, std=0.1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (class logits)
        """
        # Initial convolution
        x = self.stem(x)
        
        # Pass through cells
        for i, cell in enumerate(self.cells):
            # Get architecture weights for this cell (softmax over operations)
            weights = F.softmax(self.alphas[i], dim=-1)
            x = cell(x, weights)
        
        # Global pooling and classifier
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        return x
    
    def genotype(self):
        """
        Generate discrete architecture based on learned alpha parameters
        
        This converts the continuous architecture representation (alpha weights)
        into a discrete architecture by selecting the operation with the highest weight.
        
        Returns:
            List of selected operations for each cell
        """
        gene = []
        for i in range(len(self.cells)):
            # Get softmax weights for this cell
            weights = F.softmax(self.alphas[i], dim=-1)
            # Select operation with highest weight
            op_idx = weights.argmax().item()
            gene.append(self.operations[op_idx][0])  # Get operation name
        return gene

class DARTSSearch:
    """
    Differentiable Architecture Search (DARTS)
    
    DARTS is a gradient-based NAS approach that:
    1. Relaxes the discrete architecture search space to be continuous
    2. Optimizes architecture parameters using gradient descent
    3. Supports first-order (and second-order approximations - not implemented here)
    """
    def __init__(self, search_space, method='first_order', channels=16, num_cells=8,
                 epochs=50, batch_size=64, device=None, small_subset=True):
        """
        Initialize DARTS search
        
        Args:
            search_space: SearchSpace object defining the architecture search space
            method: DARTS method ('first_order' or 'second_order')
            channels: Initial number of channels
            num_cells: Number of cells in the network
            epochs: Number of epochs for search
            batch_size: Batch size for training
            device: Device to use for training (if None, use best available)
            small_subset: Whether to use a small subset of data for faster evaluation
        """
        self.search_space = search_space
        self.method = method
        self.channels = channels
        self.num_cells = num_cells
        self.epochs = epochs
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
        self.best_genotype = None
        self.best_accuracy = 0
        self.best_model = None
        self.best_architecture = None
        self.search_history = []
    
    def search(self, verbose=True, use_simple_sampling=False):
        """
        Perform architecture search using DARTS
        
        Args:
            verbose: Whether to print progress
            use_simple_sampling: Whether to use simple sampling instead of DARTS (for demonstration)
            
        Returns:
            Best architecture found and its corresponding model
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting DARTS Search ({self.method}) for {self.epochs} epochs...")
        
        # Step 1: Create network with mixed operations
        model = DARTSNetwork(
            operations=self.search_space.conv_ops,
            channels=self.channels,
            num_cells=self.num_cells
        ).to(self.device)
        
        # Step 2: Split parameters into weights and architecture parameters
        # - Weight parameters (w): Parameters of the operations
        # - Architecture parameters (alpha): Parameters that determine operation weights
        w_params = []
        alpha_params = []
        
        for name, param in model.named_parameters():
            if 'alphas' in name:
                alpha_params.append(param)
            else:
                w_params.append(param)
        
        # Step 3: Create optimizers for both sets of parameters
        w_optimizer = optim.SGD(w_params, lr=0.025, momentum=0.9, weight_decay=3e-4)
        alpha_optimizer = optim.Adam(alpha_params, lr=0.001, betas=(0.5, 0.999), weight_decay=1e-3)
        
        # Step 4: Learning rate schedulers
        w_scheduler = optim.lr_scheduler.CosineAnnealingLR(w_optimizer, T_max=self.epochs)
        
        # Step 5: Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Step 6: Training loop
        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{self.epochs}")
                train_iterator = tqdm(self.train_loader)
            else:
                train_iterator = self.train_loader
            
            for step, (inputs, targets) in enumerate(train_iterator):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # First-order approximation: Update architecture then weights
            
                # Update architecture parameters
                alpha_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                alpha_optimizer.step()
                
                # Update weights
                w_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                w_optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if verbose:
                    train_iterator.set_description(
                        f"Train Loss: {train_loss/(step+1):.3f} | Train Acc: {100.*correct/total:.2f}%"
                    )
            
            # Update learning rate
            w_scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for step, (inputs, targets) in enumerate(self.val_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * correct / total
            
            # Store search history
            self.search_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(self.train_loader),
                'val_loss': val_loss / len(self.val_loader),
                'val_acc': val_acc,
                'genotype': model.genotype()
            })
            
            if verbose:
                print(f"Validation Loss: {val_loss/len(self.val_loader):.3f} | Validation Acc: {val_acc:.2f}%")
                print(f"Genotype: {model.genotype()}")
        
        # Step 7: Get final genotype
        genotype = model.genotype()
        self.best_genotype = genotype
        
        # Step 8: Convert genotype to architecture
        architecture = self._genotype_to_architecture(genotype)
        self.best_architecture = architecture
        
        # Step 9: Build and evaluate final model
        final_model = self.search_space.build_model_from_architecture(architecture)
        final_model = final_model.to(self.device)
        
        # Step 10: Train final model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001)
        
        if verbose:
            print("\nTraining final model...")
        
        train_model(
            model=final_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            num_epochs=20,
            verbose=verbose
        )
        
        # Step 11: Evaluate final model
        eval_results = evaluate_model(
            model=final_model,
            test_loader=self.test_loader,
            device=self.device,
            verbose=verbose
        )
        
        self.best_accuracy = eval_results['accuracy']
        self.best_model = final_model
        
        end_time = time.time()
        search_time = (end_time - start_time) / 60  # in minutes
        
        if verbose:
            print(f"DARTS Search completed in {search_time:.2f} minutes")
            print(f"Best architecture accuracy: {self.best_accuracy:.2f}%")
        
        return self.best_architecture, self.best_model
    
    def _genotype_to_architecture(self, genotype):
        """
        Convert genotype to architecture
        
        This converts the genotype (list of operation names) to a full architecture
        that can be used to build a model.
        
        Args:
            genotype: List of operation names
            
        Returns:
            Architecture dictionary
        """
        # Create architecture
        architecture = {'conv_layers': [], 'fc_layers': []}
        
        # Add convolutional layers
        in_channels = 3
        channels = 64
        
        for i, op_name in enumerate(genotype):
            # Find operation in search space
            op_idx = None
            for j, (name, _) in enumerate(self.search_space.conv_ops):
                if name == op_name:
                    op_idx = j
                    break
            
            # Determine pooling
            use_pooling = (i % 2 == 1)  # Every other layer has pooling
            pool_name = 'max' if use_pooling else None
            
            # Add layer
            architecture['conv_layers'].append({
                'op': op_name,
                'in_channels': in_channels,
                'out_channels': channels,
                'activation': 'relu',
                'pooling': pool_name,
                'use_pooling': use_pooling
            })
            
            in_channels = channels
        
        # Calculate feature size after convolutions and pooling
        feature_size = 32
        num_pooling = sum(1 for layer in architecture['conv_layers'] if layer['use_pooling'])
        feature_size = feature_size // (2 ** num_pooling)
        
        # Calculate flattened feature dimension
        flattened_dim = in_channels * feature_size * feature_size
        
        # Add FC layers
        architecture['fc_layers'] = [
            {
                'in_features': flattened_dim,
                'out_features': 256,
                'activation': 'relu',
                'dropout_rate': 0.5
            },
            {
                'in_features': 256,
                'out_features': 10,
                'activation': None,
                'dropout_rate': 0.0
            }
        ]
        
        return architecture
    
    def plot_search_history(self):
        """
        Plot the search history
        
        This visualizes the training and validation metrics during the search process.
        """
        if not self.search_history:
            print("No search history to plot. Run search() first.")
            return
        
        # Extract metrics
        epochs = []
        train_loss = []
        val_loss = []
        val_acc = []
        
        for h in self.search_history:
            epochs.append(h['epoch'])
            train_loss.append(h['train_loss'])
            val_loss.append(h['val_loss'])
            val_acc.append(h['val_acc'])
        
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'DARTS ({self.method}) - Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.axhline(y=self.best_accuracy, color='r', linestyle='--', 
                   label=f'Final Accuracy: {self.best_accuracy:.2f}%')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'DARTS ({self.method}) - Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_alphas(self, epoch=-1):
        """
        Visualize the architecture parameters (alpha)
        
        This shows the weights of different operations in each cell.
        
        Args:
            epoch: Epoch to visualize (-1 for the last epoch)
        """
        if not self.search_history:
            print("No search history to visualize. Run search() first.")
            return
        
        if epoch == -1:
            epoch = len(self.search_history) - 1
        elif epoch >= len(self.search_history):
            print(f"Invalid epoch {epoch}. Only {len(self.search_history)} epochs available.")
            return
        
        # Get genotype for the specified epoch
        genotype = self.search_history[epoch]['genotype']
        
        # Get operation names
        op_names = [op[0] for op in self.search_space.conv_ops]
        
        # Create a matrix of 0s and 1s indicating selected operations
        alpha_matrix = np.zeros((len(genotype), len(op_names)))
        for i, op in enumerate(genotype):
            op_idx = op_names.index(op)
            alpha_matrix[i, op_idx] = 1
        
        plt.figure(figsize=(12, 8))
        plt.imshow(alpha_matrix, cmap='Blues')
        plt.xlabel('Operations')
        plt.ylabel('Cells')
        plt.title(f'DARTS Architecture Parameters (Epoch {epoch+1})')
        plt.xticks(np.arange(len(op_names)), op_names, rotation=45, ha='right')
        plt.yticks(np.arange(len(genotype)), [f'Cell {i+1}' for i in range(len(genotype))])
        plt.colorbar(label='Selected (1) / Not Selected (0)')
        plt.tight_layout()
        plt.show()
