"""
Search Space Definition for Neural Architecture Search (NAS)

This module defines the search space for neural network architectures,
including operations, channel sizes, activation functions, and more.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class SearchSpace:
    """
    Defines the search space for CNN architectures in Neural Architecture Search.
    
    This class provides methods to:
    1. Sample random architectures from the search space
    2. Build PyTorch models from architecture specifications
    3. Mutate existing architectures (for evolutionary search)
    4. Perform crossover between architectures (for evolutionary search)
    """
    def __init__(self):
        # Available convolutional operations
        self.conv_ops = [
            ('conv3x3', self._create_conv3x3),
            ('conv5x5', self._create_conv5x5),
            ('max_pool3x3', self._create_max_pool3x3),
            ('avg_pool3x3', self._create_avg_pool3x3),
        ]
        
        # Available channel sizes
        self.channel_choices = [16, 32, 64, 128]
        
        # Range for number of layers
        self.num_layers_range = (2, 6)  # Min and max number of layers
        
        # Available activation functions
        self.activation_ops = [
            ('relu', nn.ReLU()),
            ('leaky_relu', nn.LeakyReLU(0.1)),
            ('elu', nn.ELU()),
            ('selu', nn.SELU()),
        ]
        
        # Available pooling operations
        self.pooling_ops = [
            ('max_pool2x2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('avg_pool2x2', nn.AvgPool2d(kernel_size=2, stride=2)),
        ]
        
        # Options for fully connected layers
        self.fc_sizes = [128, 256, 512, 1024]
        self.dropout_rates = [0.0, 0.2, 0.3, 0.5]

    def _create_conv3x3(self, C_in, C_out):
        """Create a 3x3 convolutional layer"""
        return nn.Conv2d(C_in, C_out, 3, padding=1, bias=False)
    
    def _create_conv5x5(self, C_in, C_out):
        """Create a 5x5 convolutional layer"""
        return nn.Conv2d(C_in, C_out, 5, padding=2, bias=False)
    
    def _create_conv7x7(self, C_in, C_out):
        """Create a 7x7 convolutional layer"""
        return nn.Conv2d(C_in, C_out, 7, padding=3, bias=False)
    
    def _create_max_pool3x3(self, C_in, C_out):
        """Create a max pooling layer followed by 1x1 convolution"""
        return nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(C_in, C_out, 1, bias=False)
        )
    
    def _create_avg_pool3x3(self, C_in, C_out):
        """Create an average pooling layer followed by 1x1 convolution"""
        return nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(C_in, C_out, 1, bias=False)
        )

    # Simple sampling method
    def sample_simple_architecture(self):
        """
        Creates a standard structure with 3 convolutional layers and 1 fully connected layer,
        but randomly selects the operations and channels.
        
        Returns:
            dict: Simple architecture specification
        """
        # Create a simple architecture with 3 conv layers and 1 FC layer
        architecture = {
            'conv_layers': [],
            'fc_layers': [],
            'input_channels': 3,  # RGB images
            'num_classes': 10,    # CIFAR-10 has 10 classes
        }
        
        # Fixed structure: 3 conv layers with pooling after each
        # This creates a standard CNN pattern: Conv->Pool->Conv->Pool->Conv->Pool
        in_channels = 3  # Start with RGB input
        for i in range(3):
            # Randomly select operation, channels, and activation
            op_index = np.random.randint(0, len(self.conv_ops))
            op_name = self.conv_ops[op_index][0]
            
            # Use progressively larger channel sizes for each layer
            out_channels = self.channel_choices[min(i, len(self.channel_choices)-1)]
            
            act_index = np.random.randint(0, len(self.activation_ops))
            act_name = self.activation_ops[act_index][0]
            
            # Always use pooling after each conv layer to reduce spatial dimensions
            pool_index = np.random.randint(0, len(self.pooling_ops))
            pool_name = self.pooling_ops[pool_index][0]
            
            # Add layer to architecture
            architecture['conv_layers'].append({
                'op': op_name,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'activation': act_name,
                'pooling': pool_name,
                'use_pooling': True
            })
            
            in_channels = out_channels  # Update input channels for next layer
        
        # Calculate feature size (32x32 input, divided by 2 for each pooling)
        feature_size = 32 // (2**3)  # 3 pooling layers = divide by 2^3 = 8
        flattened_dim = in_channels * feature_size * feature_size
        
        # Add one FC layer that goes directly to output classes
        architecture['fc_layers'].append({
            'in_features': flattened_dim,
            'out_features': 10,  # Output classes (CIFAR-10)
            'activation': None,  # No activation for final layer
            'dropout_rate': 0.5  # Fixed dropout rate
        })
        
        return architecture

    def sample_random_architecture(self):
        """
        Sample a random architecture from the search space
        
        Returns:
            dict: Architecture specification with conv_layers and fc_layers
        """
        # Sample number of convolutional layers
        num_layers = np.random.randint(self.num_layers_range[0], self.num_layers_range[1] + 1)
        
        # Initialize architecture specification
        architecture = {
            'conv_layers': [],
            'fc_layers': [],
            'input_channels': 3,  # RGB images
            'num_classes': 10,    # CIFAR-10 has 10 classes
        }
        
        # Sample convolutional layers
        in_channels = 3  # Start with RGB input
        for i in range(num_layers):
            # Sample operation type, channels, activation
            op_index = np.random.randint(0, len(self.conv_ops))
            op_name = self.conv_ops[op_index][0]
            
            channel_index = np.random.randint(0, len(self.channel_choices))
            out_channels = self.channel_choices[channel_index]
            
            act_index = np.random.randint(0, len(self.activation_ops))
            act_name = self.activation_ops[act_index][0]
            
            # Every second layer, add a pooling operation
            if i % 2 == 1:
                pool_index = np.random.randint(0, len(self.pooling_ops))
                pool_name = self.pooling_ops[pool_index][0]
                use_pooling = True
            else:
                pool_name = None
                use_pooling = False
            
            # Add layer to architecture
            architecture['conv_layers'].append({
                'op': op_name,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'activation': act_name,
                'pooling': pool_name,
                'use_pooling': use_pooling
            })
            
            in_channels = out_channels  # Update input channels for next layer
        
        # Calculate feature size after convolutions and pooling
        # Assuming input is 32x32 and each pooling reduces size by half
        feature_size = 32
        num_pooling = 0
        for layer in architecture['conv_layers']:
            if layer['use_pooling']:
                num_pooling += 1
        feature_size = feature_size // (2 ** num_pooling)
        
        # Calculate flattened feature dimension
        flattened_dim = in_channels * feature_size * feature_size
        
        # Sample fully connected layers (1 or 2)
        num_fc_layers = np.random.randint(1, 3)
        
        # First FC layer
        fc_index = np.random.randint(0, len(self.fc_sizes))
        fc_size = self.fc_sizes[fc_index]
        
        dropout_index = np.random.randint(0, len(self.dropout_rates))
        dropout_rate = self.dropout_rates[dropout_index]
        
        act_index = np.random.randint(0, len(self.activation_ops))
        act_name = self.activation_ops[act_index][0]
        
        architecture['fc_layers'].append({
            'in_features': flattened_dim,
            'out_features': fc_size,
            'activation': act_name,
            'dropout_rate': dropout_rate
        })
        
        # Second FC layer (if sampled)
        if num_fc_layers > 1:
            dropout_index = np.random.randint(0, len(self.dropout_rates))
            dropout_rate = self.dropout_rates[dropout_index]
            
            architecture['fc_layers'].append({
                'in_features': fc_size,
                'out_features': 10,  # Output classes
                'activation': None,  # No activation for final layer (will use softmax in loss)
                'dropout_rate': 0.0  # No dropout for final layer
            })
        else:
            # If only one FC layer, it goes directly to output
            architecture['fc_layers'][0]['out_features'] = 10
            architecture['fc_layers'][0]['activation'] = None
        
        return architecture

    def build_model_from_architecture(self, architecture):
        """
        Build a PyTorch model from an architecture specification
        
        Args:
            architecture (dict): Architecture specification
            
        Returns:
            nn.Sequential: PyTorch model built from the architecture
        """
        layers = []
        
        # Build convolutional layers
        for i, layer_spec in enumerate(architecture['conv_layers']):
            # Get operation constructor
            op_name = layer_spec['op']
            op_constructor = None
            for name, constructor in self.conv_ops:
                if name == op_name:
                    op_constructor = constructor
                    break
            
            # Create convolutional layer
            conv_layer = op_constructor(layer_spec['in_channels'], layer_spec['out_channels'])
            layers.append((f'conv_{i}', conv_layer))
            
            # Add activation
            if layer_spec['activation'] is not None:
                act_name = layer_spec['activation']
                
                # SIMPLIFIED: Use loop instead of dict conversion
                act_layer = None
                for name, layer in self.activation_ops:
                    if name == act_name:
                        act_layer = layer
                        break
                
                layers.append((f'act_{i}', act_layer))
            
            # Add pooling if specified
            if layer_spec['use_pooling'] and layer_spec['pooling'] is not None:
                pool_name = layer_spec['pooling']
                
                # SIMPLIFIED: Use loop instead of dict conversion
                pool_layer = None
                for name, layer in self.pooling_ops:
                    if name == pool_name:
                        pool_layer = layer
                        break
                
                layers.append((f'pool_{i}', pool_layer))
        
        # Add flatten layer
        layers.append(('flatten', nn.Flatten()))
        
        # Build fully connected layers
        for i, layer_spec in enumerate(architecture['fc_layers']):
            # Create linear layer
            fc_layer = nn.Linear(layer_spec['in_features'], layer_spec['out_features'])
            layers.append((f'fc_{i}', fc_layer))
            
            # Add activation if specified
            if layer_spec['activation'] is not None:
                act_name = layer_spec['activation']
                
                # SIMPLIFIED: Use loop instead of dict conversion
                act_layer = None
                for name, layer in self.activation_ops:
                    if name == act_name:
                        act_layer = layer
                        break
                
                layers.append((f'fc_act_{i}', act_layer))
            
            # Add dropout if rate > 0
            if layer_spec['dropout_rate'] > 0:
                dropout_layer = nn.Dropout(layer_spec['dropout_rate'])
                layers.append((f'dropout_{i}', dropout_layer))
        
        # Create sequential model
        model = nn.Sequential(OrderedDict(layers))
        return model

    def mutate_architecture(self, architecture, mutation_rate=0.2):
        """
        Mutate an existing architecture (for evolutionary search)
        
        Args:
            architecture (dict): Architecture to mutate
            mutation_rate (float): Probability of each component being mutated
            
        Returns:
            dict: Mutated architecture
        """
        # Create a copy of the architecture
        mutated = {
            'conv_layers': [],
            'fc_layers': [],
            'input_channels': 3,
            'num_classes': 10,
        }
        
        # Deep copy of layers
        for layer in architecture['conv_layers']:
            mutated['conv_layers'].append(layer.copy())
            
        for layer in architecture['fc_layers']:
            mutated['fc_layers'].append(layer.copy())
        
        # Potentially add or remove a convolutional layer
        if np.random.random() < mutation_rate:
            if len(mutated['conv_layers']) > self.num_layers_range[0] and np.random.random() < 0.5:
                # Remove a random layer
                idx_to_remove = np.random.randint(0, len(mutated['conv_layers']))
                removed_layer = mutated['conv_layers'].pop(idx_to_remove)
                
                # Update channels for the next layer if it's not the last one
                if idx_to_remove < len(mutated['conv_layers']):
                    mutated['conv_layers'][idx_to_remove]['in_channels'] = removed_layer['in_channels']
            elif len(mutated['conv_layers']) < self.num_layers_range[1]:
                # Add a new layer at a random position
                idx_to_add = np.random.randint(0, len(mutated['conv_layers']) + 1)
                
                # Determine input and output channels
                if idx_to_add == 0:
                    in_channels = 3  # Input image channels
                else:
                    in_channels = mutated['conv_layers'][idx_to_add - 1]['out_channels']
                
                if idx_to_add == len(mutated['conv_layers']):
                    channel_index = np.random.randint(0, len(self.channel_choices))
                    out_channels = self.channel_choices[channel_index]
                else:
                    out_channels = mutated['conv_layers'][idx_to_add]['in_channels']
                
                # Create new layer
                op_index = np.random.randint(0, len(self.conv_ops))
                op_name = self.conv_ops[op_index][0]
                
                act_index = np.random.randint(0, len(self.activation_ops))
                act_name = self.activation_ops[act_index][0]
                
                use_pooling = np.random.random() < 0.5
                
                if use_pooling:
                    pool_index = np.random.randint(0, len(self.pooling_ops))
                    pool_name = self.pooling_ops[pool_index][0]
                else:
                    pool_name = None
                
                new_layer = {
                    'op': op_name,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'activation': act_name,
                    'pooling': pool_name,
                    'use_pooling': use_pooling
                }
                
                mutated['conv_layers'].insert(idx_to_add, new_layer)
                
                # Update the next layer's input channels if it's not the last one
                if idx_to_add < len(mutated['conv_layers']) - 1:
                    mutated['conv_layers'][idx_to_add + 1]['in_channels'] = out_channels
        
        # Mutate existing convolutional layers
        for layer in mutated['conv_layers']:
            # Mutate operation type
            if np.random.random() < mutation_rate:
                op_index = np.random.randint(0, len(self.conv_ops))
                layer['op'] = self.conv_ops[op_index][0]
            
            # Mutate output channels
            if np.random.random() < mutation_rate:
                channel_index = np.random.randint(0, len(self.channel_choices))
                layer['out_channels'] = self.channel_choices[channel_index]
            
            # Mutate activation function
            if np.random.random() < mutation_rate:
                act_index = np.random.randint(0, len(self.activation_ops))
                layer['activation'] = self.activation_ops[act_index][0]
            
            # Mutate pooling
            if np.random.random() < mutation_rate:
                layer['use_pooling'] = not layer['use_pooling']
                if layer['use_pooling']:
                    pool_index = np.random.randint(0, len(self.pooling_ops))
                    layer['pooling'] = self.pooling_ops[pool_index][0]
        
        # Recalculate feature dimensions for FC layers
        feature_size = 32
        num_pooling = 0
        for layer in mutated['conv_layers']:
            if layer['use_pooling']:
                num_pooling += 1
        feature_size = feature_size // (2 ** num_pooling)
        flattened_dim = mutated['conv_layers'][-1]['out_channels'] * feature_size * feature_size
        
        # Update first FC layer input dimension
        if mutated['fc_layers']:
            mutated['fc_layers'][0]['in_features'] = flattened_dim
        
        # Mutate FC layers
        for i, layer in enumerate(mutated['fc_layers']):
            # Skip the last layer (output layer)
            if i == len(mutated['fc_layers']) - 1 and layer['out_features'] == 10:
                continue
                
            # Mutate output features
            if np.random.random() < mutation_rate and i < len(mutated['fc_layers']) - 1:
                fc_index = np.random.randint(0, len(self.fc_sizes))
                layer['out_features'] = self.fc_sizes[fc_index]
                # Update next layer's input features
                if i + 1 < len(mutated['fc_layers']):
                    mutated['fc_layers'][i + 1]['in_features'] = layer['out_features']
            
            # Mutate dropout rate
            if np.random.random() < mutation_rate:
                dropout_index = np.random.randint(0, len(self.dropout_rates))
                layer['dropout_rate'] = self.dropout_rates[dropout_index]
        
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent architectures (for evolutionary search)
        
        Args:
            parent1 (dict): First parent architecture
            parent2 (dict): Second parent architecture
            
        Returns:
            dict: Child architecture with traits from both parents
        """
        child = {
            'conv_layers': [],
            'fc_layers': [],
            'input_channels': 3,
            'num_classes': 10,
        }
        
        # Crossover convolutional layers
        # Take layers from either parent with equal probability
        max_layers = max(len(parent1['conv_layers']), len(parent2['conv_layers']))
        min_layers = min(len(parent1['conv_layers']), len(parent2['conv_layers']))
        
        # Decide how many layers to include in the child
        num_layers = np.random.randint(min_layers, max_layers + 1)
        
        # Initialize with input channels
        in_channels = 3
        
        for i in range(num_layers):
            # Choose which parent to take this layer from
            if i < len(parent1['conv_layers']) and i < len(parent2['conv_layers']):
                # Both parents have this layer, choose randomly
                if np.random.random() < 0.5:
                    layer = parent1['conv_layers'][i].copy()
                else:
                    layer = parent2['conv_layers'][i].copy()
            elif i < len(parent1['conv_layers']):
                # Only parent1 has this layer
                layer = parent1['conv_layers'][i].copy()
            else:
                # Only parent2 has this layer
                layer = parent2['conv_layers'][i].copy()
            
            # Ensure correct input channels
            layer['in_channels'] = in_channels
            child['conv_layers'].append(layer)
            
            # Update input channels for next layer
            in_channels = layer['out_channels']
        
        # Calculate feature dimensions for FC layers
        feature_size = 32
        num_pooling = 0
        for layer in child['conv_layers']:
            if layer['use_pooling']:
                num_pooling += 1
        feature_size = feature_size // (2 ** num_pooling)
        flattened_dim = child['conv_layers'][-1]['out_channels'] * feature_size * feature_size
        
        # Crossover FC layers
        # For simplicity, take all FC layers from one parent
        if np.random.random() < 0.5 and parent1['fc_layers']:
            # Take from parent1
            for layer in parent1['fc_layers']:
                child['fc_layers'].append(layer.copy())
        elif parent2['fc_layers']:
            # Take from parent2
            for layer in parent2['fc_layers']:
                child['fc_layers'].append(layer.copy())
        else:
            # Fallback: create a simple FC layer
            child['fc_layers'].append({
                'in_features': flattened_dim,
                'out_features': 10,
                'activation': None,
                'dropout_rate': 0.0
            })
        
        # Ensure first FC layer has correct input dimension
        if child['fc_layers']:
            child['fc_layers'][0]['in_features'] = flattened_dim
        
        return child
