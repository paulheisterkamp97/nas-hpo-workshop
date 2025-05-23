"""
Reinforcement Learning-based Neural Architecture Search (NAS)

This module implements reinforcement learning approaches for NAS,
including both policy gradient (REINFORCE) and Q-learning methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import deque

from .search_space import SearchSpace
from src.library.nas.utils_2 import get_dataset, train_model, evaluate_model, get_best_torch_device

class PolicyNetwork(nn.Module):
    """
    Policy network for the controller in policy gradient-based NAS
    
    This network learns to generate architecture decisions by predicting
    action probabilities at each decision point.
    """
    def __init__(self, input_size, hidden_size, output_size):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)  # Output probabilities for each action

class QNetwork(nn.Module):
    """
    Q-Network for Q-learning based NAS
    
    This network learns to predict the expected reward (Q-value) for
    each possible action given the current state.
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

class ReinforcementLearningSearch:
    """
    Reinforcement Learning-based Neural Architecture Search
    
    This class implements two RL approaches for NAS:
    1. Policy Gradient (REINFORCE): Learns a policy to directly generate architectures
    2. Q-Learning: Learns to select the best architecture from a set of candidates
    """
    def __init__(self, search_space, method='policy_gradient', num_episodes=50, 
                 epochs_per_model=5, batch_size=64, device=None, small_subset=True):
        """
        Initialize the RL-based NAS
        
        Args:
            search_space: SearchSpace object defining the architecture search space
            method: RL method to use ('policy_gradient' or 'q_learning')
            num_episodes: Number of episodes/iterations for the RL algorithm
            epochs_per_model: Number of epochs to train each sampled model
            batch_size: Batch size for training
            device: Device to use for training (if None, use best available)
            small_subset: Whether to use a small subset of data for faster evaluation
        """
        self.search_space = search_space
        self.method = method
        self.num_episodes = num_episodes
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
        
        # Define the action space for the RL agent
        self.setup_action_space()
        
        # Initialize the RL agent based on the chosen method
        if method == 'policy_gradient':
            self.setup_policy_gradient()
        elif method == 'q_learning':
            self.setup_q_learning()
        else:
            raise ValueError(f"Unknown RL method: {method}. Choose 'policy_gradient' or 'q_learning'.")
    
    def setup_action_space(self):
        """
        Set up the action space for the RL agent
        
        This defines the possible actions the agent can take at each decision point
        when designing a neural architecture.
        """
        # Define the possible actions for each decision point
        self.action_spaces = {
            'num_layers': len(range(self.search_space.num_layers_range[0], 
                                   self.search_space.num_layers_range[1] + 1)),
            'conv_op': len(self.search_space.conv_ops),
            'channel_size': len(self.search_space.channel_choices),
            'activation': len(self.search_space.activation_ops),
            'pooling': len(self.search_space.pooling_ops) + 1,  # +1 for "no pooling"
            'fc_size': len(self.search_space.fc_sizes),
            'dropout_rate': len(self.search_space.dropout_rates),
            'num_fc_layers': 2  # Either 1 or 2 FC layers
        }
        
        # Calculate total state and action dimensions for the RL agent
        if self.method == 'policy_gradient':
            # For policy gradient, we'll use a sequence of decisions
            self.max_sequence_length = (
                1 +  # num_layers
                self.search_space.num_layers_range[1] * 4 +  # conv_op, channel_size, activation, pooling for each layer
                2 +  # num_fc_layers, fc_size for first layer
                1    # dropout_rate for first layer
            )
            self.state_dim = self.max_sequence_length
            self.action_dim = max(self.action_spaces.values())
        
        elif self.method == 'q_learning':
            # For Q-learning, we'll encode the entire architecture as a state vector
            self.state_dim = (
                1 +  # num_layers
                self.search_space.num_layers_range[1] * 4 +  # conv_op, channel_size, activation, pooling for each layer
                2 +  # num_fc_layers, fc_size for first layer
                1    # dropout_rate for first layer
            )
            # Action is to select a complete architecture
            self.action_dim = self.num_episodes  # We'll generate this many candidate architectures
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, use a simplified policy gradient setup
    def setup_policy_gradient(self):
        """
        Set up the policy gradient (REINFORCE) agent
        
        The policy gradient method learns a policy that directly generates
        architecture decisions.
        """
        # Initialize the policy network
        self.policy_network = PolicyNetwork(
            input_size=self.state_dim,
            hidden_size=128,
            output_size=self.action_dim
        ).to(self.device)
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        
        # Storage for episode history
        self.saved_log_probs = []
        self.rewards = []
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, use a simplified Q-learning setup
    def setup_q_learning(self):
        """
        Set up the Q-learning agent
        
        Q-learning learns to predict the expected reward for each action
        and selects the action with the highest expected reward.
        """
        # Initialize the Q-network
        self.q_network = QNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim,
            hidden_size=128
        ).to(self.device)
        
        # Initialize the target network (for stable learning)
        self.target_network = QNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim,
            hidden_size=128
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Initialize the replay buffer for experience replay
        self.replay_buffer = deque(maxlen=1000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size_rl = 32  # Batch size for RL training (not to be confused with neural network training)
        self.update_target_every = 5  # Update target network every N episodes
    # OPTIONAL REMOVAL END
    
    def search(self, verbose=True, use_simple_sampling=False):
        """
        Perform architecture search using the selected RL method
        
        Args:
            verbose: Whether to print progress
            use_simple_sampling: Whether to use simple sampling instead of RL (for demonstration)
            
        Returns:
            Best architecture found and its corresponding model
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting {self.method.replace('_', ' ').title()} Search with {self.num_episodes} episodes...")
        
        # OPTIONAL REMOVAL START - For simpler explanation, use simple sampling instead of RL
        if use_simple_sampling:
            # Simple sampling for demonstration purposes
            for i in range(self.num_episodes):
                # Sample a simple architecture
                architecture = self.search_space.sample_simple_architecture()
                
                # Build and evaluate the model
                model = self.search_space.build_model_from_architecture(architecture)
                
                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Train the model
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
                
                # Evaluate the model
                eval_results = evaluate_model(
                    model=model,
                    test_loader=self.test_loader,
                    device=self.device,
                    verbose=False
                )
                
                # Store results
                result = {
                    'architecture': architecture,
                    'accuracy': eval_results['accuracy'],
                    'ms_per_image': eval_results['ms_per_image'],
                    'val_accuracy': history['best_val_acc'],
                    'model': model
                }
                
                self.results.append(result)
                
                # Update best architecture if needed
                if result['accuracy'] > self.best_accuracy:
                    self.best_architecture = architecture
                    self.best_accuracy = result['accuracy']
                    self.best_model = model
                    
                    if verbose:
                        print(f"New best architecture found! Accuracy: {self.best_accuracy:.2f}%")
        else:
            # Use the selected RL method
            if self.method == 'policy_gradient':
                self.search_policy_gradient(verbose)
            elif self.method == 'q_learning':
                self.search_q_learning(verbose)
        # OPTIONAL REMOVAL END
        
        end_time = time.time()
        search_time = (end_time - start_time) / 60  # in minutes
        
        if verbose:
            print(f"RL Search completed in {search_time:.2f} minutes")
            print(f"Best architecture accuracy: {self.best_accuracy:.2f}%")
        
        return self.best_architecture, self.best_model
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def search_policy_gradient(self, verbose=True):
        """
        Perform architecture search using policy gradient (REINFORCE)
        """
        # Use tqdm for progress tracking if verbose
        if verbose:
            iterator = tqdm(range(self.num_episodes))
        else:
            iterator = range(self.num_episodes)
        
        for episode in iterator:
            # Reset episode variables
            self.saved_log_probs = []
            
            # Generate an architecture by sampling from the policy
            actions = []
            state = np.zeros(self.state_dim)  # Initial state
            
            # Sample actions for each decision point
            for i in range(self.max_sequence_length):
                action = self.select_action_policy_gradient(state)
                actions.append(action)
                
                # Update state to include the selected action
                state[i] = action / self.action_dim  # Normalize
            
            # Decode actions to architecture
            architecture = self.decode_actions_to_architecture(actions)
            
            # Build and evaluate the model
            model = self.search_space.build_model_from_architecture(architecture)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train the model
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
            
            # Evaluate the model
            eval_results = evaluate_model(
                model=model,
                test_loader=self.test_loader,
                device=self.device,
                verbose=False
            )
            
            # Store results
            result = {
                'architecture': architecture,
                'accuracy': eval_results['accuracy'],
                'ms_per_image': eval_results['ms_per_image'],
                'val_accuracy': history['best_val_acc'],
                'model': model
            }
            
            self.results.append(result)
            
            # Update best architecture if needed
            if result['accuracy'] > self.best_accuracy:
                self.best_architecture = architecture
                self.best_accuracy = result['accuracy']
                self.best_model = model
                
                if verbose:
                    tqdm.write(f"New best architecture found! Accuracy: {self.best_accuracy:.2f}%")
            
            # Calculate reward (use validation accuracy as reward)
            reward = history['best_val_acc']
            self.rewards.append(reward)
            
            # Update policy network
            self.update_policy()
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def search_q_learning(self, verbose=True):
        """
        Perform architecture search using Q-learning
        """
        # Generate a pool of candidate architectures
        candidate_architectures = []
        for _ in range(self.action_dim):
            architecture = self.search_space.sample_random_architecture()
            candidate_architectures.append(architecture)
        
        # Use tqdm for progress tracking if verbose
        if verbose:
            iterator = tqdm(range(self.num_episodes))
        else:
            iterator = range(self.num_episodes)
        
        for episode in iterator:
            # Encode current state (initially random)
            if episode == 0:
                state = np.random.rand(self.state_dim)
            
            # Select an architecture using epsilon-greedy policy
            action = self.select_action_q_learning(state, self.epsilon)
            architecture = candidate_architectures[action]
            
            # Build and evaluate the model
            model = self.search_space.build_model_from_architecture(architecture)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train the model
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
            
            # Evaluate the model
            eval_results = evaluate_model(
                model=model,
                test_loader=self.test_loader,
                device=self.device,
                verbose=False
            )
            
            # Store results
            result = {
                'architecture': architecture,
                'accuracy': eval_results['accuracy'],
                'ms_per_image': eval_results['ms_per_image'],
                'val_accuracy': history['best_val_acc'],
                'model': model
            }
            
            self.results.append(result)
            
            # Update best architecture if needed
            if result['accuracy'] > self.best_accuracy:
                self.best_architecture = architecture
                self.best_accuracy = result['accuracy']
                self.best_model = model
                
                if verbose:
                    tqdm.write(f"New best architecture found! Accuracy: {self.best_accuracy:.2f}%")
            
            # Calculate reward (use validation accuracy as reward)
            reward = history['best_val_acc']
            
            # Encode next state (the architecture we just evaluated)
            next_state = self.encode_architecture_to_state(architecture)
            
            # Store transition in replay buffer
            self.replay_buffer.append((state, action, reward, next_state))
            
            # Update state
            state = next_state
            
            # Update Q-network
            if len(self.replay_buffer) > self.batch_size_rl:
                self.update_q_network()
            
            # Update target network
            if episode % self.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def select_action_policy_gradient(self, state):
        """
        Select an action using the policy network
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy_network(state)
        
        # Sample action from the probability distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # Save log probability for later use in policy update
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def select_action_q_learning(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Exploration: select a random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploitation: select the action with highest Q-value
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def update_policy(self):
        """
        Update the policy network using the REINFORCE algorithm
        """
        # Calculate returns (discounted rewards)
        R = 0
        returns = []
        
        # Calculate returns in reverse order
        for r in self.rewards[::-1]:
            R = r + 0.99 * R  # 0.99 is the discount factor
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode history
        self.saved_log_probs = []
        self.rewards = []
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def update_q_network(self):
        """
        Update the Q-network using experience replay
        """
        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size_rl)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
        
        # Compute target Q values
        target_q = rewards + self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def encode_architecture_to_state(self, architecture):
        """
        Encode an architecture specification into a state vector
        
        This converts the architecture dictionary into a fixed-length
        vector that can be used as input to the RL agent.
        """
        state = np.zeros(self.state_dim)
        
        # Encode number of convolutional layers
        num_conv_layers = len(architecture['conv_layers'])
        state[0] = num_conv_layers / self.search_space.num_layers_range[1]  # Normalize
        
        # Encode convolutional layers
        for i, layer in enumerate(architecture['conv_layers']):
            if i >= self.search_space.num_layers_range[1]:
                break
                
            base_idx = 1 + i * 4
            
            # Encode operation type
            op_idx = -1
            for j, (name, _) in enumerate(self.search_space.conv_ops):
                if name == layer['op']:
                    op_idx = j
                    break
            state[base_idx] = op_idx / (len(self.search_space.conv_ops) - 1)  # Normalize
            
            # Encode channel size
            channel_idx = self.search_space.channel_choices.index(layer['out_channels'])
            state[base_idx + 1] = channel_idx / (len(self.search_space.channel_choices) - 1)  # Normalize
            
            # Encode activation
            act_idx = -1
            for j, (name, _) in enumerate(self.search_space.activation_ops):
                if name == layer['activation']:
                    act_idx = j
                    break
            state[base_idx + 2] = act_idx / (len(self.search_space.activation_ops) - 1)  # Normalize
            
            # Encode pooling
            if layer['use_pooling']:
                pool_idx = -1
                for j, (name, _) in enumerate(self.search_space.pooling_ops):
                    if name == layer['pooling']:
                        pool_idx = j
                        break
                state[base_idx + 3] = (pool_idx + 1) / len(self.search_space.pooling_ops)  # +1 because 0 means no pooling
            else:
                state[base_idx + 3] = 0  # No pooling
        
        # Encode FC layers
        fc_base_idx = 1 + self.search_space.num_layers_range[1] * 4
        
        # Encode number of FC layers
        num_fc_layers = len(architecture['fc_layers'])
        state[fc_base_idx] = (num_fc_layers - 1) / 1  # Normalize (1 or 2 layers)
        
        # Encode first FC layer size
        if architecture['fc_layers']:
            fc_size = architecture['fc_layers'][0]['out_features']
            fc_size_idx = -1
            if fc_size in self.search_space.fc_sizes:
                fc_size_idx = self.search_space.fc_sizes.index(fc_size)
            else:
                fc_size_idx = 0
            state[fc_base_idx + 1] = fc_size_idx / (len(self.search_space.fc_sizes) - 1)  # Normalize
            
            # Encode dropout rate
            dropout_rate = architecture['fc_layers'][0]['dropout_rate']
            dropout_idx = self.search_space.dropout_rates.index(dropout_rate)
            state[fc_base_idx + 2] = dropout_idx / (len(self.search_space.dropout_rates) - 1)  # Normalize
        
        return state
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def decode_actions_to_architecture(self, actions):
        """
        Decode a sequence of actions into an architecture specification
        
        This converts the actions selected by the RL agent into a
        complete architecture specification.
        """
        architecture = {
            'conv_layers': [],
            'fc_layers': [],
            'input_channels': 3,
            'num_classes': 10,
        }
        
        # Decode number of convolutional layers
        num_layers_idx = actions[0] % self.action_spaces['num_layers']
        num_layers = self.search_space.num_layers_range[0] + num_layers_idx
        
        # Decode convolutional layers
        in_channels = 3  # Start with RGB input
        for i in range(num_layers):
            base_idx = 1 + i * 4
            
            # Decode operation type
            op_idx = actions[base_idx] % len(self.search_space.conv_ops)
            op_name = self.search_space.conv_ops[op_idx][0]
            
            # Decode channel size
            channel_idx = actions[base_idx + 1] % len(self.search_space.channel_choices)
            out_channels = self.search_space.channel_choices[channel_idx]
            
            # Decode activation
            act_idx = actions[base_idx + 2] % len(self.search_space.activation_ops)
            act_name = self.search_space.activation_ops[act_idx][0]
            
            # Decode pooling
            pool_idx = actions[base_idx + 3] % (len(self.search_space.pooling_ops) + 1)
            if pool_idx == 0:
                use_pooling = False
                pool_name = None
            else:
                use_pooling = True
                pool_name = self.search_space.pooling_ops[pool_idx - 1][0]
            
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
        feature_size = 32
        num_pooling = 0
        for layer in architecture['conv_layers']:
            if layer['use_pooling']:
                num_pooling += 1
        feature_size = feature_size // (2 ** num_pooling)
        
        # Calculate flattened feature dimension
        flattened_dim = in_channels * feature_size * feature_size
        
        # Decode FC layers
        fc_base_idx = 1 + self.search_space.num_layers_range[1] * 4
        
        # Decode number of FC layers
        num_fc_layers_idx = actions[fc_base_idx % len(actions)] % 2
        num_fc_layers = num_fc_layers_idx + 1  # Either 1 or 2 FC layers
        
        # Decode first FC layer
        fc_size_idx = actions[(fc_base_idx + 1) % len(actions)] % len(self.search_space.fc_sizes)
        fc_size = self.search_space.fc_sizes[fc_size_idx]
        
        dropout_idx = actions[(fc_base_idx + 2) % len(actions)] % len(self.search_space.dropout_rates)
        dropout_rate = self.search_space.dropout_rates[dropout_idx]
        
        architecture['fc_layers'].append({
            'in_features': flattened_dim,
            'out_features': fc_size if num_fc_layers > 1 else 10,
            'activation': 'relu' if num_fc_layers > 1 else None,
            'dropout_rate': dropout_rate
        })
        
        # Add second FC layer if needed
        if num_fc_layers > 1:
            architecture['fc_layers'].append({
                'in_features': fc_size,
                'out_features': 10,
                'activation': None,
                'dropout_rate': 0.0
            })
        
        return architecture
    # OPTIONAL REMOVAL END
    
    # OPTIONAL REMOVAL START - For simpler explanation, this method can be removed
    def plot_search_results(self):
        """
        Plot the results of the RL-based search
        """
        if not self.results:
            print("No search results to plot. Run search() first.")
            return
        
        # Extract accuracies and inference times
        accuracies = []
        inference_times = []
        
        for result in self.results:
            accuracies.append(result['accuracy'])
            inference_times.append(result['ms_per_image'])
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy over episodes
        plt.subplot(1, 2, 1)
        plt.plot(accuracies)
        plt.axhline(self.best_accuracy, color='r', linestyle='--', 
                   label=f'Best: {self.best_accuracy:.2f}%')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{self.method.replace("_", " ").title()} Search - Accuracy over Episodes')
        plt.legend()
        
        # Plot inference time over episodes
        plt.subplot(1, 2, 2)
        plt.plot(inference_times)
        
        # Find the inference time of the best model
        best_time = None
        for r in self.results:
            if r['accuracy'] == self.best_accuracy:
                best_time = r['ms_per_image']
                break
                
        plt.axhline(best_time, color='r', linestyle='--', 
                   label=f'Best model: {best_time:.2f} ms')
        plt.xlabel('Episode')
        plt.ylabel('Inference Time (ms/image)')
        plt.title(f'{self.method.replace("_", " ").title()} Search - Inference Time over Episodes')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot accuracy vs. inference time scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(inference_times, accuracies, alpha=0.7)
        plt.scatter([best_time], [self.best_accuracy], color='r', s=100, 
                   label='Best model')
        plt.xlabel('Inference Time (ms/image)')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{self.method.replace("_", " ").title()} Search - Accuracy vs. Inference Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
    # OPTIONAL REMOVAL END
    
    def get_best_architecture_summary(self):
        """
        Get a human-readable summary of the best architecture found
        """
        if self.best_architecture is None:
            return "No architecture found. Run search() first."
        
        arch = self.best_architecture
        
        summary = "Best Architecture Summary:\n"
        summary += f"Accuracy: {self.best_accuracy:.2f}%\n\n"
        
        summary += "Convolutional Layers:\n"
        for i, layer in enumerate(arch['conv_layers']):
            summary += f"Layer {i+1}:\n"
            summary += f"  Operation: {layer['op']}\n"
            summary += f"  Channels: {layer['in_channels']} -> {layer['out_channels']}\n"
            summary += f"  Activation: {layer['activation']}\n"
            summary += f"  Pooling: {layer['pooling'] if layer['use_pooling'] else 'None'}\n"
        
        summary += "\nFully Connected Layers:\n"
        for i, layer in enumerate(arch['fc_layers']):
            summary += f"Layer {i+1}:\n"
            summary += f"  Size: {layer['in_features']} -> {layer['out_features']}\n"
            summary += f"  Activation: {layer['activation'] if layer['activation'] else 'None'}\n"
            summary += f"  Dropout: {layer['dropout_rate']}\n"
        
        return summary
