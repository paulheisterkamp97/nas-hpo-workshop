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
    Policy network for the controller in RL-based NAS
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class QNetwork(nn.Module):
    """
    Q-Network for Q-learning based NAS
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReinforcementLearningSearch:
    """
    Reinforcement Learning-based Neural Architecture Search
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
        
        if device is None:
            self.device = get_best_torch_device()
        else:
            self.device = device
            
        # Load datasets
        self.train_loader, self.val_loader, self.test_loader = get_dataset(
            batch_size=batch_size, small_subset=small_subset
        )
        
        # Results storage
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
    
    def setup_policy_gradient(self):
        """
        Set up the policy gradient (REINFORCE) agent
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
    
    def setup_q_learning(self):
        """
        Set up the Q-learning agent
        """
        # Initialize the Q-network
        self.q_network = QNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim,
            hidden_size=128
        ).to(self.device)
        
        # Initialize the target network
        self.target_network = QNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim,
            hidden_size=128
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Initialize the replay buffer
        self.replay_buffer = deque(maxlen=1000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size_rl = 32  # Batch size for RL training (not to be confused with neural network training)
        self.update_target_every = 5  # Update target network every N episodes
    
    def select_action_policy_gradient(self, state):
        """
        Select an action using the policy network
        """
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy_network(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def select_action_q_learning(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()
    
    def encode_architecture_to_state(self, architecture):
        """
        Encode an architecture specification into a state vector
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
            op_idx = next(j for j, (name, _) in enumerate(self.search_space.conv_ops) if name == layer['op'])
            state[base_idx] = op_idx / (len(self.search_space.conv_ops) - 1)  # Normalize
            
            # Encode channel size
            channel_idx = self.search_space.channel_choices.index(layer['out_channels'])
            state[base_idx + 1] = channel_idx / (len(self.search_space.channel_choices) - 1)  # Normalize
            
            # Encode activation
            act_idx = next(j for j, (name, _) in enumerate(self.search_space.activation_ops) if name == layer['activation'])
            state[base_idx + 2] = act_idx / (len(self.search_space.activation_ops) - 1)  # Normalize
            
            # Encode pooling
            if layer['use_pooling']:
                pool_idx = next(j for j, (name, _) in enumerate(self.search_space.pooling_ops) if name == layer['pooling'])
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
            fc_size_idx = self.search_space.fc_sizes.index(architecture['fc_layers'][0]['out_features']) if architecture['fc_layers'][0]['out_features'] in self.search_space.fc_sizes else 0
            state[fc_base_idx + 1] = fc_size_idx / (len(self.search_space.fc_sizes) - 1)  # Normalize
            
            # Encode dropout rate
            dropout_idx = self.search_space.dropout_rates.index(architecture['fc_layers'][0]['dropout_rate'])
            state[fc_base_idx + 2] = dropout_idx / (len(self.search_space.dropout_rates) - 1)  # Normalize
        
        return state
    
    def decode_actions_to_architecture(self, actions):
        """
        Decode a sequence of actions into an architecture specification
        """
        architecture = {
            'conv_layers': [],
            'fc_layers': [],
            'input_channels': 3,
            'num_classes': 10,
        }
        
        # Decode number of convolutional layers
        num_layers_idx = actions[0]
        num_layers = self.search_space.num_layers_range[0] + num_layers_idx
        
        # Decode convolutional layers
        in_channels = 3  # Start with RGB input
        for i in range(num_layers):
            base_idx = 1 + i * 4
            
            # Decode operation type
            op_idx = actions[base_idx] % len(self.search_space.conv_ops)
            op_name, _ = self.search_space.conv_ops[op_idx]
            
            # Decode channel size
            channel_idx = actions[base_idx + 1] % len(self.search_space.channel_choices)
            out_channels = self.search_space.channel_choices[channel_idx]
            
            # Decode activation
            act_idx = actions[base_idx + 2] % len(self.search_space.activation_ops)
            act_name, _ = self.search_space.activation_ops[act_idx]
            
            # Decode pooling
            pool_idx = actions[base_idx + 3] % (len(self.search_space.pooling_ops) + 1)
            if pool_idx == 0:
                use_pooling = False
                pool_name = None
            else:
                use_pooling = True
                pool_name, _ = self.search_space.pooling_ops[pool_idx - 1]
            
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
        num_pooling = sum(1 for layer in architecture['conv_layers'] if layer['use_pooling'])
        feature_size = feature_size // (2 ** num_pooling)
        
        # Calculate flattened feature dimension
        flattened_dim = in_channels * feature_size * feature_size
        
        # Decode FC layers
        fc_base_idx = 1 + num_layers * 4
        
        # Decode number of FC layers
        num_fc_layers_idx = actions[fc_base_idx] % 2
        num_fc_layers = num_fc_layers_idx + 1  # Either 1 or 2 FC layers
        
        # Decode first FC layer
        fc_size_idx = actions[fc_base_idx + 1] % len(self.search_space.fc_sizes)
        fc_size = self.search_space.fc_sizes[fc_size_idx]
        
        dropout_idx = actions[fc_base_idx + 2] % len(self.search_space.dropout_rates)
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
    
    def generate_random_architecture(self):
        """
        Generate a random architecture by sampling random actions
        """
        actions = []
        
        # Sample number of convolutional layers
        num_layers_idx = np.random.randint(0, self.action_spaces['num_layers'])
        actions.append(num_layers_idx)
        
        # Sample convolutional layers
        num_layers = self.search_space.num_layers_range[0] + num_layers_idx
        for _ in range(num_layers):
            # Sample operation type
            op_idx = np.random.randint(0, self.action_spaces['conv_op'])
            actions.append(op_idx)
            
            # Sample channel size
            channel_idx = np.random.randint(0, self.action_spaces['channel_size'])
            actions.append(channel_idx)
            
            # Sample activation
            act_idx = np.random.randint(0, self.action_spaces['activation'])
            actions.append(act_idx)
            
            # Sample pooling
            pool_idx = np.random.randint(0, self.action_spaces['pooling'])
            actions.append(pool_idx)
        
        # Pad with zeros for unused layers
        for _ in range(self.search_space.num_layers_range[1] - num_layers):
            actions.extend([0, 0, 0, 0])
        
        # Sample FC layers
        num_fc_layers_idx = np.random.randint(0, self.action_spaces['num_fc_layers'])
        actions.append(num_fc_layers_idx)
        
        fc_size_idx = np.random.randint(0, self.action_spaces['fc_size'])
        actions.append(fc_size_idx)
        
        dropout_idx = np.random.randint(0, self.action_spaces['dropout_rate'])
        actions.append(dropout_idx)
        
        # Pad with zeros to fixed length
        while len(actions) < self.max_sequence_length:
            actions.append(0)
        
        return actions
    
    def evaluate_architecture(self, architecture):
        """
        Evaluate an architecture by training and testing a model
        
        Returns:
            Reward (accuracy)
        """
        # Build model from architecture
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
        
        # Return reward (accuracy)
        return eval_results['accuracy']
    
    def policy_gradient_update(self):
        """
        Update the policy network using the REINFORCE algorithm
        """
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate returns
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode history
        self.saved_log_probs = []
        self.rewards = []
    
    def q_learning_update(self):
        """
        Update the Q-network using experience replay
        """
        if len(self.replay_buffer) < self.batch_size_rl:
            return
        
        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size_rl)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        next_q = self.target_network(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def search_policy_gradient(self, verbose=True):
        """
        Perform architecture search using policy gradient (REINFORCE)
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting Policy Gradient Search with {self.num_episodes} episodes...")
            iterator = tqdm(range(self.num_episodes))
        else:
            iterator = range(self.num_episodes)
        
        for i in iterator:
            # Reset state
            state = np.zeros(self.state_dim)
            actions = []
            
            # Generate an architecture by sampling actions from the policy network
            for t in range(self.max_sequence_length):
                action = self.select_action_policy_gradient(state)
                actions.append(action)
                
                # Update state (in a real environment, this would be done by the environment)
                # Here we just set the corresponding element to 1 to indicate the action taken
                if t < len(state):
                    state[t] = action / self.action_dim  # Normalize
            
            # Decode actions to architecture
            architecture = self.decode_actions_to_architecture(actions)
            
            # Evaluate architecture and get reward
            reward = self.evaluate_architecture(architecture)
            self.rewards.append(reward)
            
            # Update policy network
            if (i + 1) % 5 == 0 or i == self.num_episodes - 1:
                self.policy_gradient_update()
                
                if verbose:
                    tqdm.write(f"Episode {i+1}/{self.num_episodes}, Best accuracy: {self.best_accuracy:.2f}%")
        
        end_time = time.time()
        search_time = (end_time - start_time) / 60  # in minutes
        
        if verbose:
            print(f"Policy Gradient Search completed in {search_time:.2f} minutes")
            print(f"Best architecture accuracy: {self.best_accuracy:.2f}%")
        
        return self.best_architecture, self.best_model
    
    def search_q_learning(self, verbose=True):
        """
        Perform architecture search using Q-learning
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting Q-Learning Search with {self.num_episodes} episodes...")
            iterator = tqdm(range(self.num_episodes))
        else:
            iterator = range(self.num_episodes)
        
        # Generate a pool of candidate architectures
        candidate_architectures = [self.search_space.sample_random_architecture() 
                                  for _ in range(self.action_dim)]
        
        for i in iterator:
            # Current state is a random initial architecture
            current_arch_idx = np.random.randint(0, len(candidate_architectures))
            current_arch = candidate_architectures[current_arch_idx]
            current_state = self.encode_architecture_to_state(current_arch)
            
            # Select action (which architecture to try next)
            action = self.select_action_q_learning(current_state, self.epsilon)
            next_arch = candidate_architectures[action]
            next_state = self.encode_architecture_to_state(next_arch)
            
            # Evaluate architecture and get reward
            reward = self.evaluate_architecture(next_arch)
            
            # Store transition in replay buffer
            done = (i == self.num_episodes - 1)
            self.replay_buffer.append((current_state, action, reward, next_state, done))
            
            # Update Q-network
            self.q_learning_update()
            
            # Update target network
            if (i + 1) % self.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if verbose and ((i + 1) % 5 == 0 or i == self.num_episodes - 1):
                tqdm.write(f"Episode {i+1}/{self.num_episodes}, "
                          f"Epsilon: {self.epsilon:.4f}, "
                          f"Best accuracy: {self.best_accuracy:.2f}%")
        
        end_time = time.time()
        search_time = (end_time - start_time) / 60  # in minutes
        
        if verbose:
            print(f"Q-Learning Search completed in {search_time:.2f} minutes")
            print(f"Best architecture accuracy: {self.best_accuracy:.2f}%")
        
        return self.best_architecture, self.best_model
    
    def search(self, verbose=True):
        """
        Perform architecture search using the selected RL method
        """
        if self.method == 'policy_gradient':
            return self.search_policy_gradient(verbose)
        elif self.method == 'q_learning':
            return self.search_q_learning(verbose)
    
    def plot_search_results(self):
        """
        Plot the results of the RL-based search
        """
        if not self.results:
            print("No search results to plot. Run search() first.")
            return
        
        # Extract accuracies and inference times
        accuracies = [result['accuracy'] for result in self.results]
        inference_times = [result['ms_per_image'] for result in self.results]
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy distribution
        plt.subplot(1, 2, 1)
        plt.hist(accuracies, bins=10, alpha=0.7)
        plt.axvline(self.best_accuracy, color='r', linestyle='--', 
                   label=f'Best: {self.best_accuracy:.2f}%')
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Count')
        plt.title(f'Distribution of Model Accuracies ({self.method})')
        plt.legend()
        
        # Plot inference time distribution
        plt.subplot(1, 2, 2)
        plt.hist(inference_times, bins=10, alpha=0.7)
        best_time = next(r['ms_per_image'] for r in self.results 
                         if r['accuracy'] == self.best_accuracy)
        plt.axvline(best_time, color='r', linestyle='--', 
                   label=f'Best model: {best_time:.2f} ms')
        plt.xlabel('Inference Time (ms/image)')
        plt.ylabel('Count')
        plt.title('Distribution of Inference Times')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot learning curve if using policy gradient
        if self.method == 'policy_gradient' and len(self.results) >= 5:
            plt.figure(figsize=(10, 6))
            
            # Calculate moving average of accuracy
            window_size = min(5, len(accuracies) // 5)
            moving_avg = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
            
            plt.plot(range(len(moving_avg)), moving_avg, label='Moving Average')
            plt.plot(range(len(accuracies)), accuracies, 'o', alpha=0.5, label='Individual Models')
            plt.axhline(self.best_accuracy, color='r', linestyle='--', 
                       label=f'Best: {self.best_accuracy:.2f}%')
            
            plt.xlabel('Model Index')
            plt.ylabel('Accuracy (%)')
            plt.title('Learning Curve (Policy Gradient)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
    
    def get_best_architecture_summary(self):
        """
        Get a summary of the best architecture found
        """
        if self.best_architecture is None:
            return "No architecture found. Run search() first."
        
        arch = self.best_architecture
        
        summary = f"Best Architecture Summary ({self.method}):\n"
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
