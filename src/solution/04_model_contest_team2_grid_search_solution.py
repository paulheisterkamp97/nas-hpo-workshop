# Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO) Workshop
# Welcome to the practical part of the workshop! 🎉

# Goal:
# In this workshop section, you will work in groups to:
# - Perform Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO)
# - Implement different search strategies
# - Compare your results and see which group finds the best model! 🚀

# Instructions:
# 1. Task: Your task is to optimize a model for the FashionMNIST dataset.
# 2. Group Formation: Each group will choose a search strategy (e.g., Random Search, Bayesian Optimization, Grid Search).
# 3. Implementation: Try to fill in all ToDos.
# 4. Comparison: At the end of the workshop, we will compare the results of the groups.

# Notes:
# - Each group works independently and fills out the provided cells.
# - Document your steps well so that other groups can understand what you have done.
# - Have fun and good luck! 😊

# Preparation
# Please install the necessary libraries.

# Data Loading
# We will use the FashionMNIST dataset. The following code loads and prepares the data.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna
import optuna_dashboard
import itertools

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training data: {len(train_dataset)}, Test data: {len(test_dataset)}")

storage = "sqlite:///optuna_study.db"

class FashionMNISTCNN(nn.Module):
    def __init__(self, num_conv_layers, num_filters, kernel_size, dropout_rate):
        super(FashionMNISTCNN, self).__init__()
        layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            dummy_output = self.conv(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        self.fc = nn.Linear(flattened_size, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

search_space = {
    'num_conv_layers': [1, 2, 3],
    'num_filters': [16, 32],
    'kernel_size': [3, 5],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.001, 0.01]
}

n_trials = 1
for values in search_space.values():
    n_trials *= len(values)

print(f"Count Trials Grid Search: {n_trials}")

def objective_grid(trial):
    num_conv_layers = trial.suggest_categorical('num_conv_layers', search_space['num_conv_layers'])
    num_filters = trial.suggest_categorical('num_filters', search_space['num_filters'])
    kernel_size = trial.suggest_categorical('kernel_size', search_space['kernel_size'])
    dropout_rate = trial.suggest_categorical('dropout_rate', search_space['dropout_rate'])
    learning_rate = trial.suggest_categorical('learning_rate', search_space['learning_rate'])

    model = FashionMNISTCNN(num_conv_layers, num_filters, kernel_size, dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy

grid_sampler = optuna.samplers.GridSampler(search_space)

# Create the study and optimize with Grid Search
study_grid = optuna.create_study(
    direction='maximize',
    study_name="grid_search_optimization",
    storage=storage,
    load_if_exists=True,
    sampler=grid_sampler)
study_grid.optimize(objective_grid, n_trials=5)

# Print the best result
print(f"Best configuration (Grid Search): {study_grid.best_params}, Accuracy: {study_grid.best_value}")

optuna_dashboard.run_server(storage="sqlite:///optuna_study.db")
