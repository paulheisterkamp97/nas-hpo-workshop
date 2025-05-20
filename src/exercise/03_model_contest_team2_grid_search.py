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

# FashionMNIST Daten laden
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training data: {len(train_dataset)}, Test data: {len(test_dataset)}")

# Study storage
storage = "sqlite:///optuna_study.db"


# Define Base Model
class FashionMNISTModel(nn.Module):
    def __init__(self, input_size, num_layers, num_units, dropout_rate, activation_function):
        super(FashionMNISTModel, self).__init__()
        layers = [nn.Flatten()]
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_units))
            layers.append(activation_function)
            layers.append(nn.Dropout(dropout_rate))
            input_size = num_units
        layers.append(nn.Linear(num_units, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Search Strategies
# Each group will choose a search strategy. Examples:
# - Group 1: Random Search
# - Group 2: Grid Search
# - Group 3: Bayesian Optimization (e.g., with [Optuna](https://optuna.org/))

# Group 2: Grid Search
# Search Space
# ToDo: search_space = {
#     'num_layers': ...,
#     'num_units': ...,
#     'dropout_rate': ...,
#     'learning_rate': ...,
#    'activation_function': ['ReLU', 'Tanh', 'Sigmoid']
# }

n_trials = 1
for values in search_space.values():
    n_trials *= len(values)

print(f"Count Trials Grid Search: {n_trials}")


# Define the objective function for Random Search
def objective_grid(trial):
    # Hyperparameter aus dem Suchraum auswählen
    num_layers = trial.suggest_categorical('num_layers', search_space['num_layers'])
    num_units = trial.suggest_categorical('num_units', search_space['num_units'])
    dropout_rate = trial.suggest_categorical('dropout_rate', search_space['dropout_rate'])
    learning_rate = trial.suggest_categorical('learning_rate', search_space['learning_rate'])

    activation_choice = trial.suggest_categorical('activation_function', search_space['activation_function'])
    if activation_choice == 'ReLU':
        activation_function = nn.ReLU()
    elif activation_choice == 'Tanh':
        activation_function = nn.Tanh()
    else:
        activation_function = nn.Sigmoid()

    model = FashionMNISTModel(
        input_size=28 * 28,
        num_layers=num_layers,
        num_units=num_units,
        dropout_rate=dropout_rate,
        activation_function=activation_function
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
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


# Create a GridSampler
grid_sampler = optuna.samplers.GridSampler(search_space)

# Create the study and optimize with Grid Search
study_grid = optuna.create_study(
    direction='maximize',
    study_name="grid_search_optimization",
    storage=storage,
    load_if_exists=True,
    sampler=grid_sampler)
study_grid.optimize(objective_grid, n_trials=n_trials)

# Print the best result
print(f"Best configuration (Grid Search): {study_grid.best_params}, Accuracy: {study_grid.best_value}")

optuna_dashboard.run_server(storage="sqlite:///optuna_study.db")
