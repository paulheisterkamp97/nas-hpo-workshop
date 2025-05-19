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
# Please install the necessary libraries. Run the following code to ensure everything is ready:
# !pip install tensorflow keras scikit-learn optuna

# Data Loading
# We will use the FashionMNIST dataset. The following code loads and prepares the data.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna
import optuna_dashboard

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
    def __init__(self, input_size, num_layers, num_units, dropout_rate):
        super(FashionMNISTModel, self).__init__()
        layers = [nn.Flatten()]
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_units))
            layers.append(nn.ReLU())
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


# Group 3: Bayesian Optimization
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 4)
    num_units = trial.suggest_categorical('num_units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    # Modell erstellen
    model = FashionMNISTModel(input_size=28 * 28, num_layers=num_layers, num_units=num_units, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimierer und Verlustfunktion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    for epoch in range(5):  # 5 Epochen
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


# Create a study and optimize
study = optuna.create_study(
    direction='maximize',
    study_name="bayesian_optimization",
    storage=storage,
    load_if_exists=True,)
study.optimize(objective, n_trials=20)

print(f"Best configuration: {study.best_params}, Accuracy: {study.best_value}")

optuna_dashboard.run_server(storage="sqlite:///optuna_study.db")

