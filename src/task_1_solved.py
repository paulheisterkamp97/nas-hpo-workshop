import torch.nn as nn
import torch.optim as optim

from utils import get_dataset, get_best_torch_device, evaluate_model

"""
This task involves manually configuring a simple neural network for a data science task.
The dataset provided is CIFAR-10, which is an image classification dataset with 10 classes 
and images of resolution 32x32. The dataset classes are:
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'].

Focus on building the network structure. The training and evaluation loops are already provided.
"""

################################# Neural Network ##################################

class SimpleCNN(nn.Module):
    def __init__(self):
        """
        Define the structure of the neural network.
        The input tensor has dimensions 3x32x32 (3 color channels: RGB, image size: 32x32).
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3x32x32, Output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x8x8

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(256, 10)  # Output layer (10 classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
################################# Training Configuration ###########################

# Select the best available device (GPU, MPS, or CPU)
device = get_best_torch_device()

# Initialize the model and move it to the selected device
model = SimpleCNN().to(device)

# Define the loss function (CrossEntropyLoss is standard for classification tasks)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer with a learning rate of 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load Datasets (set batch size)
train_loader, test_loader = get_dataset(batch_size=64)

################################# Training Loop ##################################

num_epochs = 2  # Number of epochs for training

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0 
    
    # Iterate over the training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass: compute model predictions
        outputs = model(images)
        loss = criterion(outputs, labels) 
        
        # Backward pass: optimize the model
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()  
        
        running_loss += loss.item()  
    
    # Print the average loss for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


################################# Evaluation ##################################

# Evaluate the model on the test dataset and print the results
evaluate_model(model=model, test_loader=test_loader, verbose=True)
