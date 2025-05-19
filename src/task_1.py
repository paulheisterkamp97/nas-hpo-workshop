import torch.nn as nn
import torch.optim as optim

from utils import get_dataset, get_best_torch_device, evaluate_model

"""
This task involves manually configuring a simple neural network for a data science task.
The dataset provided is CIFAR-10, which is an image classification dataset with 10 classes 
and images of resolution 32x32. The dataset classes are:
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'].

This Task is comprised of two parts (a,b)

Task 1a:
In the first part, focus on building the network structure. The training and evaluation 
loops are already provided. Training Hyperparameters are set to suitable values, we will 
fine tune them later.

Task 1b:
In the "Training Configuration" section, you can improve the effectiveness of model 
training by adjusting hyperparameters. Start by modifying parameters like batch size, 
epoch count, or learning rate. You can also expand the configuration by introducing 
new elements, such as adding Dropout to the model, and then testing different Dropout 
rates to evaluate their impact on performance.
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
            # TASK 1a:  Your Model goes here
            


            nn.Linear(42 , 10)
            # The output shape has to be (10), change the first parameter to suit your model
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
################################# Training Configuration ###########################
# Select the best available device (GPU, MPS, or CPU)
device = get_best_torch_device()

# Initialize the model and move it to the selected device
# If you want to give hyperparameters to the model, you could do that via the constructor
model = SimpleCNN().to(device)

#### TASK 1b: Experiment with different Hyperparameters: ####
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Select optimizer and set the defined learn-rate check the 'optim' package for alternatives
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Select the loss function (CE is a good default but there are others for multi-class classification)
criterion = nn.CrossEntropyLoss()
#### --------------------------------------------------- ####

# Load Datasets
train_loader, test_loader = get_dataset(batch_size=batch_size)

################################# Training Loop ##################################

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