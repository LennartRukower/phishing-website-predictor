import torch.nn as nn
import torch.nn.functional as F

class FFNet(nn.Module):

    hidden_layers = []
    hidden_len = 0
    activation_functions = []

    
    def __init__(self, input_size, hidden_sizes, output_size, activations):
        super().__init__()
        hidden_len = len(hidden_sizes)
        self.activation_functions = activations

        # Check if an activation function is provided for every hidden layer
        if hidden_len != len(activations):
            raise Exception("Number of hidden layers and activation functions do not match!") 

        print(f"Dimension of input: {input_size}")
        print(f"Number of hidden layers: {hidden_len}")

        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_len):
            if i == 0:
                # Input Layer + First Hidden Layer
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                # Hidden Layers
                self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            nn.init.xavier_uniform_(self.hidden_layers[i].weight)
            
        # Output Layer
        print(f"Dimension of output: {output_size}")
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)
        

    def forward(self, x):
        for i in range(self.hidden_len):
            activation_type = self.activation_functions[i]
            if (activation_type == "ReLU"):
                x = F.relu(self.hidden_layers[i](x))
            elif (activation_type == "Sigmoid"):
                x = F.sigmoid(self.hidden_layers[i](x))
        x = self.output(x)
        return x


config = {
    "input": 48,
    "output": 2,
    "hidden": [48, 52, 52, 52, 48, 48, 48],
    "activations": ["ReLU", "ReLU", "ReLU", "Sigmoid", "ReLU", "Sigmoid", "Sigmoid"]
}
input_size = config['input']
hidden_sizes = config['hidden']
output_size = config['output']
activations = config["activations"]

import torch.optim as optim

# Init feed forward network
net = FFNet(input_size, hidden_sizes, output_size, activations)
criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(net.parameters(), lr=0.001) # Optimizer for backpropagation

# Training
# Read data from csv file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Read data from csv file
df = pd.read_csv(filepath_or_buffer="./exp/dataset.csv", sep=";")

df = df.replace('10.000.000.000', 1.0)
# Split data into features and targets
X = df.drop(["id", "CLASS_LABEL"], axis=1)
y = df["CLASS_LABEL"]


# Create training and test data
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Encode labels TODO: what happens here?
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Convert data to tensors
import torch
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Scale the features TODO: What happens here?
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Transform data to float32
X_train = X_train.float()
X_test = X_test.float()
y_train = y_train.long()
y_test = y_test.long()

# Print dtypes and shapes
print(f"X_train: {X_train.dtype}, {X_train.shape}")
print(f"y_train: {y_train.dtype}, {y_train.shape}")
print(f"X_test: {X_test.dtype}, {X_test.shape}")
print(f"y_test: {y_test.dtype}, {y_test.shape}")


# Train model
from Trainer import Trainer
trainer = Trainer(model=net, criterion=criterion, optimizer=optimizer)
trainer.train(num_epochs=10, batch_size=32, train_x=X_train, train_y=y_train)
trainer.validate(test_x=X_test, test_y=y_test)






