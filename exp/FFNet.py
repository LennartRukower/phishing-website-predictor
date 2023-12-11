import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
                self.hidden_layers[i].activation_function = activations[i]
            else:
                # Hidden Layers
                self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                self.hidden_layers[i].activation_function = activations[i]
            nn.init.xavier_uniform_(self.hidden_layers[i].weight)
            
        # Output Layer
        print(f"Dimension of output: {output_size}")
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        # Initialize weights
        for hidden_layer in self.hidden_layers:
            if hidden_layer.activation_function == 'ReLU':
                init.kaiming_uniform_(hidden_layer.weight, nonlinearity='relu')
            elif hidden_layer.activation_function in ['Sigmoid', 'tanh']:
                init.xavier_uniform_(hidden_layer.weight)
            else:
                 init.uniform_(hidden_layer.weight, -0.01, 0.01)
            hidden_layer.bias.data.fill_(0.01)        


    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            activation_type = hidden_layer.activation_function
            if (activation_type == "ReLU"):
                x = F.relu(hidden_layer(x))
            elif (activation_type == "Sigmoid"):
                x = F.sigmoid(hidden_layer(x))
        x = self.output(x)
        return x

def create_FFNet():
    # >>>>>> PREPARE DATA
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Read data from csv file
    df = pd.read_csv(filepath_or_buffer="./exp/dataset.csv", sep=";")

    model_features = [
        "NumDots",
        "SubdomainLevel",
        "PathLevel",    
        "UrlLength",
        "NumDash",
        "NumDashInHostname",
        "AtSymbol",
        "TildeSymbol",
        "NumUnderscore",
        "NumPercent",
        "NumQueryComponents",
        "NumAmpersand",
        "NumHash",
        "NumNumericChars",
        "IpAddress",
        "DomainInSubdomains",
        "DomainInPaths",
        "HttpsInHostname",
        "HostnameLength",
        "PathLength",
        "QueryLength",
        "DoubleSlashInPath",
        "NumSensitiveWords",
        "PctExtResourceUrls",
        "InsecureForms",
        "ExtFormAction",
        "PopUpWindow",
        "SubmitInfoToEmail",
        "IframeOrFrame",
        "MissingTitle",
        "ImagesOnlyInForm",
    ]
    df = df.replace('10.000.000.000', 1.0)
    # Split data into features and targets
    X = df.drop(["id", "CLASS_LABEL"], axis=1)
    X = X[model_features]
    y = df["CLASS_LABEL"]

    # Create training, test and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

    from sklearn.preprocessing import StandardScaler

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    # Scale the features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    import torch
    from torch.utils.data import TensorDataset

    # Convert to tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # >>>>>> INIT MODEL
    config = {
        "input": len(model_features),
        "output": 2,
        "hidden": [len(model_features),  48, 48],
        "activations": ["ReLU", "ReLU", "Sigmoid"]
    }
    input_size = config['input']
    hidden_sizes = config['hidden']
    output_size = config['output']
    activations = config["activations"]
    net = FFNet(input_size, hidden_sizes, output_size, activations)

    # >>>>>> TRAIN MODEL
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(net.parameters(), lr=0.00001) # Optimizer for backpropagation
    batch_size = 64
    epochs = 500

    from Trainer import Trainer
    trainer = Trainer(model=net, criterion=criterion, optimizer=optimizer)
    trainer.load_data(train_dataset, val_dataset, batch_size=batch_size)
    losses = trainer.train(num_epochs=epochs)
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

    print(trainer.evaluate())
    # Save training results to file
    with open("./exp/results.txt", "w") as f:
        f.write(f"Number of hidden layers: {len(hidden_sizes)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Training loss: {losses[-1]}\n")
        f.write(f"Validation accuracy: {trainer.evaluate()}\n")
        f.write(f"Max loss: {max(losses)}, Min loss: {min(losses)}\n")
        f.write("-------------------------------------\n")

    torch.save(net.state_dict(), "./exp/model.pt")

if __name__ == "__main__":
    create_FFNet()







