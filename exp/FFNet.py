import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class FFNet(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, activations, dropout_positions=None, dropout_probs=None):
        super().__init__()
        hidden_len = len(hidden_sizes)
        self.activation_functions = activations

        if hidden_len + 1 != len(activations):
            raise Exception("Number of hidden + output layers and activation functions do not match!") 

        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList() if dropout_positions else None

        for i in range(hidden_len):
            in_features = input_size if i == 0 else hidden_sizes[i-1]
            layer = nn.Linear(in_features, hidden_sizes[i])
            self.hidden_layers.append(layer)
            if activations[i] == 'ReLU':
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.00)

            # Adding dropout layers
            if dropout_positions and i in dropout_positions:
                prob = dropout_probs[dropout_positions.index(i)]
                self.dropout_layers.append(nn.Dropout(p=prob))

        self.output = nn.Linear(hidden_sizes[-1], output_size)
        if activations[-1] == 'ReLU':
            init.kaiming_uniform_(self.output.weight, nonlinearity='relu')
        else:
            init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0.00)

    def forward(self, x):
        for i, (hidden_layer, activation_function) in enumerate(zip(self.hidden_layers, self.activation_functions)):
            x = hidden_layer(x)
            if activation_function == "ReLU":
                x = F.relu(x)
            elif activation_function == "Sigmoid":
                x = torch.sigmoid(x)
            if self.dropout_layers and i < len(self.dropout_layers) and self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x)
        x = self.output(x)
        if self.activation_functions[-1] == "Sigmoid":
            x = torch.sigmoid(x)
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    from sklearn.preprocessing import StandardScaler

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    # Scale the features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

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
        "output": 1,
        "hidden": [64, 80, 126, 80, 64, 32],
        "activations": ["ReLU", "ReLU", "ReLU", "ReLU","ReLU", "ReLU", "Sigmoid"],
        "dropout_positions": [],
        "dropout_probs": [],
    }
    input_size = config['input']
    hidden_sizes = config['hidden']
    output_size = config['output']
    activations = config["activations"]
    net = FFNet(input_size, hidden_sizes, output_size, activations)

    # Print model summary
    print(net)

    # >>>>>> TRAIN MODEL
    import torch.optim as optim
    criterion = nn.BCELoss() # Loss function
    lr = 0.0001 # Learning rate
    optimizer = optim.Adam(net.parameters(), lr=lr) # Optimizer for backpropagation
    batch_size = 64
    epochs = 200

    from Trainer import Trainer
    trainer = Trainer(model=net, criterion=criterion, optimizer=optimizer)
    trainer.load_data(train_dataset, val_dataset, batch_size=batch_size)
    losses = trainer.train(num_epochs=epochs)
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

    accuracy, precision, recall, f1 = trainer.evaluate()
    with open("./exp/results.txt", "a") as f:
        f.write(f"Number of hidden layers: {len(hidden_sizes)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Loss function: {criterion.__class__.__name__}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Training loss: {losses[-1]}\n")
        f.write(f"Validation accuracy: {accuracy}\n")
        f.write(f"Max loss: {max(losses)}, Min loss: {min(losses)}\n")
        f.write("-------------------------------------\n")

    # Save training results to file
    torch.save(net.state_dict(), "./exp/model.pt")

if __name__ == "__main__":
    create_FFNet()
