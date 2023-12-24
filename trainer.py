# Script for training the models
from config.ConfigLoader import ConfigLoader

def train_ffnn():
    # >>>>>> PREPARE DATA
    import os
    import datetime
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import pickle
    from exp.FFNet import FFNet
    from exp.Trainer import Trainer
    import matplotlib.pyplot as plt
    import json

    # Read data from csv file
    df = pd.read_csv(filepath_or_buffer="./exp/dataset.csv", sep=";")

    # Load config
    config_loader = ConfigLoader("./config/config.json")
    config_loader.load()
    config = config_loader.get_config()
    print(config)
    model_features = config["ffnn"]["model_features"]
    model_config = config["ffnn"]["model_config"]
    training_config = config["ffnn"]["training_config"]

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
    input_size = model_config['input']
    hidden_sizes = model_config['hidden']
    output_size = model_config['output']
    activations = model_config["activations"]
    net = FFNet(input_size, hidden_sizes, output_size, activations)

    # Print model summary
    print(net)

    # >>>>>> TRAIN MODEL
    criterion = None
    if training_config["loss_function"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif training_config["loss_function"] == "bce":
        criterion = torch.nn.BCELoss() # Loss function
    lr =  training_config["learning_rate"]
    optimizer = None
    if training_config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif training_config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]

    trainer = Trainer(model=net, criterion=criterion, optimizer=optimizer)
    trainer.load_data(train_dataset, val_dataset, batch_size=batch_size)
    losses = trainer.train(num_epochs=epochs)
    plt.plot(losses)
    plt.show()

    accuracy, precision, recall, f1 = trainer.evaluate()

    # Create new folder for the model
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("./exp/models", folder_name)
    # Check if the folder already exists
    if os.path.exists(folder_path):
        # Append a number to the folder name
        i = 1
        while os.path.exists(os.path.join(folder_path + f"_{i}")):
            i += 1
        folder_path = folder_path + f"_{i}"    
    os.mkdir(folder_path)


    # Save specific training config and results to file
    with open(os.path.join(folder_path, "info.txt"), "w") as file:
        file.write(f"Number of hidden layers: {len(hidden_sizes)}\n")
        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Number of epochs: {epochs}\n")
        file.write(f"Loss function: {criterion.__class__.__name__}\n")
        file.write(f"Learning rate: {lr}\n")
        file.write(f"Training loss: {losses[-1]}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1: {f1}\n")
    
    # Save the used config
    with open(os.path.join(folder_path, "config.json"), "w") as file:
        json.dump(config, file, indent=4)
    # Save the scaler
    with open(os.path.join(folder_path, "scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)

    # Save training results to file
    torch.save(net.state_dict(), os.path.join(folder_path, "model.pt"))


if __name__ == "__main__":
    import sys
    type = str(sys.argv[1])

    if type == "ffnn":
        train_ffnn()
    else:
        print("Invalid model type")

if __name__ != "__main__":
    raise Exception("This script should not be imported")