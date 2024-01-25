# Script for training the models
import numpy as np
from config.ConfigLoader import ConfigLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import json

def train_rf():
    from sklearn.ensemble import RandomForestClassifier   

    # Read data from csv file
    df = pd.read_csv(filepath_or_buffer="./exp/dataset.csv", sep=";")
    # Load config
    config_loader = ConfigLoader("./config/config.json")
    config_loader.load()
    config = config_loader.get_config()

    # TODO: Use config
    model_features = config["rf"]["model_features"]
    model_config = config["rf"]["model_config"]
    training_config = config["rf"]["training_config"]

    df = df.replace('10.000.000.000', 1.0)
    # Split data into features and targets
    X = df.drop(["id", "CLASS_LABEL"], axis=1)
    X = X[model_features]
    y = df["CLASS_LABEL"]

    # Create training, test and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    # Scale the features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # >>>>>> INIT MODEL
    n_estimators = model_config["nEstimators"]
    max_depth = model_config["maxDepth"] if model_config["maxDepth"] != -1 else None
    min_samples_split = model_config["minSamplesSplit"]
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    # >>>>>> TRAIN MODEL
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    # >>>>>> EVALUATE MODEL
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.show()

    # >>>>>> SAVE MODEL
    import os
    import datetime
    import pickle
    now = datetime.datetime.now()
    model_version = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("./exp/models/rf", model_version)
    # Check if the folder (and therefore the version) already exists
    if os.path.exists(folder_path):
        # Append a number to the model version
        i = 1
        while os.path.exists(os.path.join(folder_path + f"_{i}")):
            i += 1
        model_version = model_version + f"_{i}"
        folder_path = os.path.join("./exp/models/rf", model_version)

    os.mkdir(folder_path)

    # Save specific training config and results to file
    with open(os.path.join(folder_path, "info.txt"), "w") as file:
        file.write(f"Number of trees: {rf.n_estimators}\n")
        file.write(f"Max depth: {rf.max_depth}\n")
        file.write(f"Training accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1: {f1}\n")
    
    # Update model version in config
    config["rf"]["model_version"] = model_version

    # Save the used config
    with open(os.path.join(folder_path, "config.json"), "w") as file:
        json.dump(config, file, indent=4)
    # Save the scaler
    with open(os.path.join(folder_path, "scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)
    
    # Save training results to file
    with open(os.path.join(folder_path, "model.pkl"), "wb") as file:
        pickle.dump(rf, file)


def train_ffnn():
    # >>>>>> PREPARE DATA
    import os
    import datetime
    import torch
    import pickle
    from exp.FFNet import FFNet
    from exp.Trainer import Trainer
    import matplotlib.pyplot as plt

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
    model_version = now.strftime("%Y-%m-%d")
    folder_path = os.path.join("./exp/models/ffnn", model_version)
    # Check if the folder (and therefore the version) already exists
    if os.path.exists(folder_path):
        # Append a number to the model version
        i = 1
        while os.path.exists(os.path.join(folder_path + f"_{i}")):
            i += 1
        model_version = model_version + f"_{i}"
        folder_path = os.path.join("./exp/models/ffnn", model_version)

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

    # Update model version in config
    config["ffnn"]["model_version"] = model_version

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
    elif type == "rf":
        train_rf()
    else:
        print("Invalid model type")

if __name__ != "__main__":
    raise Exception("This script should not be imported")