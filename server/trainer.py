# Script for training the models
import datetime
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from config.ConfigLoader import ConfigLoader
from exp.FFNet import FFNet
from exp.Trainer import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import TensorDataset


def create_info_file(model, model_version, model_config, training_config, accuracy, precision, recall, f1):
    # Create a json file with the model info
    info = {
        "model": model,
        "model_version": model_version,
        "model_config": model_config,
        "training_config": training_config,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open(f"./exp/models/{model}/{model_version}/info.json", "w") as file:
        json.dump(info, file, indent=4)

def create_model_folder(model):
    # Create new folder for the model
    now = datetime.datetime.now()
    model_version = now.strftime("%Y-%m-%d")
    folder_path = os.path.join(f"./exp/models/{model}", model_version)
    # Check if the folder (and therefore the version) already exists
    if os.path.exists(folder_path):
        # Append a number to the model version
        i = 1
        while os.path.exists(os.path.join(folder_path + f"_{i}")):
            i += 1
        model_version = model_version + f"_{i}"
        folder_path = os.path.join(f"./exp/models/{model}", model_version)

    os.mkdir(folder_path)
    return folder_path, model_version

def train_svm():
    # Read data from csv file
    data = pd.read_csv(filepath_or_buffer="./exp/dataset.csv", sep=";")
    
    # Load config
    config_loader = ConfigLoader("./config/config.json")
    config_loader.load()
    config = config_loader.get_config()

    model_features = config["svm"]["model_features"]
    model_config = config["svm"]["model_config"]
    training_config = config["svm"]["training_config"]
    
    # Split data into features and targets
    X = data[model_features]
    y = data['CLASS_LABEL']
    
    # Create training, test and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    # Encode labels
    
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Init model
    C = model_config["C"]
    kernel = model_config["kernel"]
    gamma = model_config["gamma"]
    optimize = model_config["optimize"] #if model should choose parameters by itself
    
    svm_model = SVC()
    
    if optimize == True:
        param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        params = grid_search.best_params_
        print("Best Hyperparameters:", params)
    else:
        params = {"C": C, "kernel": kernel, "gamma": gamma}
    
    # Train model
    best_svm_model = SVC(**params)
    best_svm_model.fit(X_train, y_train)
    y_pred = best_svm_model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.show()
    
    # Save model
    folder_path, model_version = create_model_folder("svm")

    # Save specific training config and results to file
    create_info_file("svm", model_version, model_config, training_config, accuracy, precision, recall, f1)
    
    # Update model version in config
    config["svm"]["model_version"] = model_version

    # Save the used config
    with open(os.path.join(folder_path, "config.json"), "w") as file:
        json.dump(config, file, indent=4)
    # Save the scaler
    with open(os.path.join(folder_path, "scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)
    # Save training results to file
    with open(os.path.join(folder_path, "model.pkl"), "wb") as file:
        pickle.dump(best_svm_model, file)

def train_rf():
    
    # Read data from csv file
    df = pd.read_csv(filepath_or_buffer="./exp/dataset.csv", sep=";")
    # Load config
    config_loader = ConfigLoader("./config/config.json")
    config_loader.load()
    config = config_loader.get_config()

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
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.show()

    # >>>>>> SAVE MODEL
    folder_path, model_version = create_model_folder("rf")

    # Save specific training config and results to file
    create_info_file("rf", model_version, model_config, training_config, accuracy, precision, recall, f1)
    
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

    # >>>>>> SAVE MODEL
    folder_path, model_version = create_model_folder("ffnn")

    # Save specific training config and results to file
    create_info_file("ffnn", model_version, model_config, training_config, accuracy, precision, recall, f1)

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
    elif type == "svm":
        train_svm()
    else:
        print("Invalid model type")

if __name__ != "__main__":
    raise Exception("This script should not be imported")