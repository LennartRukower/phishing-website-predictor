import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Trainer():
    
    model = None
    all_preds = []
    all_labels = []

    def __init__(self, model: nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def load_data(self, train_data, val_data, batch_size):
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size)
        
    def train(self, num_epochs, stop_criterion = None):
        # Check if loaders are initialized
        if self.train_loader is None or self.val_loader is None:
            raise Exception('Loaders are not initialized. Call load_data() before training.')
        losses = []
        all_outputs = []
        for epoch in range(num_epochs):
            for inputs, labels in self.train_loader:
                # Forward pass
                outputs = self.model.forward(inputs)
                # Add every single output to the list of all outputs
                all_outputs.extend(outputs.cpu().detach().numpy())

                # Check which loss function to use
                if self.criterion.__class__.__name__ == "CrossEntropyLoss":
                    loss = self.criterion(outputs, labels)
                elif self.criterion.__class__.__name__ == "BCELoss":
                    loss = self.criterion(outputs, labels.unsqueeze(1).type(torch.float))
                losses.append(loss.item())

                # Checks if the loss is below the stop criterion
                if stop_criterion is not None and loss < stop_criterion:
                    print (f"Training stopped at epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]:.4f}")
                    return losses
            
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]:.4f}')
                print(f"Max output: {max(outputs.cpu().detach().numpy())}, Min output: {min(outputs.cpu().detach().numpy())}")
        print (f"Final Training Loss: {losses[-1]:.4f}")
        print(f"Max output: {max(all_outputs)}, Min output: {min(all_outputs)}")
        return losses
            
    def evaluate(self):
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)

                # Convert outputs to predicted class (argmax over the output probabilities)
                preds = torch.argmax(outputs, dim=1)

                # Store predictions and actual labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Compute metrics for validation set
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro')  # 'macro' for multi-class
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'Validation Accuracy: {accuracy}')
            print(f'Validation Precision: {precision}')
            print(f'Validation Recall: {recall}')
            print(f'Validation F1 Score: {f1}')
            return accuracy, precision, recall, f1
        