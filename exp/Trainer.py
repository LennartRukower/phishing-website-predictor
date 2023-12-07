import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Trainer():
    
    model = None

    def __init__(self, model: nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, num_epochs, batch_size, train_x, train_y):
        num_batches = len(train_x) // batch_size
        for epoch in range(num_epochs):
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Print statistics
                if (i+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_batches}], Loss: {loss.item():.4f}')
            print(f'>>>> Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def validate(self, test_x, test_y):
        with torch.no_grad():
            outputs = self.model(test_x)
            val_loss = self.criterion(outputs, test_y)
            
            # Calculate metrics
            val_acc = accuracy_score(test_y, torch.argmax(outputs, dim=1))
            val_prec = precision_score(test_y, torch.argmax(outputs, dim=1), average='macro')
            val_rec = recall_score(test_y, torch.argmax(outputs, dim=1), average='macro')
            val_f1 = f1_score(test_y, torch.argmax(outputs, dim=1), average='macro')

            print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}, Validation precision: {val_prec:.4f}, Validation recall: {val_rec:.4f}, Validation F1: {val_f1:.4f}')
            