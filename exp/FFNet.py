import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import pickle
from config.ConfigLoader import ConfigLoader

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
    
    def predict(self, features):
        output = self.forward(features)
        pred = torch.round(output.squeeze(1))
        pred = pred.item()
        print("OUTPUT", pred)
        return pred
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()
