import torch.nn as nn
import torch.nn.init as init


class SelfNormalizingFNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:
                layers.append(nn.SELU())

        self.snn = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.snn:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='selu')

    def forward(self, x):
        x = self.snn.forward(x)
        return x
