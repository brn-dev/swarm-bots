import numpy as np
import torch.nn as nn
import torch.nn.init as init


class SelfNormalizingFNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layers_sizes) - 1):
            linear = nn.Linear(layers_sizes[i], layers_sizes[i + 1])

            # lecun initialization
            init.normal_(self.linear.weight, mean=0.0, std=np.sqrt(1.0 / layers_sizes[i]))
            init.constant_(self.linear.bias, 0.0)

            layers.append(linear)

            if i < len(layers_sizes) - 2:
                layers.append(nn.SELU())

        self.snn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.snn.forward(x)
        return x
