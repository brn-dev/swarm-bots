import torch.nn as nn


class SelfNormalizingFNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers_sizes = [input_size] + hidden_sizes + [output_size]

        layers = []
        for i in range(len(layers_sizes) - 1):
            linear = nn.Linear(layers_sizes[i], layers_sizes[i + 1])

            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(linear.bias)

            layers.append(linear)

            if i < len(layers_sizes) - 2:
                layers.append(nn.SELU())

        self.snn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.snn.forward(x)
        return x
