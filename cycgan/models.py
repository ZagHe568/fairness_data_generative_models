import torch.nn as nn


class Generater(nn.Module):
    def __init__(self, n_features):
        super(Generater, self).__init__()
        model = []

        # downsampling layers
        model += [nn.Linear(n_features, 32),
                  nn.ReLU(inplace=True),
                  nn.Linear(32, 16),
                  nn.ReLU(inplace=True),
                  nn.Linear(16, 4),
                  # nn.ReLU(inplace=True)
                  ]

        # upsampling layers
        model += [nn.Linear(4, 16),
                  nn.ReLU(inplace=True),
                  nn.Linear(16, 32),
                  nn.ReLU(inplace=True),
                  nn.Linear(32, n_features),
                  # nn.Sigmoid()
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()
        model = []

        # downsampling layers
        model += [nn.Linear(n_features, 32),
                  nn.ReLU(inplace=True),
                  nn.Linear(32, 16),
                  nn.ReLU(inplace=True),
                  nn.Linear(16, 4),
                  nn.ReLU(inplace=True)]

        # FCN classification layer
        model += [nn.Linear(4, 1),
                  nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x