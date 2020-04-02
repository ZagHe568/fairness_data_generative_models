import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, n_features):
        super(Generator, self).__init__()
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


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim)

    def forward(self, x):
        outputs, (hidden, cell) = self.rnn(x)
        return outputs, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hid_dim, seq_len):
        super(LSTMDecoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim)
        for i in range(seq_len):
            setattr(self, f'fc_{i}', nn.Linear(hid_dim, 1))
        self.seq_len = seq_len

    def forward(self, hidden, cell, x=None):
        batch_size = hidden.shape[1]
        seq_len = self.seq_len

        if x is not None:
            x_ = torch.zeros_like(x)
            x_[1:] = x[:-1]
            x = x_
            outputs, (_, _) = self.rnn(x, (hidden, cell))
            predictions = []
            for i in range(seq_len):
                output = outputs[i]
                prediction = getattr(self, f'fc_{i}')(output).squeeze()
                predictions.append(prediction)
        else:
            predictions = []
            lstm_input = torch.zeros(1, batch_size, 1, device=hidden.device)
            for i in range(seq_len):
                output, (hidden, cell) = self.rnn(lstm_input, (hidden, cell))
                prediction = getattr(self, f'fc_{i}')(output[0])
                predictions.append(prediction.squeeze())
                lstm_input = prediction.unsqueeze(0)
        predictions = torch.stack(tuple(predictions), dim=0).transpose(0, 1)
        return predictions


class GeneratorLSTM(nn.Module):
    def __init__(self, n_features, hid_dim=128):
        super(GeneratorLSTM, self).__init__()
        self.encoder = LSTMEncoder(1, hid_dim)
        self.decoder = LSTMDecoder(1, hid_dim, n_features)

    def forward(self, x, y=None):
        x = x.transpose(0, 1)
        x = x.unsqueeze(-1)
        _, hidden, cell = self.encoder(x)
        if y is not None:
            y = y.transpose(0, 1)
            y = y.unsqueeze(-1)
        predictions = self.decoder(hidden, cell, y)
        return predictions


class DiscriminatorLSTM(nn.Module):
    def __init__(self, n_features, hid_dim=128):
        super(DiscriminatorLSTM, self).__init__()
        self.encoder = LSTMEncoder(1, hid_dim)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = x.transpose(0, 1)   # [batch_size, seq_len] --> [seq_len, batch_size]
        x = x.unsqueeze(-1)
        outputs, _, _ = self.encoder(x)
        last_output = outputs[-1]   # [batch_size, hid_dim]
        return torch.sigmoid(self.fc(last_output))