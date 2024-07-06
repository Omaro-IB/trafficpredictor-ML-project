import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(device)

        # out = output features of hidden states from all time-steps
        out, _ = self.rnn(x, h0)  # size: batch_size x sequence_length x hidden_size = Nx17x128
        out = out[:, -1, :]  # size: batch_size x hidden_size = Nx128
        out = self.fc(out)
        return out
