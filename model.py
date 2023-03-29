import torch.nn as nn
from conf import CLASSES

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.LSTM(10, 16, 1, batch_first=True)
        self.rnn2 = nn.LSTM(16, 64, 1, batch_first=True)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, len(CLASSES) + 1)
    def forward(self, x):
        out, _ = self.rnn1(x)
        out = nn.Tanh()(out)
        out, _ = self.rnn2(out)
        out = nn.Tanh()(out)
        out = self.linear1(out)
        out = nn.Tanh()(out)
        out = self.linear2(out)
        out = out.transpose(0, 1)
        return out