import torch
import torch.nn as nn
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(D_in, H),
                        nn.BatchNorm1d(H),
                        nn.ReLU())
        self.decoder = nn.Sequential(
                        nn.Linear(H, D_out),
                        nn.Sigmoid())

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder