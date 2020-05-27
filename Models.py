import torch
import torch.nn as nn
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Linear(D_in, H),
                        nn.ReLU())
        self.decoder = nn.Sequential(
                        nn.Linear(H, D_out),
                        nn.Softmax())

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder