import torch
import torch.nn as nn


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


class AutoEncoder_with_WE(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder_with_WE, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(D_in, H))

        self.word_embedding = nn.Linear(200, H)

        self.input_layer = nn.Sequential(
                            nn.BatchNorm1d(H),
                            nn.ReLU())

        self.decoder = nn.Sequential(
                        nn.Linear(H, D_out),
                        nn.Sigmoid())

    def forward(self, x, w):
        out_encoder = self.encoder(x)
        out_word_embedding = self.word_embedding(w)

        _input = self.input_layer(torch.add(out_encoder, out_word_embedding))

        _output = self.decoder(_input)
        return _output
