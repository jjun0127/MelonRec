import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        encoder_layer = nn.Linear(D_in, H, bias=True)
        decoder_layer = nn.Linear(H, D_out, bias=True)

        torch.nn.init.xavier_uniform_(encoder_layer.weight)
        torch.nn.init.xavier_uniform_(decoder_layer.weight)

        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        encoder_layer,
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU())
        self.decoder = nn.Sequential(
                        decoder_layer,
                        nn.Sigmoid())

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder

