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


class AutoEncoder_var_song_only(nn.Module):
    def __init__(self, D_in, H, n_songs, dropout):
        super(AutoEncoder_var_song_only, self).__init__()

        encoder = nn.Linear(D_in, H, bias=True)
        decoder = nn.Linear(H, n_songs, bias=True)

        torch.nn.init.xavier_uniform_(encoder.weight)
        torch.nn.init.xavier_uniform_(decoder.weight)

        self.autoencoder = nn.Sequential(
                        nn.Dropout(dropout),
                        encoder,
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU(),
                        decoder,
                        nn.Sigmoid())

    def forward(self, x):
        out = self.autoencoder(x)
        return out


class AutoEncoder_var(nn.Module):
    def __init__(self, D_in, H, n_songs, n_tags, dropout):
        super(AutoEncoder_var, self).__init__()
        H_ae1 = H + 60
        H_ae2 = H - 60
        ae1_layer1 = nn.Linear(D_in, H_ae1, bias=True)
        ae1_layer2 = nn.Linear(H_ae1, n_songs, bias=True)

        ae2_layer1 = nn.Linear(D_in, H_ae2, bias=True)
        ae2_layer2 = nn.Linear(H_ae2, n_tags, bias=True)

        torch.nn.init.xavier_uniform_(ae1_layer1.weight)
        torch.nn.init.xavier_uniform_(ae1_layer2.weight)
        torch.nn.init.xavier_uniform_(ae2_layer1.weight)
        torch.nn.init.xavier_uniform_(ae2_layer2.weight)

        self.autoencoder1 = nn.Sequential(
                        ae1_layer1,
                        nn.BatchNorm1d(H_ae1),
                        nn.LeakyReLU(),
                        ae1_layer2,
                        nn.Sigmoid())

        self.autoencoder2 = nn.Sequential(
                        ae2_layer1,
                        nn.BatchNorm1d(H_ae2),
                        nn.LeakyReLU(),
                        ae2_layer2,
                        nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        out1 = self.autoencoder1(x)
        out2 = self.autoencoder2(x)
        return out1, out2

