import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(D_in, H),
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU())
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
        self.autoencoder = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(D_in, H),
                            nn.BatchNorm1d(H),
                            nn.LeakyReLU(),
                            nn.Linear(H, D_out),
                            nn.BatchNorm1d(D_out))

        self.word_embedding_decoder = nn.Sequential(
                                        nn.Linear(200, H),
                                        nn.BatchNorm1d(H),
                                        nn.LeakyReLU(),
                                        nn.Linear(H, D_out),
                                        nn.BatchNorm1d(D_out))

        self.sig_layer = nn.Sigmoid()

    def forward(self, x, w):
        out1 = self.autoencoder(x)
        out2 = self.word_embedding_decoder(w)
        out3 = self.sig_layer(torch.add(out1, out2))
        return out3


class AutoEncoder_var(nn.Module):
    def __init__(self, D_in, H, n_songs, n_tags, dropout):
        super(AutoEncoder_var, self).__init__()
        self.autoencoder1 = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(D_in, H),
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU(),
                        nn.Linear(H, n_songs),
                        nn.Sigmoid())

        self.autoencoder2 = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(D_in, H),
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU(),
                        nn.Linear(H, n_tags),
                        nn.Sigmoid())

    def forward(self, x):
        out_autoencoder1 = self.autoencoder1(x)
        out_autoencoder2 = self.autoencoder2(x)
        return out_autoencoder1, out_autoencoder2


class Word2plylst_tag(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Word2plylst_tag, self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(D_in, H),
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU(),
                        nn.Linear(H, D_out),
                        nn.Sigmoid())

    def forward(self, x):
        out = self.layer(x)
        return out

