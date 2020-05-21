import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from MelonDataset import SongTagDataset
from Models import Encoder, Decoder


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    songtag_dataset = SongTagDataset()

    dataloader = DataLoader(songtag_dataset, batch_size=128, num_workers=4)
    # parameters
    num_songs = 707989
    num_tags = 29160

    # hyper parameters
    D_in = num_songs + num_tags
    H = 100
    D_out = num_songs + num_tags

    epoch = 10
    batch_size = 128
    learning_rate = 0.001

    encoder = Encoder(D_in, H).to(device)
    decoder = Decoder(D_out, H).to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # for _ids, _inputs in dataloader:
    #     print(len(_ids))
    #     print(_inputs.size())
    #     break
    try:
        encoder, decoder = torch.load('model/deno_autoencoder.pkl')
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    for i in range(epoch):
        print('epoch: ', i)
        running_loss = 0.0
        for _input in tqdm(dataloader):
            _input = Variable(_input)
            if device.type == 'cuda':
                _input = _input.cuda()

            optimizer.zero_grad()

            output = encoder(_input)
            output = decoder(output)

            loss = loss_func(output, _input)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            # if i % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %(epoch + 1, j + 1, running_loss / 100))
            #     running_loss = 0.0

        torch.save([encoder, decoder], 'model/deno_autoencoder.pkl')
        print(loss)