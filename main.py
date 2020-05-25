# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from Models import Encoder, Decoder
from data_util import *
import json


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    train_file_path = 'arena_data/orig/train.json'
    test_file_path = 'arena_data/orig/test.json'
    tag_id_file_path = 'arena_data/orig/tag_to_id.npy'
    id_tag_file_path = 'arena_data/orig/id_to_file.npy'

    if not (os.path.isdir(tag_id_file_path) & os.path.isdir(id_tag_file_path)):
        tags_ids_convert(train_file_path, tag_id_file_path, id_tag_file_path)
    id_to_tag_dict = dict(np.load(id_tag_file_path, allow_pickle=True).item())

    ##train_code
    train_dataset = SongTagDataset(train_file_path, tag_id_file_path)
    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags

    # hyper parameters
    D_in = num_songs + num_tags
    H = 100
    D_out = num_songs + num_tags

    epoch = 10
    batch_size = 128
    learning_rate = 0.001
    num_workers = 4

    #train
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    encoder = Encoder(D_in, H).to(device)
    decoder = Decoder(D_out, H).to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        encoder, decoder = torch.load('model/deno_autoencoder.pkl')
        print("\n--------model restored--------\n")
    except:
        print("\n--------model nost restored--------\n")
        pass

    for i in range(epoch):
        print('epoch: ', i)
        running_loss = 0.0
        for (_ids, _inputs) in tqdm(data_loader):
            _inputs = Variable(_inputs)
            if device.type == 'cuda':
                _inputs = _inputs.cuda()

            optimizer.zero_grad()

            output = encoder(_inputs)
            output = decoder(output)

            loss = loss_func(output, _inputs)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            # if i % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %(epoch + 1, j + 1, running_loss / 100))
            #     running_loss = 0.0

        torch.save([encoder, decoder], 'model/deno_autoencoder.pkl')
        print(loss)
    print('train completed')
          
    ## Test code
    test_dataset = SongTagDataset(test_file_path, tag_id_file_path)
    data_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    encoder, decoder = torch.load('model/deno_autoencoder.pkl')
    with open('results.json', 'a', encoding='utf-8') as json_file:
        json_file.write('[')
        for idx, (_id, _input) in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                _input = Variable(_input)

                _output = encoder(_input)
                output = decoder(_output)

                _id = list(map(int, _id))
                songs_ids, tags_ids = binary2ids(_input, output, num_songs)
                tags = ids2tags(tags_ids, id_to_tag_dict)
                for i in range(len(_id)):
                    element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                    if idx != 0 | i != 0:
                        json_file.write(',')
                    json.dump(element, json_file, ensure_ascii=False)
        json_file.write(']')
