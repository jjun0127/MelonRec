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
from evaluate import ArenaEvaluator

def train(train_file_path, tag2id_file_path, id2tag_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_dataset = SongTagDataset(train_file_path, tag2id_file_path)
    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2freq_song_dict = dict(np.load('arena_data/orig/id_to_freq_song.npy', allow_pickle=True).item())

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags

    # hyper parameters
    D_in = num_songs + num_tags
    H = 100
    D_out = num_songs + num_tags

    epochs = 10
    batch_size = 128
    learning_rate = 0.001
    num_workers = 4

    evaluator = ArenaEvaluator()
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
        print("\n--------model not restored--------\n")
        pass

    temp_fn = 'results/temp.json'
    if os.path.exists(temp_fn):
        os.remove(temp_fn)

    for epoch in range(epochs):
        print('epoch: ', epoch)
        running_loss = 0.0
        with open(temp_fn, 'a', encoding='utf-8') as json_file:
            json_file.write('[')
            for idx, (_id, _input) in tqdm(enumerate(data_loader)):
                _input = Variable(_input)
                if device.type == 'cuda':
                    _inputs = _inputs.cuda()

                optimizer.zero_grad()

                _output = encoder(_input)
                output = decoder(_output)

                loss = loss_func(output, _input)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if epoch % 100 == 99:
                    print('[%d] loss: %.3f' %(epoch + 1, running_loss / 100))
                    running_loss = 0.0

                _id = list(map(int, _id))
                songs_ids, tags_ids = binary2ids(_input, output, num_songs, id2freq_song_dict)
                tags = ids2tags(tags_ids, id2tag_dict)
                for i in range(len(_id)):
                    element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                    if not (idx == 0 and i == 0):
                        json_file.write(',')
                    json.dump(element, json_file, ensure_ascii=False)
            json_file.write(']')
        evaluator.evaluate(train_file_path, temp_fn)
        os.remove(temp_fn)
        torch.save([encoder, decoder], 'model/deno_autoencoder.pkl')


if __name__ == "__main__":
    train_file_path = 'arena_data/orig/train.json'
    tag2id_file_path = 'arena_data/orig/tag_to_id.npy'
    id2tag_file_path = 'arena_data/orig/id_to_file.npy'

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_file_path, tag2id_file_path, id2tag_file_path)

    train(train_file_path, tag2id_file_path, id2tag_file_path)

    print('train completed')
