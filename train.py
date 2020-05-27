# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from Models import AutoEncoder
from data_util import *
from arena_util import write_json
from evaluate import ArenaEvaluator


def train(train_file_path, tag2id_file_path, id2tag_file_path, question_file_path, answer_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_dataset = SongTagDataset(train_file_path, tag2id_file_path)
    question_dataset = SongTagDataset(question_file_path, tag2id_file_path)
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
    batch_size = 256
    learning_rate = 0.001
    num_workers = 4

    evaluator = ArenaEvaluator()
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    qestion_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    autoencoder = AutoEncoder(D_in, H, D_out).to(device)

    testevery = 5

    parameters = autoencoder.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        autoencoder = torch.load(model_file_path)
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    temp_fn = 'arena_data/answers/temp.json'
    if os.path.exists(temp_fn):
        os.remove(temp_fn)

    for epoch in range(epochs):
        print()
        print('epoch: ', epoch)
        running_loss = 0.0
        for idx, (_id, _input) in tqdm(enumerate(data_loader), desc='training'):
            _input = Variable(_input)
            if device.type == 'cuda':
                _input = _input.cuda()

            optimizer.zero_grad()

            output = autoencoder(_input)

            loss = loss_func(output, _input)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(autoencoder, model_file_path)

        if epoch % testevery == 0:
            elements = []
            for idx, (_id, _input) in tqdm(enumerate(qestion_data_loader), desc='testing...'):
                with torch.no_grad():
                    _input = Variable(_input)
                    if device.type == 'cuda':
                        _input = _input.cuda()

                    output = autoencoder(_input)

                    _id = list(map(int, _id))
                    songs_ids, tags = binary2ids(_input, output, num_songs, id2freq_song_dict, id2tag_dict)
                    for i in range(len(_id)):
                        element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                        elements.append(element)

            write_json(elements, temp_fn)
            evaluator.evaluate(answer_file_path, temp_fn)
            os.remove(temp_fn)


if __name__ == "__main__":
    train_file_path = 'arena_data/orig/train.json'

    question_file_path = 'arena_data/questions/sample_val.json'
    answer_file_path = 'arena_data/answers/sample_val.json'

    tag2id_file_path = 'arena_data/orig/tag_to_id.npy'
    id2tag_file_path = 'arena_data/orig/id_to_file.npy'

    model_file_path = 'model/autoencoder_bce.pkl'

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_file_path, tag2id_file_path, id2tag_file_path)

    train(train_file_path, tag2id_file_path, id2tag_file_path, question_file_path, answer_file_path, model_file_path)

    print('train completed')
