# -*- coding: utf-8 -*-

import os
import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset, SongTagDataset_with_WE
from Models import AutoEncoder, AutoEncoder_with_WE
from data_util import *
from arena_util import write_json
from evaluate import ArenaEvaluator


def train_type1(train_dataset, question_dataset, id2tag_file_path, answer_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2freq_song_dict = dict(np.load('arena_data/orig/id_to_freq_song.npy', allow_pickle=True).item())

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags

    # hyper parameters
    D_in = num_songs + num_tags
    D_out = num_songs + num_tags

    evaluator = ArenaEvaluator()
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    qestion_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model = AutoEncoder(D_in, H, D_out, dropout=dropout).to(device)

    testevery = 5

    parameters = model.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        model = torch.load(model_file_path)
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
        for idx, (_id, _data) in enumerate(tqdm(data_loader, desc='training...')):

            _data = Variable(_data)
            if device.type == 'cuda':
                _data = _data.cuda()

            optimizer.zero_grad()

            output = model(_data)

            loss = loss_func(output, _data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(model, model_file_path)

        if epoch % testevery == 0:
            elements = []
            for idx, (_id, _data) in enumerate(tqdm(qestion_data_loader, desc='testing...')):
                with torch.no_grad():
                    _data = Variable(_data)
                    if device.type == 'cuda':
                        _data = _data.cuda()

                    output = model(_data)

                    _id = list(map(int, _id))
                    songs_ids, tags = binary2ids(_data, output, num_songs, id2freq_song_dict, id2tag_dict)
                    for i in range(len(_id)):
                        element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                        elements.append(element)

            write_json(elements, temp_fn)
            evaluator.evaluate(answer_file_path, temp_fn)
            os.remove(temp_fn)


def train_type2(train_dataset, question_dataset, id2tag_file_path, answer_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2freq_song_dict = dict(np.load('arena_data/orig/id_to_freq_song.npy', allow_pickle=True).item())

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags

    # hyper parameters
    D_in = num_songs + num_tags
    D_out = num_songs + num_tags

    evaluator = ArenaEvaluator()
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    qestion_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model = AutoEncoder_with_WE(D_in, H, D_out, dropout=dropout).to(device)

    testevery = 5

    parameters = model.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        model = torch.load(model_file_path)
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
        for idx, (_id, _data, _we) in enumerate(tqdm(data_loader, desc='training...')):
            _data = Variable(_data)
            _we = Variable(_we)

            if device.type == 'cuda':
                _data = _data.cuda()
                _we = _we.cuda()

            optimizer.zero_grad()

            output = model(_data, _we.float())

            loss = loss_func(output, _data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(model, model_file_path)

        if epoch % testevery == 0:
            elements = []
            for idx, (_id, _data, _we)  in enumerate(tqdm(qestion_data_loader, desc='testing...')):
                with torch.no_grad():
                    _data = Variable(_data)
                    _we = Variable(_we)

                    if device.type == 'cuda':
                        _data = _data.cuda()
                        _we = _we.cuda()

                    output = model(_data, _we.float())

                    _id = list(map(int, _id))
                    songs_ids, tags = binary2ids(_data, output, num_songs, id2freq_song_dict, id2tag_dict)
                    for i in range(len(_id)):
                        element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                        elements.append(element)

            write_json(elements, temp_fn)
            evaluator.evaluate(answer_file_path, temp_fn)
            os.remove(temp_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', type=int, help="model selection, basic AE: 1, using word embedding: 2", default=1)
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=100)
    parser.add_argument('-epochs', type=int, help="total epochs", default=10)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.001)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.0)
    parser.add_argument('-num_workers', type=int, help="num workers", default=4)

    args = parser.parse_args()
    print(args)

    model_type = args.model_type
    H = args.dimension
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers

    train_file_path = 'arena_data/orig/train.json'

    question_file_path = 'arena_data/questions/sample_val.json'
    answer_file_path = 'arena_data/answers/sample_val.json'

    tag2id_file_path = 'arena_data/orig/tag_to_id.npy'
    id2tag_file_path = 'arena_data/orig/id_to_file.npy'
    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_file_path, tag2id_file_path, id2tag_file_path)

    if model_type == 1:
        model_file_path = 'model/autoencoder_{}_{}_{}_{}.pkl'.format(H, batch_size, learning_rate, dropout)

        train_dataset = SongTagDataset(train_file_path, tag2id_file_path)
        question_dataset = SongTagDataset(question_file_path, tag2id_file_path)

        train_type1(train_dataset, question_dataset, id2tag_file_path, answer_file_path, model_file_path)
    elif model_type == 2:
        wv_file_path = 'model/wv/w2v_bpe_100000.model'
        tokenizer_file_path = 'model/tokenizer/tokenizer_bpe_100000.model'
        model_file_path = 'model/autoencoder_with_we_{}_{}_{}_{}.pkl'.format(H, batch_size, learning_rate, dropout)

        train_dataset = SongTagDataset_with_WE(train_file_path, tag2id_file_path, wv_file_path, tokenizer_file_path)
        question_dataset = SongTagDataset_with_WE(question_file_path, tag2id_file_path, wv_file_path, tokenizer_file_path)

        train_type2(train_dataset, question_dataset, id2tag_file_path, answer_file_path, model_file_path)

    print('train completed')
