# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from Models import AutoEncoder
from w2v import train_tokenizer_w2v

from evaluate import ArenaEvaluator
from data_util import tags_ids_convert, save_freq_song_id_dict, binary_songs2ids, binary_tags2ids
from arena_util import load_json, write_json
from tqdm import tqdm


def train(train_dataset, model_file_path, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2prep_song_dict = dict(np.load(id2prep_song_file_path, allow_pickle=True).item())

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags

    # hyper parameters
    D_in = D_out = num_songs + num_tags

    #local_val mode인 경우 중간 중간 결과 확인
    q_data_loader = None
    check_every = 5
    tmp_result_file_path = 'results/tmp_results.json'
    evaluator = ArenaEvaluator()
    if question_dataset is not None:
        q_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model = AutoEncoder(D_in, H, D_out, dropout=dropout).to(device)

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
            _data = _data.to(device)

            optimizer.zero_grad()
            output = model(_data)
            loss = loss_func(output, _data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(model, model_file_path)

        if mode == 0:
            if epoch % check_every == 0:
                if os.path.exists(tmp_result_file_path):
                    os.remove(tmp_result_file_path)
                elements = []
                for idx, (_id, _data) in enumerate(tqdm(q_data_loader, desc='testing...')):
                    with torch.no_grad():
                        _data = _data.to(device)
                        output = model(_data)

                        songs_input, tags_input = torch.split(_data, num_songs, dim=1)
                        songs_output, tags_output = torch.split(output, num_songs, dim=1)

                        songs_ids = binary_songs2ids(songs_input, songs_output, id2prep_song_dict)
                        tag_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

                        _id = list(map(int, _id))
                        for i in range(len(_id)):
                            element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tag_ids[i]}
                            elements.append(element)

                write_json(elements, tmp_result_file_path)
                evaluator.evaluate(answer_file_path, tmp_result_file_path)
                os.remove(tmp_result_file_path)


if __name__ == "__main__":
    # 하이퍼 파라미터 입력
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=450)
    parser.add_argument('-epochs', type=int, help="total epochs", default=41)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=20)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-mode', type=int, help="local_val: 0, val: 1, test: 2", default=2)

    args = parser.parse_args()
    print(args)

    H = args.dimension
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers
    freq_thr = args.freq_thr
    mode = args.mode
    
    # mode에 따른 train dataset과 관련 데이터 로드
    question_data = None
    question_dataset = None
    answer_file_path = None
    if mode == 0: # split data에 대해서는 훈련 중간 중간 성능 확인을 위해서 question, answer 불러옴
        default_file_path = 'arena_data/'
        model_postfix = 'local_val'

        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        answer_file_path = f'{default_file_path}/answers/val.json'

        train_data = load_json(train_file_path)
        question_data = load_json(question_file_path)

    elif mode == 1:
        default_file_path = 'res'
        model_postfix = 'val'

        train_file_path = f'{default_file_path}/train.json'
        val_file_path = f'{default_file_path}/val.json'
        train_data = load_json(train_file_path) + load_json(val_file_path)

    elif mode == 2:
        default_file_path = 'res'
        model_postfix = 'test'

        train_file_path = f'{default_file_path}/train.json'
        val_file_path = f'{default_file_path}/val.json'
        test_file_path = f'{default_file_path}/test.json'
        train_data = load_json(train_file_path) + load_json(val_file_path) + load_json(test_file_path)

    else:
        print('mode error! local_val: 0, val: 1, test: 2')
        sys.exit(1)

    # Autoencoder의 input: song, tag binary vector의 concatenate, tags는 str이므로 id로 변형할 필요 있음
    tag2id_file_path = f'{default_file_path}/tag2id_{model_postfix}.npy'
    id2tag_file_path = f'{default_file_path}/id2tag_{model_postfix}.npy'
    # Song이 너무 많기 때문에 frequency에 기반하여 freq_thr번 이상 등장한 곡들만 남김, 남은 곡들에게 새로운 id 부여
    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr{freq_thr}_{model_postfix}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr{freq_thr}_{model_postfix}.npy'
    # 관련 데이터들이 없으면 default file path에 새로 만들음
    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_data, tag2id_file_path, id2tag_file_path)

    if not (os.path.exists(prep_song2id_file_path) & os.path.exists(id2prep_song_file_path)):
        save_freq_song_id_dict(train_data, freq_thr, default_file_path, model_postfix)

    train_dataset = SongTagDataset(train_data, tag2id_file_path, prep_song2id_file_path)
    if question_data is not None:
        question_dataset = SongTagDataset(question_data, tag2id_file_path, prep_song2id_file_path)

    model_file_path = 'model/autoencoder_{}_{}_{}_{}_{}_{}.pkl'. \
        format(H, batch_size, learning_rate, dropout, freq_thr, model_postfix)
    
    train(train_dataset, model_file_path, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path)

    # w2v 학습 시작
    vocab_size = 24000
    method = 'bpe'
    if model_postfix == 'val':
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        train_file_path = 'res/train.json'
    elif model_postfix == 'test':
        default_file_path = 'res'
        val_file_path = 'res/val.json'
        question_file_path = 'res/test.json'
        train_file_path = 'res/train.json'
    elif model_postfix == 'local_val':
        default_file_path = 'arena_data'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        default_file_path = f'{default_file_path}/orig'

    genre_file_path = 'res/genre_gn_all.json'

    tokenize_input_file_path = f'model/tokenizer_input_{method}_{vocab_size}_{model_postfix}.txt'

    if model_postfix == 'local_val':
        val_file_path = None
        test_file_path = None
        train = load_json(train_file_path)
        question = load_json(question_file_path)
    elif model_postfix == 'val':
        test_file_path = None
        val_file_path = question_file_path
        train = load_json(train_file_path)
        question = load_json(question_file_path)
    elif model_postfix == 'test':
        val_file_path = val_file_path
        test_file_path = question_file_path
        train = load_json(train_file_path)
        val = load_json(val_file_path)
        test = load_json(test_file_path)
        train = train + val
        question = test

    train_tokenizer_w2v(train_file_path, val_file_path, test_file_path, genre_file_path, tokenize_input_file_path,
                        model_postfix)

    print('train completed')
