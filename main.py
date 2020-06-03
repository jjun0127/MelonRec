# -*- coding: utf-8 -*-

import os
import sys
import argparse
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset, SongTagDataset_with_WE
from data_util import *
from evaluate import ArenaEvaluator
from arena_util import write_json


def test_type1(question_dataset, answer_file_path, pred_file_path, id2tag_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2freq_song_dict = dict(np.load(f'{default_file_path}/id2freq_song.npy', allow_pickle=True).item())
    freq_song2id = dict(np.load(f'{default_file_path}/freq_song2id.npy', allow_pickle=True).item())
    num_songs = len(freq_song2id)

    question_data_loader = DataLoader(question_dataset, batch_size=batch_size, num_workers=num_workers)

    try:
        model = torch.load(model_file_path)
        print("\n--------model loaded--------\n")
    except:
        print("\n--------model not found--------\n")
        sys.exit(1)

    evaluator = ArenaEvaluator()

    if os.path.exists(pred_file_path):
        os.remove(pred_file_path)

    elements = []
    for idx, (_id, _input) in enumerate(tqdm(question_data_loader, desc='testing...')):
        with torch.no_grad():
            _input = _input.to(device)

            output = model(_input)

            _id = list(map(int, _id))
            songs_ids, tags = binary2ids(_input, output, num_songs, id2freq_song_dict, id2tag_dict)
            for i in range(len(_id)):
                element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                elements.append(element)

    write_json(elements, pred_file_path)
    if not submit:
        evaluator.evaluate_with_save(answer_file_path, pred_file_path, model_file_path, default_file_path)


def test_type2(question_dataset, answer_file_path, pred_file_path, id2tag_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2freq_song_dict = dict(np.load(f'{default_file_path}/id2freq_song.npy', allow_pickle=True).item())
    freq_song2id = dict(np.load(f'{default_file_path}/freq_song2id.npy', allow_pickle=True).item())
    num_songs = len(freq_song2id)

    question_data_loader = DataLoader(question_dataset, batch_size=batch_size, num_workers=num_workers)

    try:
        model = torch.load(model_file_path)
        print("\n--------model loaded--------\n")
    except:
        print("\n--------model not found--------\n")
        sys.exit(1)

    evaluator = ArenaEvaluator()

    if os.path.exists(pred_file_path):
        os.remove(pred_file_path)

    elements = []
    for idx, (_id, _data, _we) in enumerate(tqdm(question_data_loader, desc='testing...')):
        with torch.no_grad():
            _data = _data.to(device)
            _we = _we.to(device)

            output = model(_data, _we.float())

            _id = list(map(int, _id))
            songs_ids, tags = binary2ids(_data, output, num_songs, id2freq_song_dict, id2tag_dict)
            for i in range(len(_id)):
                element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                elements.append(element)

    write_json(elements, pred_file_path)
    if not submit:
        evaluator.evaluate_with_save(answer_file_path, pred_file_path, model_file_path, default_file_path)


def test_type3(question_dataset, answer_file_path, pred_file_path, id2tag_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2freq_song_dict = dict(np.load(f'{default_file_path}/id2freq_song.npy', allow_pickle=True).item())
    freq_song2id = dict(np.load(f'{default_file_path}/freq_song2id.npy', allow_pickle=True).item())
    num_songs = len(freq_song2id)

    question_data_loader = DataLoader(question_dataset, batch_size=batch_size, num_workers=num_workers)

    try:
        model = torch.load(model_file_path)
        print("\n--------model loaded--------\n")
    except:
        print("\n--------model not found--------\n")
        sys.exit(1)

    evaluator = ArenaEvaluator()

    if os.path.exists(pred_file_path):
        os.remove(pred_file_path)

    elements = []
    for idx, (_id, _data) in enumerate(tqdm(question_data_loader, desc='testing...')):
        with torch.no_grad():
            _data = _data.to(device)
            output1, output2 = model(_data)

            _id = list(map(int, _id))
            songs_ids, tags = binary2ids(_data, torch.cat((output1, output2), 1), num_songs, id2freq_song_dict, id2tag_dict)
            for i in range(len(_id)):
                element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                elements.append(element)

    write_json(elements, pred_file_path)
    if not submit:
        evaluator.evaluate_with_save(answer_file_path, pred_file_path, model_file_path, default_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', type=int, help="model selection, basic AE: 1, using word embedding: 2", default=1)
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=100)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.001)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.0)
    parser.add_argument('-num_workers', type=int, help="num workers", default=4)
    parser.add_argument('-submit', type=int, help="arena_data/arig: 0 res: 1", default=0)

    args = parser.parse_args()
    print(args)

    model_type = args.model_type
    H = args.dimension
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers
    submit = args.submit

    if submit:
        default_file_path = 'res'
    else:
        default_file_path = 'arena_data/orig'

    train_file_path = f'{default_file_path}/train.json'

    question_file_path = f'{default_file_path}/val.json'
    answer_file_path = 'arena_data/answers/sample_val.json'

    tag2id_file_path = f'{default_file_path}/tag2id.npy'
    id2tag_file_path = f'{default_file_path}/id2tag.npy'
    freq_song2id_file_path = f'{default_file_path}/freq_song2id.npy'

    pred_file_path = f'{default_file_path}/results.json'

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        print('no tag_id file')
        sys.exit(1)

    if model_type == 1:
        model_file_path = 'model/autoencoder_{}_{}_{}_{}_submit.pkl'.format(H, batch_size, learning_rate, dropout)

        question_dataset = SongTagDataset(question_file_path, tag2id_file_path, freq_song2id_file_path)

        ## test
        test_type1(question_dataset, answer_file_path, pred_file_path, id2tag_file_path, model_file_path)

    elif model_type == 2:
        wv_file_path = 'model/wv/w2v_bpe_100000.model'
        tokenizer_file_path = 'model/tokenizer/tokenizer_bpe_100000.model'
        model_file_path = 'model/autoencoder_with_we_{}_{}_{}_{}_submit.pkl'.format(H, batch_size, learning_rate, dropout)

        question_dataset = SongTagDataset_with_WE(question_file_path, tag2id_file_path, freq_song2id_file_path,
                                                  wv_file_path, tokenizer_file_path)

        test_type2(question_dataset, answer_file_path, pred_file_path, id2tag_file_path, model_file_path)

    elif model_type == 3:
        model_file_path = 'model/autoencoder_var{}_{}_{}_{}_submit.pkl'.format(H, batch_size, learning_rate, dropout)

        question_dataset = SongTagDataset(question_file_path, tag2id_file_path, freq_song2id_file_path)

        ## test
        test_type3(question_dataset, answer_file_path, pred_file_path, id2tag_file_path, model_file_path)
