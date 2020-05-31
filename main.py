# -*- coding: utf-8 -*-

import os
import sys
import argparse
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from data_util import *
from evaluate import ArenaEvaluator
from arena_util import write_json


def test_model(question_file_path, answer_file_path, pred_file_path, tag2id_file_path, id2tag_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    question_dataset = SongTagDataset(question_file_path, tag2id_file_path)
    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())

    batch_size = 128
    num_workers = 4

    id2freq_song_dict = dict(np.load('arena_data/orig/id_to_freq_song.npy', allow_pickle=True).item())
    freq_song2id = dict(np.load('arena_data/orig/freq_song_to_id.npy', allow_pickle=True).item())
    num_songs = len(freq_song2id)

    question_data_loader = DataLoader(question_dataset, batch_size=batch_size, num_workers=num_workers)

    try:
        autoencoder = torch.load(model_file_path)
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
            _input = Variable(_input)
            if device.type == 'cuda':
                _input = _input.cuda()

            output = autoencoder(_input)

            _id = list(map(int, _id))
            songs_ids, tags = binary2ids(_input, output, num_songs, id2freq_song_dict, id2tag_dict)
            for i in range(len(_id)):
                element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tags[i]}
                elements.append(element)

    write_json(elements, pred_file_path)
    evaluator.evaluate(answer_file_path, pred_file_path, model_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=100)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.001)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.0)
    parser.add_argument('-num_workers', type=int, help="num workers", default=4)

    args = parser.parse_args()
    print(args)

    H = args.dimension
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers

    question_file_path = 'arena_data/questions/val.json'
    answer_file_path = 'arena_data/answers/val.json'

    tag2id_file_path = 'arena_data/orig/tag_to_id.npy'
    id2tag_file_path = 'arena_data/orig/id_to_file.npy'

    pred_file_path = 'arena_data/answers/pred.json'
    model_file_path = 'model/autoencoder_bce_{}_{}_{}_{}.pkl'.format(H, batch_size, learning_rate, dropout)

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        print('no tag_id file')
        sys.exit(1)

    ## test
    test_model(question_file_path, answer_file_path, pred_file_path, tag2id_file_path, id2tag_file_path, model_file_path)
