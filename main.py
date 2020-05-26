# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from Models import Encoder, Decoder
from data_util import *
import json
import time
from evaluate import ArenaEvaluator


def test_model(test_file_path, gt_file_path, pred_file_path, tag2id_file_path, id2tag_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    test_dataset = SongTagDataset(test_file_path, tag2id_file_path)
    id_to_tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())

    batch_size = 128
    num_workers = 4

    freq_song_to_id = dict(np.load('arena_data/orig/freq_song_to_id.npy', allow_pickle=True).item())
    num_songs = len(freq_song_to_id)

    data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    encoder, decoder = torch.load('model/deno_autoencoder.pkl')

    evaluator = ArenaEvaluator()

    if os.path.exists(pred_file_path):
        os.remove(pred_file_path)

    with open(pred_file_path, 'a', encoding='utf-8') as json_file:
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
                    if not (idx == 0 and i == 0):
                        json_file.write(',')
                    json.dump(element, json_file, ensure_ascii=False)
        json_file.write(']')
    evaluator.evaluate(gt_file_path, pred_file_path)


if __name__ == "__main__":
    test_file_path = 'arena_data/questions/val.json'

    gt_file_path = 'arena_data/answers/val.json'
    pred_file_path = 'results/pred.json'

    tag2id_file_path = 'arena_data/orig/tag_to_id.npy'
    id2tag_file_path = 'arena_data/orig/id_to_file.npy'

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        print('no tag_id file')
        sys.exit(1)

    ## test
    test_model(test_file_path, gt_file_path, pred_file_path, tag2id_file_path, id2tag_file_path)
