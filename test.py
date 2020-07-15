# -*- coding: utf-8 -*-

import os
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from data_util import *
from evaluate import ArenaEvaluator
from arena_util import write_json


def test(question_dataset, answer_file_path, pred_file_path, id2prep_song_file_path, id2tag_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2prep_song_dict = dict(np.load(id2prep_song_file_path, allow_pickle=True).item())
    num_songs = len(id2prep_song_dict)

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
            output = model(_data)

            songs_input, tags_input = torch.split(_data, num_songs, dim=1)
            songs_output, tags_output = torch.split(output, num_songs, dim=1)
            songs_ids = binary_songs2ids(songs_input, songs_output, id2prep_song_dict)
            tag_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

            _id = list(map(int, _id))
            for i in range(len(_id)):
                element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tag_ids[i]}
                elements.append(element)

    write_json(elements, pred_file_path)
    if not submit:
        evaluator.evaluate_with_save(answer_file_path, pred_file_path, model_file_path, default_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=520)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=10)
    parser.add_argument('-prep_method_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-submit', type=int, help="arena_data/arig: 0 res: 1", default=0)

    args = parser.parse_args()
    print(args)

    H = args.dimension
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers
    prep_method_thr = args.prep_method_thr
    submit = args.submit

    if submit:
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        model_postfix = '_sub'
    else:
        default_file_path = 'arena_data/orig'
        question_file_path = 'arena_data/questions/sample_val.json'
        model_postfix = ''

    train_file_path = f'{default_file_path}/train.json'

    answer_file_path = 'arena_data/answers/sample_val.json'

    tag2id_file_path = f'{default_file_path}/tag2id.npy'
    id2tag_file_path = f'{default_file_path}/id2tag.npy'

    prep_methods = ['freq_song', 'liked_song']
    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr{prep_method_thr}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr{prep_method_thr}.npy'

    pred_file_path = f'{default_file_path}/results.json'

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        print('no tag_id file')
        sys.exit(1)

    model_file_path = 'model/autoencoder_{}_{}_{}_{}_{}{}.pkl'. \
            format(H, batch_size, learning_rate, dropout, prep_method_thr, model_postfix)

    question_dataset = SongTagDataset(question_file_path, tag2id_file_path, prep_song2id_file_path)

    ## test
    test(question_dataset, answer_file_path, pred_file_path, id2prep_song_file_path, id2tag_file_path, model_file_path)

    print('test completed')

