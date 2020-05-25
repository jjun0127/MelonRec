# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def tags_ids_convert(train_file_path, tag2id_filepath, id2tag_filepath):
    playlist_df = pd.read_json(train_file_path)
    tags_list = playlist_df['tags'].to_list()
    _id = 0
    tags_dict = dict()
    ids_dict = dict()
    tags_set = set()
    for tags in tags_list:
        for tag in tags:
            if tag not in tags_set:
                tags_set.add(tag)
                tags_dict[tag] = _id
                ids_dict[_id] = tag
                _id += 1
    with open(tag2id_filepath, 'wb') as f:
        np.save(f, tags_dict)
        print('{} is created'.format(tag2id_filepath))
    with open(id2tag_filepath, 'wb') as f:
        np.save(f, ids_dict)
        print('{} is created'.format(id2tag_filepath))

    return True


def binary2ids(_input, output, num_songs):
    _input = _input.detach().numpy()
    output = output.detach().numpy()

    output -= _input
    songs_output, tags_output = np.split(output, [num_songs], axis=1)

    songs_ids = songs_output.argsort(axis=1)[:,::-1][:,:100]
    tags_ids = tags_output.argsort(axis=1)[:,::-1][:,:10]

    return songs_ids.tolist(), tags_ids


def ids2tags(tags_ids, id_to_tag_dict):
    to_id = lambda x: [id_to_tag_dict[_x] for _x in x]

    return list(map(to_id, tags_ids))
