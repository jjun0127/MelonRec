# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import collections
import torch
from arena_util import load_json


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


def binary2ids(_input, output, num_songs, freq_song2id_dict, id2tag_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()
        
    to_song_id = lambda x: [freq_song2id_dict[_x] for _x in x]
    to_dict_id = lambda x: [id2tag_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    songs_output, tags_output = np.split(output, [num_songs], axis=1)

    songs_idxes = songs_output.argsort(axis=1)[:, ::-1][:, :100]
    tags_idxes = tags_output.argsort(axis=1)[:, ::-1][:, :10]

    return list(map(to_song_id, songs_idxes)), list(map(to_dict_id, tags_idxes))


def save_freq_song_id_dict():
    # freq_song_to_id, id_to_freq_song
    train = load_json('res/train.json')

    song_counter = collections.Counter()
    for play_list in train:
        song_counter.update(play_list['songs'])

    selected_songs = []
    for k, v in song_counter.items():
        if v > 1:
            selected_songs.append(k)

    freq_song_to_id = {song: _id for _id, song in enumerate(selected_songs)}
    np.save('res/freq_song2id', freq_song_to_id)
    id_to_freq_song = {v: k for k, v in freq_song_to_id.items()}
    np.save('res/id2freq_song', id_to_freq_song)


def make_input4tokenizer(playlist_file_path, genre_file_path, result_file_path):
    def _wv_tags(tags_list):
        taS = []
        for tags in tags_list:
            taS.append(' '.join(tags))

        return(taS)

    def _wv_genre(genre):
        genre_dict = dict()
        for code, value in genre:
            code_num = int(code[2:])
            if not code_num % 100:
                cur_genre = value
                genre_dict[cur_genre] = []
            else:
                value = ' '.join(value.split('/'))
                genre_dict[cur_genre].append(value)
        genre_sentences = []
        for key in genre_dict:
            sub_list = genre_dict[key]
            key = ' '.join(key.split('/'))
            if not len(sub_list):
                continue
            for sub in sub_list:
                genre_sentences.append(key+' '+sub)
        return genre_sentences

    try:
        playlist_df = pd.read_json(playlist_file_path)
        genre_df = pd.read_json(genre_file_path, orient='index').reset_index()
        tiS = playlist_df['plylst_title'].tolist()
        taS = _wv_tags(playlist_df['tags'].to_numpy())
        geS = _wv_genre(genre_df.to_numpy())

        sentences = tiS + taS + geS
        with open(result_file_path, 'w', encoding='utf8') as f:
            for sentence in sentences:
                f.write(sentence+'\n')
    except Exception as e:
        print(e.with_traceback())
        return False
    print('{} is generated'.format(result_file_path))
    return sentences


if __name__ == '__main__':
    save_freq_song_id_dict()