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


def binary_songs2ids(_input, output, prep_song2id_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()
        
    to_song_id = lambda x: [prep_song2id_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    songs_idxes = output.argsort(axis=1)[:, ::-1][:, :100]

    return list(map(to_song_id, songs_idxes))


def binary_tags2ids(_input, output, id2tag_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()

    to_dict_id = lambda x: [id2tag_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    tags_idxes = output.argsort(axis=1)[:, ::-1][:, :10]

    return list(map(to_dict_id, tags_idxes))


def save_freq_song_id_dict(thr=1, submit=False):
    if submit:
        file_path = 'res'
    else:
        file_path = 'arena_data/orig'

    train = load_json(f'{file_path}/train.json')

    song_counter = collections.Counter()
    for play_list in train:
        song_counter.update(play_list['songs'])

    selected_songs = []
    song_counter = list(song_counter.items())
    for k, v in song_counter:
        if v > thr:
            selected_songs.append(k)

    print(f'{len(song_counter)} songs to {len(selected_songs)} songs')

    freq_song2id = {song: _id for _id, song in enumerate(selected_songs)}
    np.save(f'{file_path}/freq_song2id_thr{thr}', freq_song2id)
    id2freq_song = {v: k for k, v in freq_song2id.items()}
    np.save(f'{file_path}/id2freq_song_thr{thr}', id2freq_song)


def save_liked_song_id_dict(thr=0.5, submit=False):
    if submit:
        file_path = 'res'
    else:
        file_path = 'arena_data/orig'

    train = load_json(f'{file_path}/train.json')

    song_counter = collections.Counter()
    for play_list in train:
        like_cnt = play_list["like_cnt"]
        songs = play_list['songs']
        songs_dict = {song: like_cnt for song in songs}
        song_counter.update(songs_dict)

    sorted_songs = sorted(song_counter.items(), key=lambda x: x[1], reverse=True)
    selected_songs = list(dict(sorted_songs[:int(len(sorted_songs)*thr)]).keys())
    print(f'{len(sorted_songs)} songs to {len(selected_songs)} songs')

    liked_song2id = {song: _id for _id, song in enumerate(selected_songs)}
    np.save(f'{file_path}/liked_song2id_thr{thr}', liked_song2id)
    id2liked_song = {v: k for k, v in liked_song2id.items()}
    np.save(f'{file_path}/id2liked_song_thr{thr}', id2liked_song)


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
    save_freq_song_id_dict(thr=2, submit=True)
    # save_freq_song_id_dict(thr=3)
    # save_liked_song_id_dict(thr=0.3)
