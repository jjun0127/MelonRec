# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import collections
import torch


def tags_ids_convert(json_data, tag2id_filepath, id2tag_filepath):
    playlist_df = pd.DataFrame(json_data)
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


def save_freq_song_id_dict(train, thr, default_file_path, model_postfix):
    song_counter = collections.Counter()
    for plylst in train:
        song_counter.update(plylst['songs'])

    selected_songs = []
    song_counter = list(song_counter.items())
    for k, v in song_counter:
        if v > thr:
            selected_songs.append(k)

    print(f'{len(song_counter)} songs to {len(selected_songs)} songs')

    freq_song2id = {song: _id for _id, song in enumerate(selected_songs)}
    np.save(f'{default_file_path}/freq_song2id_thr{thr}_{model_postfix}', freq_song2id)
    print(f'{default_file_path}/freq_song2id_thr{thr}_{model_postfix} is created')
    id2freq_song = {v: k for k, v in freq_song2id.items()}
    np.save(f'{default_file_path}/id2freq_song_thr{thr}_{model_postfix}', id2freq_song)
    print(f'{default_file_path}/id2freq_song_thr{thr}_{model_postfix} is created')


def genre_gn_all_preprocessing(genre_gn_all):
    ## 대분류 장르코드
    # 장르코드 뒷자리 두 자리가 00인 코드를 필터링
    gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] == '00']

    ## 상세 장르코드
    # 장르코드 뒷자리 두 자리가 00이 아닌 코드를 필터링
    dtl_gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] != '00'].copy()
    dtl_gnr_code.rename(columns={'gnr_code': 'dtl_gnr_code', 'gnr_name': 'dtl_gnr_name'}, inplace=True)

    return gnr_code, dtl_gnr_code


def genre_DicGenerator(gnr_code, dtl_gnr_code, song_meta):
    ## gnr_dic (key: 대분류 장르 / value: 대분류 장르 id)
    gnr_dic = {}
    i = 0
    for gnr in gnr_code['gnr_code']:
        gnr_dic[gnr] = i
        i += 1

    ## dtl_dic (key: 상세 장르 / value: 상세 장르 id)
    dtl_dic = {}
    j = 0
    for dtl in dtl_gnr_code['dtl_gnr_code']:
        dtl_dic[dtl] = j
        j += 1

    ## song_gnr_dic (key: 곡 id / value: 해당 곡의 대분류 장르)
    ## song_dtl_dic (key: 곡 id / value: 해당 곡의 상세 장르)
    song_gnr_dic = {}
    song_dtl_dic = {}

    for s in song_meta:
        song_gnr_dic[s['id']] = s['song_gn_gnr_basket']
        song_dtl_dic[s['id']] = s['song_gn_dtl_gnr_basket']

    return gnr_dic, dtl_dic, song_gnr_dic, song_dtl_dic
