import os
import sys
import json
import torch
import io
import os
import copy
import random
import math
import datetime as dt
import distutils.dir_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sentencepiece as spm

from collections import defaultdict
from tqdm import tqdm
from gensim.models import Word2Vec as w2v
from collections import Counter
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from arena_util import write_json, load_json
from w2v import title_tokenizer

vocab_size = 24000
method = 'bpe'


def get_plylsts_embeddings(_train, _question, _submit_type):
    print('saving embeddings')

    # toekenizer model
    tokenizer_name = 'model/tokenizer_{}_{}_{}.model'.format(method, vocab_size, _submit_type)
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_name)
    tt = title_tokenizer()

    # w2v model
    w2v_model_name = 'model/w2v_{}_{}_{}.model'.format(method, vocab_size, _submit_type)
    w2v_model = w2v.load(w2v_model_name)

    # train plylsts to vectors
    t_plylst_title_tag_emb = {}  # plylst_id - vector dictionary
    for plylst in tqdm(_train):
        p_id = plylst['id']
        p_title = plylst['plylst_title']
        p_title_tokens = tt.get_tokens(sp, [p_title])
        if len(p_title_tokens):
            p_title_tokens = p_title_tokens[0]
        else:
            p_title_tokens = []
        p_tags = plylst['tags']
        p_times = plylst['updt_date'][:7].split('-')
        p_words = p_title_tokens + p_tags + p_times
        word_embs = []
        for p_word in p_words:
            try:
                word_embs.append(w2v_model.wv[p_word])
            except KeyError:
                pass
        if len(word_embs):
            p_emb = np.average(word_embs, axis=0).tolist()
        else:
            p_emb = np.zeros(200).tolist()

        t_plylst_title_tag_emb[p_id] = p_emb

    # val plylsts to vectors
    for plylst in tqdm(_question):
        p_id = plylst['id']
        p_title = plylst['plylst_title']
        p_title_tokens = tt.get_tokens(sp, [p_title])
        p_songs = plylst['songs']
        if len(p_title_tokens):
            p_title_tokens = p_title_tokens[0]
        else:
            p_title_tokens = []
        p_tags = plylst['tags']
        p_times = plylst['updt_date'][:7].split('-')
        p_words = p_title_tokens + p_tags + p_times
        word_embs = []
        for p_word in p_words:
            try:
                word_embs.append(w2v_model.wv[p_word])
            except KeyError:
                pass
        if len(word_embs):
            p_emb = np.average(word_embs, axis=0).tolist()
        else:
            p_emb = np.zeros(200).tolist()
        t_plylst_title_tag_emb[p_id] = p_emb

    return t_plylst_title_tag_emb


def save_scores(_train, _question, _autoencoder_embs, _score_type, _submit_type):
    print('saving scores...')

    def pcc(_x, _y):
        vx = _x - torch.mean(_x)
        vy = _y - torch.mean(_y, axis=1).reshape(-1, 1)
        return torch.sum((vx * vy), axis=1) / (
                    torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum((vy ** 2), axis=1)))

    def euclidean(_x, _y):
        return torch.sqrt(torch.sum((_y - _x) ** 2, axis=1))

    all_train_ids = [plylst['id'] for plylst in _train]
    all_val_ids = [plylst['id'] for plylst in _question]

    train_ids = []
    train_embs = []
    val_ids = []
    val_embs = []

    for plylst_id, emb in tqdm(_autoencoder_embs.items()):
        if plylst_id in all_train_ids:
            train_ids.append(plylst_id)
            train_embs.append(emb)
        elif plylst_id in all_val_ids:
            val_ids.append(plylst_id)
            val_embs.append(emb)

    gpu = torch.device('cuda')
    cos = nn.CosineSimilarity(dim=1)
    train_tensor = torch.tensor(train_embs).to(gpu)
    val_tensor = torch.tensor(val_embs).to(gpu)

    scores = torch.zeros([val_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64)
    sorted_idx = torch.zeros([val_tensor.shape[0], train_tensor.shape[0]], dtype=torch.int32)

    for idx, val_vector in enumerate(tqdm(val_tensor)):
        if _score_type == 'pcc':
            output = pcc(val_vector.reshape(1, -1), train_tensor)
        elif _score_type == 'cos':
            output = cos(val_vector.reshape(1, -1), train_tensor)
        elif _score_type == 'euclidean':
            output = euclidean(val_vector.reshape(1, -1), train_tensor)
        index_sorted = torch.argsort(output, descending=True)
        scores[idx] = output
        sorted_idx[idx] = index_sorted

    results = defaultdict(list)
    for i, val_id in enumerate(tqdm(val_ids)):
        for j, train_idx in enumerate(sorted_idx[i][:1000]):
            results[val_id].append((train_ids[train_idx], scores[i][train_idx].item()))

    if _submit_type == 'val':
        np.save(f'scores/val_scores_title_{_score_type}_24000', results)
    elif _submit_type == 'test':
        np.save(f'scores/test_scores_title_{_score_type}_24000', results)
    elif _submit_type == 'local_val':
        np.save(f'scores/local_val_scores_title_{_score_type}_24000', results)
    else:
        np.save(f'scores/test_scores_title_{_score_type}_24000', results)


def get_w2v_scores(submit_type, _retrain):
    if submit_type == 'val':
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        train_file_path = 'res/train.json'
    elif submit_type == 'test':
        default_file_path = 'res'
        val_file_path = 'res/val.json'
        question_file_path = 'res/test.json'
        train_file_path = 'res/train.json'
    elif submit_type == 'local_val':
        default_file_path = 'arena_data'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        default_file_path = f'{default_file_path}/orig'

    genre_file_path = 'res/genre_gn_all.json'

    tokenize_input_file_path = f'model/tokenizer_input_{method}_{vocab_size}_{submit_type}.txt'

    if submit_type == 'local_val':
        val_file_path = None
        test_file_path = None
        train = load_json(train_file_path)
        question = load_json(question_file_path)
    elif submit_type == 'val':
        test_file_path = None
        val_file_path = question_file_path
        train = load_json(train_file_path)
        question = load_json(question_file_path)
    elif submit_type == 'test':
        val_file_path = val_file_path
        test_file_path = question_file_path
        train = load_json(train_file_path)
        val = load_json(val_file_path)
        test = load_json(test_file_path)
        train = train + val
        question = test

    plylst_title_tag_emb = get_plylsts_embeddings(train, question, submit_type)
    save_scores(train, question, plylst_title_tag_emb, 'cos', submit_type)
