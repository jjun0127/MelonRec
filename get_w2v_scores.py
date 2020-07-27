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

vocab_size = 24000
method = 'bpe'


def load_json(fname):
    with open(fname, encoding='utf8') as f:
        json_obj = json.load(f)

    return json_obj


def make_input4tokenizer(train_file_path, genre_file_path, result_file_path, valid_file_path=None, test_file_path=None):
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
        plylsts = load_json(train_file_path)
        if valid_file_path is not None:
            val_plylsts = load_json(valid_file_path)
            plylsts += val_plylsts
        if test_file_path is not None:
            test_plylsts = load_json(test_file_path)
            plylsts += test_plylsts

        genre_all = load_json(genre_file_path)
        genre_all_lists = []
        for code, gnr in genre_all.items():
            if gnr != '세부장르전체':
                genre_all_lists.append([code, gnr])
        genre_all_lists = np.asarray(genre_all_lists)

        sentences = []
        for plylst in plylsts:
            tiS = plylst['plylst_title']
            taS = ' '.join(plylst['tags'])
            upS = ' '.join(plylst['updt_date'][:7].split('-'))
            sentences.append(' '.join([tiS, taS, upS]))

        geS = _wv_genre(genre_all_lists)
        sentences = sentences + geS
        with open(result_file_path, 'w', encoding='utf8') as f:
            for sentence in sentences:
                f.write(sentence+'\n')
    except Exception as e:
        print(e.with_traceback())
        return False
    return sentences


def train_tokenizer(input_file_path, model_file_path, vocab_size, model_type):
    templates = ' --input={} \
        --pad_id=0 \
        --bos_id=1 \
        --eos_id=2 \
        --unk_id=3 \
        --model_prefix={} \
        --vocab_size={} \
        --character_coverage=1.0 \
        --model_type={}'

    cmd = templates.format(input_file_path,
                model_file_path,    # output model 이름
                vocab_size,# 작을수록 문장을 잘게 쪼갬
                model_type)# unigram (default), bpe, char

    spm.SentencePieceTrainer.Train(cmd)
    print("tokenizer {} is generated".format(model_file_path))


def get_tokens_from_sentences(sp, sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = sp.EncodeAsPieces(sentence)
        new_tokens = []
        for token in tokens:
            token = token.replace("▁", "")
            if len(token) > 1:
                new_tokens.append(token)
        if len(new_tokens) > 1:
            tokenized_sentences.append(new_tokens)

    return tokenized_sentences


def get_tokens_from_sentence(sp, sentence):
    new_tokens = []
    tokens = sp.EncodeAsPieces(sentence)
    for token in tokens:
        token = token.replace("▁", "")
        if len(token) > 1:
            new_tokens.append(token)
    return new_tokens


class string2vec():
    def __init__(self, train_data, size=200, window=5, min_count=2, workers=8, sg=1, hs=1):
        self.model = w2v(train_data, size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)

    def set_model(self, model_fn):
        self.model = w2v.load(model_fn)

    def save_embeddings(self, emb_fn):
        word_vectors = self.model.wv

        vocabs = []
        vectors = []
        for key in word_vectors.vocab:
            vocabs.append(key)
            vectors.append(word_vectors[key])

        df = pd.DataFrame()
        df['voca'] = vocabs
        df['vector'] = vectors

        df.to_csv(emb_fn,index=False)

    def save_model(self, md_fn):
        self.model.save(md_fn)
        print("word embedding model {} is trained".format(md_fn))

    def show_similar_words(self,word, topn):
        print(self.model.most_similar(positive=[word], topn=topn))


class title_tokenizer():
    def make_input_file(self, input_fn, sentences):
        with open(input_fn, 'w', encoding='utf8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')

    def train_tokenizer(self, input_fn, prefix, vocab_size, model_type):
        templates = '--input={}         --pad_id=0         --bos_id=1         --eos_id=2         --unk_id=3         ' \
                    '--model_prefix={}         --vocab_size={}         --character_coverage=1.0         ' \
                    '--model_type={} '

        cmd = templates.format(input_fn,
                               prefix,  # output model 이름
                               vocab_size,  # 작을수록 문장을 잘게 쪼갬
                               model_type)  # unigram (default), bpe, char

        spm.SentencePieceTrainer.Train(cmd)
        print("tokenizer model {} is trained".format(prefix + ".model"))

    def get_tokens(self, sp, sentences):
        tokenized_sentences = []

        for sentence in sentences:
            tokens = sp.EncodeAsPieces(sentence)
            new_tokens = []
            for token in tokens:
                token = token.replace("▁", "")
                if len(token) > 1:
                    new_tokens.append(token)
            if len(new_tokens) > 1:
                tokenized_sentences.append(new_tokens)

        return tokenized_sentences


def train_tokenizer_w2v(_train_file_path, _val_file_path, _test_file_path, _genre_file_path, _tokenize_input_file_path,
                        _submit_type, _retrain):

    sentences = make_input4tokenizer(_train_file_path, _genre_file_path, _tokenize_input_file_path, _val_file_path,
                                     _test_file_path)

    if not sentences:
        sys.exit(1)
    tokenizer_name = 'model/tokenizer_{}_{}_{}.model'.format(method, vocab_size, _submit_type)
    if _retrain:
        print("start train_tokenizer....")
        train_tokenizer(_tokenize_input_file_path, tokenizer_name, vocab_size, method)
    else:
        if not os.path.exists(tokenizer_name):
            print("start train_tokenizer....")
            train_tokenizer(_tokenize_input_file_path, tokenizer_name, vocab_size, method)

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_name)

    tokenized_sentences = get_tokens_from_sentences(sp, sentences)

    w2v_name = 'model/w2v_{}_{}_{}.model'.format(method, vocab_size, _submit_type)
    if _retrain:
        print("start train_w2v....")
        model = string2vec(tokenized_sentences, size=200, window=5, min_count=1, workers=8, sg=1, hs=1)
        model.save_model(w2v_name)
    else:
        if not os.path.exists(w2v_name):
            print("start train_w2v....")
            model = string2vec(tokenized_sentences, size=200, window=5, min_count=1, workers=8, sg=1, hs=1)
            model.save_model(w2v_name)


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
        question = load_json(question_file_path)
    elif submit_type == 'val':
        test_file_path = None
        val_file_path = question_file_path
        question = load_json(question_file_path)
    elif submit_type == 'test':
        val_file_path = val_file_path
        test_file_path = question_file_path
        train = load_json(train_file_path)
        val = load_json(val_file_path)
        test = load_json(test_file_path)
        train = train + val
        question = test

    train_tokenizer_w2v(train_file_path, val_file_path, test_file_path, genre_file_path, tokenize_input_file_path,
                        submit_type, _retrain)
    plylst_title_tag_emb = get_plylsts_embeddings(train, question, submit_type)
    save_scores(train, question, plylst_title_tag_emb, 'cos', submit_type)