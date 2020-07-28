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
        templates = '--input={}         --pad_id=0         --bos_id=1         --eos_id=2         --unk_id=3         --model_prefix={}         --vocab_size={}         --character_coverage=1.0         --model_type={}'

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
                        _submit_type):

    sentences = make_input4tokenizer(_train_file_path, _genre_file_path, _tokenize_input_file_path, _val_file_path,
                                     _test_file_path)

    if not sentences:
        sys.exit(1)

    tokenizer_name = 'model/tokenizer_{}_{}_{}'.format(method, vocab_size, _submit_type)
    tokenizer_name_model = 'model/tokenizer_{}_{}_{}.model'.format(method, vocab_size, _submit_type)
    print("start train_tokenizer...w.")
    train_tokenizer(_tokenize_input_file_path, tokenizer_name, vocab_size, method)
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_name_model)
    tokenized_sentences = get_tokens_from_sentences(sp, sentences)

    w2v_name = 'model/w2v_{}_{}_{}.model'.format(method, vocab_size, _submit_type)
    print("start train_w2v....")
    model = string2vec(tokenized_sentences, size=200, window=5, min_count=1, workers=8, sg=1, hs=1)
    model.save_model(w2v_name)
