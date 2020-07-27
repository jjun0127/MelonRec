import random
import torch.nn as nn
import sentencepiece as spm
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset, SongTagGenreDataset
from data_util import *
from arena_util import write_json, load_json
from evaluate import ArenaEvaluator
from collections import Counter, defaultdict
from Models import AutoEncoder

from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

random.seed(777)
np.random.seed(777)


def get_plylsts_embeddings(_model_file_path, _submit_type, genre=False):
    if _submit_type == 'val':
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        train_file_path = 'res/train.json'
    elif _submit_type == 'test':
        default_file_path = 'res'
        question_file_path = 'res/test.json'
        train_file_path = 'res/train.json'
    elif _submit_type == 'local_val':
        default_file_path = 'arena_data'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        default_file_path = f'{default_file_path}/orig'

    tag2id_file_path = f'{default_file_path}/tag2id_{_submit_type}.npy'
    id2tag_file_path = f'{default_file_path}/id2tag_{_submit_type}.npy'
    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr2_{_submit_type}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr2_{_submit_type}.npy'

    if genre:
        train_dataset = SongTagGenreDataset(load_json(train_file_path), tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagGenreDataset(load_json(question_file_path), tag2id_file_path, prep_song2id_file_path)
    else:
        train_dataset = SongTagDataset(load_json(train_file_path), tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagDataset(load_json(question_file_path), tag2id_file_path, prep_song2id_file_path)

    plylst_embed_weight = []
    plylst_embed_bias = []

    model_file_path = _model_file_path

    model = torch.load(model_file_path)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'encoder.1.weight':
                plylst_embed_weight = param.data
            elif name == 'encoder.1.bias':
                plylst_embed_bias = param.data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=4)
    question_loader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plylst_emb_with_bias = dict()

    if genre:
        for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(train_loader, desc='get train vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]

        for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(question_loader, desc='get question vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]
    else:
        for idx, (_id, _data) in enumerate(tqdm(train_loader, desc='get train vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]

        for idx, (_id, _data) in enumerate(tqdm(question_loader, desc='get question vectors...')):
            with torch.no_grad():
                _data = _data.to(device)
                output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                _id = list(map(int, _id))
                for i in range(len(_id)):
                    plylst_emb_with_bias[_id[i]] = output_with_bias[i]
    return plylst_emb_with_bias


def save_scores(_autoencoder_embs, _score_type, _submit_type, genre=False):
    if _submit_type == 'val':
        question_file_path = 'res/val.json'
        train_file_path = 'res/train.json'
    elif _submit_type == 'test':
        question_file_path = 'res/test.json'
        train_file_path = 'res/train.json'
    elif _submit_type == 'local_val':
        default_file_path = 'arena_data'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'

    _train = load_json(train_file_path)
    _val = load_json(question_file_path)

    def pcc(_x, _y):
        vx = _x - torch.mean(_x)
        vy = _y - torch.mean(_y, axis=1).reshape(-1, 1)
        return torch.sum((vx * vy), axis=1) / (
                    torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum((vy ** 2), axis=1)))

    def euclidean(_x, _y):
        return torch.sqrt(torch.sum((_y - _x) ** 2, axis=1))

    all_train_ids = [plylst['id'] for plylst in _train]
    all_val_ids = [plylst['id'] for plylst in _val]

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
    if genre:
        if _submit_type == 'val':
            np.save(f'scores/val_scores_bias_{_score_type}_gnr', results)
        elif _submit_type == 'test':
            np.save(f'scores/test_scores_bias_{_score_type}_gnr', results)
        else:
            np.save(f'scores/local_val_scores_bias_{_score_type}_gnr', results)
    else:
        if _submit_type == 'val':
            np.save(f'scores/val_scores_bias_{_score_type}', results)
        elif _submit_type == 'test':
            np.save(f'scores/test_scores_bias_{_score_type}', results)
        else:
            np.save(f'scores/local_val_scores_bias_{_score_type}', results)


def get_autoencoder_scores(model_file_path, submit_type):
    print("get autoencoder's latent embeddings")
    plylst_emb_with_bias = get_plylsts_embeddings(model_file_path, submit_type, False)

    print("get autoencoder's latent embeddings (genre embeddings are concated)")
    plylst_emb_with_bias_gnr = get_plylsts_embeddings(model_file_path, submit_type, True)

    print("save cos-similarity scores between test embeddings")
    save_scores(plylst_emb_with_bias, 'cos', submit_type, False)

    print("save cos-similarity scores between (test + genre) embeddings and train embeddings")
    save_scores(plylst_emb_with_bias_gnr, 'cos', submit_type, True)

