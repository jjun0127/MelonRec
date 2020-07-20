import random
import os
import torch
import argparse
import torch.nn as nn
import numpy
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset
from Models import AutoEncoder
from data_util import *
from arena_util import write_json, load_json
from evaluate import ArenaEvaluator
from collections import Counter
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

default_file_path = 'arena_data/orig'
question_file_path = 'arena_data/questions/t_val.json'
model_postfix = 'sub'
train_file_path = 'arena_data/orig/t_train.json'
answer_file_path = 'arena_data/answers/sample_val.json'

tag2id_file_path = f'{default_file_path}/t_tag2id.npy'
id2tag_file_path = f'{default_file_path}/t_id2tag.npy'

prep_method_thr = 2
prep_song2id_file_path = f'{default_file_path}/t_freq_song2id_thr{prep_method_thr}.npy'
id2prep_song_file_path = f'{default_file_path}/t_id2freq_song_thr{prep_method_thr}.npy'

train_dataset = SongTagDataset(train_file_path, tag2id_file_path, prep_song2id_file_path)
question_dataset = SongTagDataset(question_file_path, tag2id_file_path, prep_song2id_file_path)

plylst_embed_weight = []
plylst_embed_bias = []

model_file_path = 'model/autoencoder_520_256_0.0005_0.2_2_t.pkl'
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

plylst_emb = dict()
plylst_emb_with_bias = dict()

for idx, (_id, _data) in enumerate(tqdm(train_loader, desc='get train vectors...')):
    with torch.no_grad():
        _data = _data.to(device)
        output = torch.matmul(_data, plylst_embed_weight.T).tolist()  # data (256, 250816), weight.T (250816, 520)
        output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

        _id = list(map(int, _id))
        for i in range(len(_id)):
            plylst_emb[_id[i]] = output[i]
            plylst_emb_with_bias[_id[i]] = output_with_bias[i]

for idx, (_id, _data) in enumerate(tqdm(question_loader, desc='get question vectors...')):
    with torch.no_grad():
        _data = _data.to(device)
        output = torch.matmul(_data, plylst_embed_weight.T).tolist()  # data (256, 250816), weight.T (250816, 520)
        output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

        _id = list(map(int, _id))
        for i in range(len(_id)):
            plylst_emb[_id[i]] = output[i]
            plylst_emb_with_bias[_id[i]] = output_with_bias[i]

numpy.save('t_autoencoder_emb', plylst_emb)
numpy.save('t_autoencoder_emb_with_bias', plylst_emb_with_bias)

