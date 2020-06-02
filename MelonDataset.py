import numpy as np
from torch.utils.data import Dataset
from arena_util import load_json
import torch
from gensim.models import Word2Vec
import sentencepiece as spm
from word_util import get_tokens_from_sentence

class SongTagDataset(Dataset):
    def __init__(self, data_file_path, tag2id_file_path, freq_song2id_file_path):
        self.train = load_json(data_file_path)
        self.tag2id = dict(np.load(tag2id_file_path, allow_pickle=True).item())
        self.freq_song2id = dict(np.load(freq_song2id_file_path, allow_pickle=True).item())
        self.num_songs = len(self.freq_song2id)
        self.num_tags = len(self.tag2id)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        _id = self.train[idx]['id']
        song_vector = self._song_ids2vec(self.train[idx]['songs'])
        tag_vector = self._tag_ids2vec(self.train[idx]['tags'])
        _input = torch.from_numpy(np.concatenate([song_vector, tag_vector]))

        return _id, _input
    
    def _song_ids2vec(self, songs):
        songs = [self.freq_song2id[song] for song in songs if song in self.freq_song2id.keys()]
        songs = np.asarray(songs, dtype=np.int)
        bin_vec = np.zeros(self.num_songs)
        if len(songs) > 0:
            bin_vec[songs] = 1
        return np.array(bin_vec, dtype=np.float32)
            
    def _tag_ids2vec(self, tags):
        tags = [self.tag2id[tag] for tag in tags if tag in self.tag2id.keys()]
        tags = np.asarray(tags, dtype=np.int)
        bin_vec = np.zeros(self.num_tags)
        bin_vec[tags] = 1
        return np.array(bin_vec, dtype=np.float32)


class SongTagDataset_with_WE(Dataset):
    def __init__(self, data_file_path, tag2id_file_path, freq_song2id_file_path, wv_file_path, tokenizer_file_path):
        self.train = load_json(data_file_path)
        self.tag2id = dict(np.load(tag2id_file_path, allow_pickle=True).item())
        self.freq_song2id = dict(np.load(freq_song2id_file_path, allow_pickle=True).item())
        self.wv = Word2Vec.load(wv_file_path).wv
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_file_path)
        self.num_songs = len(self.freq_song2id)
        self.num_tags = len(self.tag2id)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _id = self.train[idx]['id']
        _tags = self.train[idx]['tags']
        plylst_title = self.train[idx]['plylst_title']
        tokenized_title = get_tokens_from_sentence(self.sp, plylst_title)
        _words = tokenized_title + _tags
        we = []
        for _word in _words:
            if _word in self.tag2id:
                if _word in self.wv:
                    we.append(self.wv[_word])
        if not len(we):
            we = np.zeros(200, dtype=float)
        else:
            we = np.mean(we, axis=0, dtype=float)
        song_vector = self.song_ids2vec(self.train[idx]['songs'])
        tag_vector = self.tag_ids2vec(_tags)
        _input = torch.from_numpy(np.concatenate([song_vector, tag_vector]))

        return _id, _input, we

    def song_ids2vec(self, songs):
        songs = [self.freq_song2id[song] for song in songs if song in self.freq_song2id.keys()]
        songs = np.asarray(songs, dtype=np.int)
        bin_vec = np.zeros(self.num_songs)
        if len(songs) > 0:
            bin_vec[songs] = 1
        return np.array(bin_vec, dtype=np.float32)

    def tag_ids2vec(self, tags):
        tags = [self.tag2id[tag] for tag in tags if tag in self.tag2id.keys()]
        tags = np.asarray(tags, dtype=np.int)
        bin_vec = np.zeros(self.num_tags)
        bin_vec[tags] = 1
        return np.array(bin_vec, dtype=np.float32)
