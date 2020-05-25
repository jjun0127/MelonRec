import numpy as np
from torch.utils.data import Dataset, DataLoader
from arena_util import load_json
import torch


class SongTagDataset(Dataset):
    def __init__(self, data_file_path, tag_id_file_path):
        self.train = load_json(data_file_path)
        self.tag_to_id = dict(np.load(tag_id_file_path, allow_pickle=True).item())
        self.freq_song_to_id = dict(np.load('arena_data/orig/freq_song_to_id.npy', allow_pickle=True).item())
        self.num_songs = len(self.freq_song_to_id)
        self.num_tags = 29160

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        _id = self.train[idx]['id']
        song_vector = self.song_ids2vec(self.train[idx]['songs'])
        tag_vector = self.tag_ids2vec(self.train[idx]['tags'])
        _input = torch.from_numpy(np.concatenate([song_vector, tag_vector]))

        return _id, _input
    
    def song_ids2vec(self, songs):
        songs = [self.freq_song_to_id[song] for song in songs if song in self.freq_song_to_id.keys()]
        songs = np.asarray(songs)
        bin_vec = np.zeros(self.num_songs)
        if len(songs) > 0:
            bin_vec[songs] = 1
        return np.array(bin_vec, dtype=np.float32)
            
    def tag_ids2vec(self, tags):
        tags = [self.tag_to_id[tag] for tag in tags]
        tags = np.asarray(tags)
        bin_vec = np.zeros(self.num_tags)
        bin_vec[tags] = 1
        return np.array(bin_vec, dtype=np.float32)

if __name__ == '__main__':
    std = SongTagDataset()
    data_loader = DataLoader(std, batch_size=128, num_workers=0)
    for idx, (_ids, _inputs) in enumerate(data_loader):
        print(idx, len(_ids), _inputs.size())
        break
