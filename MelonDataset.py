import numpy as np
from torch.utils.data import Dataset, DataLoader
from arena_util import load_json
import torch


class SongTagDataset(Dataset):
    def __init__(self, file_path=None):
        if file_path:
            self.train = load_json(file_path)
        else:
            self.train = load_json('arena_data/orig/train.json')
        self.tag_to_id = dict(np.load('arena_data/orig/tag_to_id.npy', allow_pickle=True).item())
        self.num_songs = 707989
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
        songs = np.asarray(songs)
        bin_vec = np.zeros(self.num_songs)
        bin_vec[songs] = 1
        return np.array(bin_vec, dtype=np.float32)
            
    def tag_ids2vec(self, tags):
        tags = [self.tag_to_id[tag] for tag in tags]
        tags = np.asarray(tags)
        bin_vec = np.zeros(self.num_tags)
        bin_vec[tags] = 1
        return np.array(bin_vec, dtype=np.float32)

if __name__ == '__main__':
    playlist_fn = '../data/train.json'
    
    std = SongTagDataset('../data/SongTagDataset_train.pkl')
    dataloader = DataLoader(std, batch_size=128, num_workers=0)
    for idx, (_ids, _inputs) in enumerate(dataloader):
        print(idx, len(_idx), _input.size())
        break