import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
from tqdm import tqdm
import pandas as pd
from scipy.signal import stft
import os

class BaseDataset(Dataset):
    
    """Data Handler that load dataset."""

    def __init__(self, data_path):
        super().__init__()
        
        self.seed_is_set = False  # multi threaded loading
        # Find all files in data_path and put them in a list
        self.data_path = [os.path.join(data_path, file) for file in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file))]
        
    def set_seed(self, seed):
        raise NotImplementedError
          
    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, index):
        raise NotImplementedError

class SimDataset(BaseDataset):
    def __init__(
            self, data_path, testing=False, 
            save_to_pickle=True, split="train",
            is_real_data=False, max_s=56000, min_s=-36000, add_noise=False, **kwargs):
        super().__init__(data_path)
        self.testing = testing
        self.contact_chunk = None
        self.signal_chunk = None
        self.traj_chunk = None
        self.is_real_data = is_real_data  

        contact_chunks = []
        signal_chunks = []
        traj_chunks = []
        self.add_noise = add_noise

        pickle_save_path = os.path.join(data_path[:18], "pickle", split)
        if os.path.exists(pickle_save_path):
            self.contact_chunk = np.load(os.path.join(pickle_save_path, 'contact_chunk.npy'), allow_pickle=True)
            self.signal_chunk = np.load(os.path.join(pickle_save_path, 'signal_chunk.npy'), allow_pickle=True)
            if testing:
                self.traj_chunk = np.load(os.path.join(pickle_save_path, 'traj_chunk.npy'), allow_pickle=True)
            self.data_length = self.contact_chunk.shape[0]
        else:
            for d in tqdm(self.data_path):
                data_tbl = pd.read_csv(d, sep=',', engine='c', na_filter=False, low_memory=False)
                # contact position data (n, 3)
                P = data_tbl[['cx','cy','cz']].to_numpy()
                # sensor signal data (n, 2)
                S = data_tbl[['sx','sz']].to_numpy()
                # normalize signal value to (0, 1) based on its max and min values
                max_s = max_s 
                min_s = min_s
                S = np.clip(S, min_s, max_s)
                S = (S - min_s) / (max_s - min_s)
                
                # stack contact position and sensor matrix (n, traj_length, data_dim)
                P = P[199::200]
                S = S[199::200]

                Traj = data_tbl[['wbx', 'wby', 'wbz']].to_numpy()
                Traj = Traj[199::200]

                # S and P are (100, 3), expand S and P at the beginning by some random times to (n + 100,3)
                # the repeated values are the first value of S and P
                for _ in range(40):
                    repeat_times = np.random.randint(0, 60)
                    repeat_array_S = np.repeat(np.array([(- min_s) / (max_s - min_s),(- min_s) / (max_s - min_s)]).reshape(1,2), repeat_times, axis=0)
                    new_S = np.concatenate([repeat_array_S, S], axis=0)
                    repeat_array_P = np.repeat(P[0:1], repeat_times, axis=0)
                    new_P = np.concatenate([repeat_array_P, P], axis=0)
                    repeat_array_Traj = np.repeat(Traj[0:1], repeat_times, axis=0)
                    new_Traj = np.concatenate([repeat_array_Traj, Traj], axis=0)

                    new_S = new_S[:100]
                    new_P = new_P[:100]
                    new_Traj = new_Traj[:100]
                    new_C = np.any(abs(new_P) > 1e-5, axis=1)

                    dis = np.linalg.norm(new_P[1:] - new_P[:-1], axis=1)
                    num_jumping = np.sum(dis > 0.005)
                    if num_jumping >= 3 or num_jumping == 0:
                        continue
                    
                    if self.add_noise:
                        # add gaussian noise to the signal
                        noise = np.random.normal(0, 0.001, new_S.shape)
                        new_S += noise

                    contact_chunks.append(np.expand_dims(new_P, axis=0))
                    signal_chunks.append(np.expand_dims(new_S, axis=0))
                    traj_chunks.append(np.expand_dims(new_Traj, axis=0))

            self.contact_chunk = np.concatenate(contact_chunks, axis=0)
            self.signal_chunk = np.concatenate(signal_chunks, axis=0)
            self.traj_chunk = np.concatenate(traj_chunks, axis=0)
            self.data_length = self.contact_chunk.shape[0]

            if save_to_pickle:
                os.system(f"mkdir -p {pickle_save_path}")
                np.save(os.path.join(pickle_save_path, 'contact_chunk.npy'), self.contact_chunk)
                np.save(os.path.join(pickle_save_path, 'signal_chunk.npy'), self.signal_chunk)
                np.save(os.path.join(pickle_save_path, 'traj_chunk.npy'), self.traj_chunk)
        
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        return_data = {}
        # (traj_len, 3)
        position = torch.from_numpy(self.contact_chunk[index]).type(torch.FloatTensor)
        position = position[1:]  # shift position by 1
        return_data['position'] = position  
        sensor_signal = torch.from_numpy(self.signal_chunk[index]).type(torch.FloatTensor)
        sensor_signal = sensor_signal[:-1]  # shift sensor_signal by 1
        return_data['signal'] = sensor_signal

        # traj of the whisker base
        if self.testing:
            traj = torch.from_numpy(self.traj_chunk[index]).type(torch.FloatTensor)
            traj = traj[1:]
            return_data['traj'] = traj

        return return_data 