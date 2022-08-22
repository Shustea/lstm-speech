# from asyncio.windows_events import NULL
from pickle import NONE
import torch
from torch.utils.data import Dataset
import os
import torchaudio
import numpy as np
from make_dataset import getOriginal

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BasicDataset(Dataset):
    """Generated Input and Output to the model"""

    def __init__(self, cfg, mode='train'):
        """ 
        Args:
            cfg (yaml): Config file
            data_path (string): Data directory
        """
        self.cfg = cfg
        self.data_path = cfg.data_path + f'/{mode}'

    def __len__(self):
        return len(os.listdir(self.data_path))
                
    def __getitem__(self, i):

        # ======== load audio file ====================
        five_sec = 5*16000
        eps = 1e-6
        win_len = 512
        file_name = os.listdir(self.data_path)[i]
        mix, sample_rate = torchaudio.load(self.data_path + '/' + file_name)
        mix = mix.squeeze()
        fpfs, fpss, _, spfs, spss, _= getOriginal(file_name, self.cfg)
        wav = [fpfs, fpss, spfs, spss]
        og_spec = []
        for index in range(4):
            if not (wav[index] is None):
                wav_stft = torch.stft(torch.cat((torch.tensor(wav[index]),torch.zeros((1916))),dim=-1),win_len, int(win_len/2), window=torch.hamming_window(win_len), return_complex= True)
                wav_spec = torch.abs(wav_stft)
                wav_log_spec = torch.log10(wav_spec + eps)
                og_spec.append(wav_log_spec)
            else:
                og_spec.append(torch.ones((257, 320))*float(-1000))
        mix_stft = torch.stft(torch.cat((mix,torch.zeros(3828)),dim=-1),win_len, int(win_len/2), window=torch.hamming_window(win_len), return_complex= True)
        mix_spec = torch.abs(mix_stft)
        mix_log_spec = torch.log10(mix_spec + eps)
        fpfs_stft, fpss_stft, spfs_stft, spss_stft = og_spec[0], og_spec[1], og_spec[2], og_spec[3]
        target = torch.stack((fpfs_stft, fpss_stft, spfs_stft, spss_stft),dim=0)
        return mix_log_spec, target


class LstmSpeechDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        self.train_set = BasicDataset(self.cfg)
        self.val_set = BasicDataset(self.cfg, mode = 'val')
        # self.test_set = BasicDataset(self.cfg, self.cfg.data_dir.test, 'test')
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=self.cfg.data_loader_shuffle, num_workers =self.cfg.num_workers, pin_memory= self.cfg.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.cfg.batch_size, shuffle=self.cfg.data_loader_shuffle, num_workers = self.cfg.num_workers, pin_memory= self.cfg.pin_memory)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.cfg.test_batch_size, shuffle=self.cfg.data_loader_shuffle, num_workers = self.cfg.num_workers, pin_memory= self.cfg.pin_memory)