import os
import h5py
import random
import logging

import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from data_utils import AudioPreProcessing, cut_interval

log = logging.getLogger(__name__)


class SwitchSpeakerdDataset(Dataset):
    """Switching Speakers Mixturedataset."""

    def __init__(self, cfg, data_dir, nist_data_dir=None, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.combined_data = False

        # if nist_data_dir:
        #     self.nist_data_dir = nist_data_dir + "/aud_files/"
        #     self.nist_labels_dir = nist_data_dir + "/switch_labels/"
        #     self.nist_specto_dir = nist_data_dir + "/spectograms/"
        #     self.os_listdir_nist = os.listdir(self.nist_data_dir)
        #     self.os_listdir_nist.sort()
        #     self.combined_data = True

        self.data_dir = data_dir + "/aud_files/"
        self.labels_dir = data_dir + "/switch_labels/"
        self.specto_dir = data_dir + "/spectograms/"
        self.os_listdir = os.listdir(self.data_dir)
        self.os_listdir.sort()

        self.audio_pre_process = AudioPreProcessing(cfg)

    def __len__(self):
        if self.combined_data:
            return len(self.os_listdir) + len(self.os_listdir_nist)
        else:
            return len(self.os_listdir)

    def get_single_speaker_labels(self, file_name):
        h5f_file_labels = h5py.File(self.labels_dir + file_name, "r")

        switch_labels = h5f_file_labels["labels_samples"][:]
        switch_labels_frames = h5f_file_labels[f'labels_n_fft_{self.cfg.window_size}_hop_length_{self.cfg.overlap}'][:]

        h5f_file_labels.close() 

        return switch_labels, switch_labels_frames

    def __getitem__(self, idx):
        # ======== get audio file name====================
            
        aud_file_name = self.os_listdir[idx]
        aud_file_name_no_wav = aud_file_name[:-4]

        # ======== load labels from H5 files ====================

        switch_labels, switch_labels_frames = self.get_single_speaker_labels(aud_file_name_no_wav + "_labels.h5")

        # ======== load audio file ====================

        aud, fs = torchaudio.load(self.data_dir + aud_file_name)

        switch_labels_frames = switch_labels_frames.squeeze()
        
        log_spec_audfile = self.audio_pre_process.wav_to_log_spec(aud)

        # ======== Decide if should taking switch interval or non-switch interval ====================
        take_switch_interval = random.choices([0, 1], weights=[0.75, 0.25], k=1)[0]

        # ======== Cut into a specific interval ====================
        if self.mode == 'train' or self.mode == 'val':
            if self.cfg.switch_cut:
                cut_length = self.cfg.num_frames  # 128 frames ~2 sec
            else:
                cut_length = 512  # 512 frames ~8 sec
            log_spec_audfile, switch_labels_frames = cut_interval(self.cfg, log_spec_audfile, switch_labels_frames, cut_length=cut_length, switch_interval=take_switch_interval)            

        if self.cfg.spec_augment and self.mode == 'train':
            log_spec_audfile = self.audio_pre_process.spec_aug(self.cfg, log_spec_audfile)

        log_spec_audfile = log_spec_audfile.unsqueeze(0)

        # switch_indices = np.where(switch_labels_frames == 1)
        # if switch_indices:
        #     switch_inx = switch_indices[0]
        # else:
        #     switch_inx = -1 

        if self.cfg.switch_cut and (self.mode == 'train' or self.mode == 'val'):
            if switch_labels_frames.any():
                switch_labels_frames = np.where(switch_labels_frames == 1)[0][0]  # First switch point
                if not self.cfg.choose_from_all:
                    switch_labels_frames -= cut_length // 2 - self.cfg.classify_window_size // 2
            else:
                if self.cfg.choose_from_all:
                    switch_labels_frames = cut_length  # No-switch flag, label is num_frames (num_classes) + 1
                else:
                    switch_labels_frames = self.cfg.classify_window_size
            if switch_labels_frames < 0 or switch_labels_frames > 10:
                switch_labels_frames = self.cfg.classify_window_size

        if self.cfg.predict_spec:
            # Create a spec target with ones on switch frame for imag-to-imag task
            log_spec_target = log_spec_audfile.clone()
            log_spec_target[..., np.where(switch_labels_frames == 1)] = 10

            return log_spec_audfile, log_spec_target
        else:
            return log_spec_audfile, switch_labels_frames


class SwitchSpeakerdDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        if self.cfg.data_type == 'nist':
            self.train_set = SwitchSpeakerdDataset(self.cfg, self.cfg.data_nist_dir.train)
        else:
            self.train_set = SwitchSpeakerdDataset(self.cfg, self.cfg.data_dir.train)
        self.val_set = SwitchSpeakerdDataset(self.cfg, self.cfg.data_dir.val, 'val')
        # self.test_set = BasicDataset(self.cfg, self.cfg.data_dir.test, 'test')
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=self.cfg.data_loader_shuffle, num_workers =self.cfg.num_workers, pin_memory= self.cfg.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.cfg.batch_size, shuffle=self.cfg.data_loader_shuffle, num_workers = self.cfg.num_workers, pin_memory= self.cfg.pin_memory)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.cfg.test_batch_size, shuffle=self.cfg.data_loader_shuffle, num_workers = self.cfg.num_workers, pin_memory= self.cfg.pin_memory)