import torch
from torch.utils.data import Dataset
import os
import torchaudio
from make_dataset import getOriginal

class BasicDataset(Dataset):
    """Generated Input and Output to the model"""

    def __init__(self, cfg, data_path = '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/processed/', mode='train'):
        """ 
        Args:
            cfg (yaml): Config file
            data_path (string): Data directory
        """
        self.cfg = cfg
        self.data_path = data_path + mode

    def __len__(self):
        return len(os.listdir(self.data_path))
                
    def __getitem__(self, i):

        # ======== load audio file ====================
        eps = 1e-6
        win_len = 512
        file_name = os.listdir(self.data_path)[i]
        mix, sample_rate = torchaudio.load(self.data_path + '/' + file_name)
        fpfs, fpss, _, spfs, spss, _ = getOriginal(self.data_path + '/' + file_name, self.cfg)
        og_spec = []
        for wav in [fpfs, fpss, spfs, spss]:
            if wav != None:
                wav_stft = torch.stft(wav,win_len, win_len/2, window=torch.hamming_window(win_len), return_complex= True)
                wav_spec = torch.abs(wav_stft)
                wav_log_spec = torch.log10(wav_spec + eps)
                og_spec.append(wav_log_spec)
            else:
                og_spec.append(torch.ones((257, 257))*float('nan'))

        mix_stft = torch.stft(mix,win_len, win_len/2, window=torch.hamming_window(win_len), return_complex= True)
        mix_spec = torch.abs(mix_stft)
        mix_log_spec = torch.log10(mix_spec + eps)

        fpfs, fpss, spfs, spss = og_spec[0], og_spec[1], og_spec[2], og_spec[3]
        target = torch.stack((fpfs, fpss, spfs, spss),dim=0)
        return mix_log_spec, target


