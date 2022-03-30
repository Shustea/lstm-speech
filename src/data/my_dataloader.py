import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    """Generated Input and Output to the model"""

    def __init__(self, cfg, data_path, mode='train', use_raw=False, diverse=''):
        """ 
        Args:
            cfg (yaml): Config file
            data_path (string): Data directory
        """
        self.cfg = cfg

    def __len__(self):
        if self.cfg.mode == 'test' or self.mode == 'test':
            return len(self.data)
        else:
            return len(self.hdf.get('Noisy_input'))        
                
    def __getitem__(self, i):

        # ======== load audio/img file ====================

        if self.cfg.use_h5 and not self.use_raw:
            if self.cfg.mode == 'test' or self.mode == 'test':
                input_map = self.data[i]["Noisy_input"].float()
                target = self.data[i]["Target"].float()
                snr = self.data[i]["SNR"]
                noisy_stft = self.data[i]["Noisy_complex"]
                S_stft_complex = self.data[i]["Target_complex"]
                raw_wave = self.data[i]["Raw"]

                if self.cfg.test_diverse:
                    snr = [str(s) for s in snr]

                return input_map, target, raw_wave, noisy_stft, S_stft_complex, snr
            else:
                input_map = self.hdf.get('Noisy_input')[i]
                input_map = torch.tensor(input_map)
                target = self.hdf.get('Target')[i]
                target = torch.tensor(target)
                if self.cfg.diff_ref:
                    which_ref = self.hdf.get('Diff_ref')[i]
                if self.cfg.use_sisdri:
                    mixed = self.hdf.get('Raw')[i]
                    mixed = torch.tensor(mixed)
                    noisy_stft = self.hdf.get('Noisy_complex')[i]
                    noisy_stft = torch.tensor(noisy_stft)
                if self.cfg.use_ref:
                    input_map = input_map.float()
                else:
                    input_map = input_map.float().unsqueeze(0)
                if self.cfg.reconst_noise:
                    target = target.float()
                else:
                    target = target.float().unsqueeze(0)
                
                if self.cfg.input_map == 'stft_RI':
                    # real_noisy = noisy_stft.real
                    # imag_noisy = noisy_stft.imag
                    # input_map = torch.stack((real_noisy, imag_noisy), dim=0).float()
                    # clean = mixed[2, :]
                    # clean_stft = torch.stft(clean, self.cfg.window_size, int(self.cfg.window_size * self.cfg.overlap), window=torch.hamming_window(self.cfg.window_size), return_complex=True)
                    # clean_real = clean_stft.real
                    # clean_imag = clean_stft.imag
                    # target = torch.stack((clean_real, clean_imag), dim=0).float()
                    target = target.squeeze(0).float()
                    input_map = input_map.squeeze(0).float()
                    if not self.cfg.use_ref:
                        input_map = input_map[:2, ...]
                    if not self.cfg.reconst_noise:
                        target = target[:2, ...]             
                    input_map[:, 0:3, :] = input_map[:, 0:3, :] * 0.001
                    if self.cfg.use_ref:
                        mx_mix = torch.max(torch.max(torch.abs(input_map[0, ...]), torch.max(torch.abs(input_map[1, ...]), torch.max(torch.abs(input_map[2, ...]), torch.max(torch.abs(input_map[3, ...]))))))
                    else:
                        mx_mix = torch.max(torch.max(torch.abs(input_map[0, ...]), torch.max(torch.abs(input_map[1, ...]))))
                    input_map /= mx_mix
                    if self.cfg.remove_small_freq:
                        target[:, 0:3, :] = target[:, 0:3, :] * 0.001

                if self.cfg.diff_ref:  # Target is 0 for another ref speaker
                    target = target.squeeze(0).float()
                    if self.cfg.diff_noisy_target and not self.cfg.input_map == 'stft_RI':
                        target[0, ...][which_ref == 1] = torch.randn(target[0, ...][which_ref == 1].shape, device=target.device) * 0.5 - 5  # Target is white noise with std=0.5 mean=-5
                    elif not self.cfg.input_map == 'stft_RI':
                        target[0, ...][which_ref == 1] = torch.zeros(target[0, ...][which_ref == 1].shape, device=target.device)
                        # target[:, 0, ...][which_ref == 1] = torch.ones(target[:, 0, ...][which_ref == 1].shape, device=target.device) * (-5)
                    else:
                        target[:2, ...][which_ref == 1] = torch.zeros(target[:2, ...][which_ref == 1].shape, device=target.device)
                
                if self.cfg.diff_ref:
                    if self.cfg.use_sisdri:
                        return input_map, target, which_ref, mixed, noisy_stft
                    else:
                        return input_map, target, which_ref
                else:
                    if self.cfg.use_sisdri:
                        return input_map, target, mixed, noisy_stft
                    else:
                        return input_map, target
        else:
            wave_file = self.waves_list[i]
            sample = wave_file.split("_")[0]
            s_id = wave_file.split("_")[4]
            snr = int(wave_file.split("_")[-1].split(".")[0])
            mixed, fs = librosa.load(self.data_path + '/' + wave_file, sr=self.cfg.fs, mono=False)

        # ==================== data-preprocessing ======================

            if self.cfg.mode == 'train':
                input_map, target = self.audio_pre_process.preprocess(mixed)
                # if self.save_img:
                #     plt.figure()
                #     pos = plt.imshow(input_map.squeeze().cpu().detach().numpy()[::-1],aspect='auto')
                #     plt.title('Noisy')
                #     plt.colorbar(pos)
                #     plt.savefig('{}/stft/noisy'.format(self.cfg.save_plots_path))
                #     plt.close()
                    
                #     plt.figure()
                #     pos = plt.imshow(target.cpu().detach().numpy()[::-1],aspect='auto')
                #     plt.colorbar(pos)
                #     plt.title('Clean')
                #     plt.savefig('{}/stft/clean'.format(self.cfg.save_plots_path))
                #     plt.close()
                #     self.save_img == False
                
                return input_map.float().unsqueeze(0), target.float().unsqueeze(0)  # Channel dim 1

            elif self.cfg.mode == 'test':
                input_map, target, noisy_stft, S_stft_complex, V_stft_complex = self.audio_pre_process.preprocess(mixed)
                
                return input_map.float(), target.float(), mixed, noisy_stft, S_stft_complex, snr