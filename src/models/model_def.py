import os
import warnings
import logging
from collections import OrderedDict
from cmath import isnan
from re import M
from unicodedata import bidirectional

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

warnings.filterwarnings("ignore", category=UserWarning)

# A logger for this file
log = logging.getLogger(__name__)



R_THRESHOLD = 20
ID_SIZE = 128


class speech_lstm(nn.Module):
    def __init__(self):
        #first block - BLSTM and FC
        super(speech_lstm, self).__init__()
        self.blstm = nn.LSTM(257*2, ID_SIZE, 2, batch_first = True, bidirectional = True)
        self.fclstm = nn.Linear(ID_SIZE*2,ID_SIZE, bias = False)

        #second - speaker adapt
        self.fc_speaker_adapt = nn.Linear(ID_SIZE,ID_SIZE, bias = False)

        #masking
        self.fc_mask = nn.Linear(ID_SIZE , 257, bias = False)

        #speaker embedding
        self.fc_spk_1 = nn.Linear(ID_SIZE,50, bias = False)
        self.fc_spk_2 = nn.Linear(50,50, bias = False)
        self.fc_spk_3 = nn.Linear(50,ID_SIZE, bias = False)

        self.fc_affine = nn.Linear(ID_SIZE,ID_SIZE) 

        #gate
        self.fc_gate = nn.Linear(ID_SIZE,ID_SIZE, bias = False)
        self.fc_norm = nn.Sequential(nn.Linear(160,70),nn.Linear(70,1))



    def sys_pass(self, x): # changed forward to sys_pass
        Y, R, z = x
        #BLSTM
        lstm_out, self.hidden = self.blstm(torch.cat((Y,R),1).permute(0,2,1))
        lstm_out =  torch.sigmoid(self.fclstm(lstm_out))
        #SPEAKER ADAPT
        speaker_adapt = lstm_out * torch.sigmoid(self.fc_speaker_adapt(z).unsqueeze(1))

        #MASK & SPEAKER ADAPT - OUTPUT
        M_out = torch.sigmoid(self.fc_mask(lstm_out * torch.sigmoid(self.fc_speaker_adapt(z).unsqueeze(1)))).permute(0,2,1)

        #SPK. EMBEDDING
        speaker_embedding = self.fc_affine(torch.mean(torch.sigmoid(self.fc_spk_3(torch.sigmoid(self.fc_spk_2(torch.sigmoid(self.fc_spk_1(speaker_adapt)))))), dim = 1))

        #GATE - OUTPUT
        pre_norm = torch.sigmoid(self.fc_gate(speaker_adapt)) * speaker_embedding.unsqueeze(1)
        z_out = F.normalize(self.fc_norm(pre_norm.permute(0,2,1)).squeeze(-1) + z , dim = 1).detach().requires_grad_() #this line might be problematic

        #RES-MASK - OUTPUT
        R_out = torch.max(R-M_out, torch.zeros(R.shape,device=R.device)).detach() # need something that converges -normalize

        del pre_norm
        del speaker_embedding
        del speaker_adapt
        del lstm_out
        del x

        return M_out, R_out, z_out


    def forward(self, Y): # this is meant to create the different cells of our forward feed
        sample_length = Y.shape[-1]
        R = torch.ones(Y.shape, device = Y.device)
        num_of_parts = sample_length // 160
        M = torch.zeros(num_of_parts,3,Y.shape[0],Y.shape[1],Y.shape[2]//4, device = Y.device)
        z = torch.ones((Y.shape[0] ,num_of_parts, ID_SIZE),device = Y.device)
        for time in range(num_of_parts):
            Y_i = Y[..., 160 *time :160 * (time+1)]
            for id_num in range(3):
                M_i, R_i, z_i = self.sys_pass((Y_i, R[..., 160 *time :160 * (time+1)], z[:, id_num, :]))
                R[..., 160 *time :160 * (time+1)] = R_i
                z[:, id_num, :] = z_i
                M[time, id_num, ...] = M_i
        
        del Y
        del Y_i
        del M_i
        del R_i
        del z_i
        
        return M ,R, z
        
    def init_hidden(self, device):
        return (torch.zeros((4,32,ID_SIZE)).to(device),
                torch.zeros((4,32,ID_SIZE)).to(device))

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

class speech_lstm_PL(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.model = speech_lstm()
        self.cfg = cfg

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def loss_mmse(self, mask, spectogram, target, device): # need help from Lior <-----
        loss = []
        batch_size = spectogram.shape[0]
        sample_length = spectogram.shape[-1]
        mask_length = mask.shape[-1]
        blocks_num = sample_length // 160
        targets = tuple_of_tensors_to_tensor(torch.cat(torch.split(target, 2, dim=1),dim = -1))
        for batch in range(batch_size):
            for time in range(blocks_num):
                sources = targets[...,time * 160: (time + 1) * 160].to(device)
                Y = spectogram[..., mask_length * time :mask_length * (time+1)]
                product = []
                for speaker in range(sources.shape[1]): # added 1 for noise (we could also use sizze of masks)
                    product.append(torch.sum(torch.abs((mask[time,speaker,batch,...] * Y) - sources[batch,speaker,...])**2))
                loss.append(torch.min(torch.tensor(product, requires_grad=True).float()))
        del mask, spectogram, target, targets, sources, Y, product
        return torch.sum(torch.tensor(loss, requires_grad=True).float()) / (batch_size*blocks_num)


    def loss_res_mask(self, mask, device): 
        values = torch.sum(1-mask, dim = 1)
        product = torch.maximum(values, torch.zeros(values.shape).to(device))
        del mask, values
        return torch.sum(torch.sum(product, dim = [0,2,3])).long()
            
    def training_step(self, batch, batch_idx):
        mix_log_spec, target = batch

        output = self.model(mix_log_spec)
        mask, Y, z = output
        loss_mmse = self.loss_mmse(mask, mix_log_spec, target, mix_log_spec.device)
        loss_res_mask = self.loss_res_mask(mask, mix_log_spec.device)
        loss = loss_mmse + loss_res_mask

        self.log("train_loss", loss, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log("train_loss_mmse", loss_mmse, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log("train_loss_res_mask", loss_res_mask, rank_zero_only=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, loader_idx=None):
        mix_log_spec, target = batch

        output = self.model(mix_log_spec)
        mask, Y, z = output
        loss_mmse = self.loss_mmse(mask, mix_log_spec, target, mix_log_spec.device)
        loss_res_mask = self.loss_res_mask(mask, mix_log_spec.device)
        loss = loss_mmse + loss_res_mask

        self.log("val_loss", loss, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log("val_loss_mmse", loss_mmse, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log("val_loss_res_mask", loss_res_mask, rank_zero_only=True, on_step=False, on_epoch=True)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        if self.cfg.use_sched:
            sch = self.lr_scheduler['scheduler']
            sch.step(avg_loss)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=0)
        elif self.cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9)
        else:
            print("error no optimzier has been configrues - got optimzier {}".format(self.cfg.optimizer))

        if self.cfg.use_sched:
            self.lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                    factor=0.8, patience=5, cooldown=3, verbose=True)}
            # self.lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.step.step_size, gamma=self.cfg.step.gamma)}
        
        return optimizer