from cmath import isnan
from re import M
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

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
        
        