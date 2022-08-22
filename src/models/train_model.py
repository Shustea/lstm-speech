### Adam Shusterman & Yahav Avraham ###
###### Last Edited - 11/8/22 ######

import errno
from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
# import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import gc
# import matplotlib.pyplot as plt
from model_def import speech_lstm
sys.path.append("/dsi/scratch/from_netapp/users/shustea1/Data/lstm-speech/src/data")
from my_dataloader import BasicDataset

RES_THRESHOLD = 20

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def createData(args):

    train_set = BasicDataset(args)

    val_set = BasicDataset(args,mode = 'val')

    # test_set = BasicDataset(args,test_path)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)


def loss_mmse(mask, spectogram, target): # need help from Lior <-----
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


def loss_res_mask(mask): 
    values = torch.sum(1-mask, dim = 1)
    product = torch.maximum(values, torch.zeros(values.shape).to(mask.device))
    del mask, values
    return torch.sum(torch.sum(product, dim = [0,2,3])).long()


def loss_triplet(sn_an, sn_ap, args):
    return torch.sum(max(sn_an - sn_ap + args.eps, 0))

def run(train_loader, val_loader, model, device, args):
    epochs, lr, meu, alpha, batch_size = args.epochs, args.lr, args.meu, args.alpha, args.batch_size
    optimizer = optim.SGD(model.parameters(), lr, momentum=meu, weight_decay=alpha)
    train_loss, val_loss = ([] for i in range(2))
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for index, data in enumerate(train_loader, 0):

            mix_log_spec, target = data
            # mix_log_spec_ = mix_log_spec.cuda()
            mix_log_spec = mix_log_spec.cuda().detach()



            optimizer.zero_grad()

            output = model(mix_log_spec)
            mask, Y, z = output
            loss = loss_mmse(mask, mix_log_spec, target) + loss_res_mask(mask) #+ loss_triplet()
            loss.backward()

            optimizer.step()

            # save stat
            running_loss += loss.item()

            print(f'finished batch #{index+1}')
        valloss = val(val_loader, model, device)
        train_loss.append(running_loss / index)
        val_loss.append(val_loss)
        print(f'loss of the network on the {epoch + 1} epoch is {valloss}')
    results = val_loss,  train_loss
    print('Finished Training')
    return model, results


def val(val_loader, model, device):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(val_loader, 1):
            mix_log_spec, target = data
            mix_log_spec = mix_log_spec.to(device)
            output = model(mix_log_spec)
            mask, Y, z = output
            loss = loss_mmse(mask, mix_log_spec, target) + loss_res_mask(mask)
            val_loss += loss.item()
    return  val_loss / index


def test(test_loader, model, device, args):
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(test_loader, 1):
            mix_log_spec, target = data
            mix_log_spec = mix_log_spec.to(device)
            target = target.to(device)
            output = model(mix_log_spec)
            mask, Y, z = output
            DER = torch.sum(torch.isnan(mask[:,:,0,0]), dim = 1)
            result = mask * (10**(mix_log_spec))
            mix_stft_angle = torch.angle((10**(mix_log_spec)))
            result = result * torch.exp(1j*mix_stft_angle)
            result_wav = torch.istft(result, args.win_len, int(args.win_len/2), window=torch.hamming_window(args.win_len), return_complex= True)
    return  result_wav, DER


# def plot_results(test_losses, train_losses, epochs):
#      # plot loss and accuracy on validation set
#     steps = np.arange(epochs)

#     fig, ax1 = plt.subplots(figsize=(8, 5))

#     ax1.set_xlabel('epochs')
#     ax1.set_ylabel('loss')
#     ax1.plot(steps, train_losses, label="train loss", color='red')
#     ax1.plot(steps, test_losses, label="val loss", color='green')

#     fig.legend()
#     # fig.suptitle('Epochs={}, LR={}, momentum={}, reg={}'.format(epochs, lr, momentum, weight_decay))

#     plt.show()


def main(args):

    lstm_model = speech_lstm()
    lstm_model.to(device)
    lstm_model.hidden = lstm_model.init_hidden(device=device)

    train_loader, val_loader = createData(args)

    learned_model, results = run(train_loader, val_loader, lstm_model, device, args)
    torch.save(learned_model,f'/dsi/scratch/from_netapp/users/shustea1/Data/lstm-speech/models/{args.batch_size}_{args.lr}_{args.meu}_{args.alpha}.ckpt')
    test_losses, train_losses = results
    # plot_results(test_losses, train_losses, args.epochs)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= 'Diarization, Counting and Separation')

    parser.add_argument('--epochs', type=int, default= 40,
                    help='Number of Epochs')
    parser.add_argument('--num_workers', type=int, default= 16,
                    help='Number of GPU workers')
    parser.add_argument('--lr', type=float, default= 0.002,
                    help='Learning rate constant')
    parser.add_argument('--meu', type=float, default= 0.9,
                    help='momenteum constant')
    parser.add_argument('--alpha', type=float, default= 0.01,
                    help='decay constant')
    parser.add_argument('--batch_size', type=int, default= 64,
                    help='Batch size')
    parser.add_argument('--val_num', type=int, default= 250,
                    help='Number of validation samples')
    parser.add_argument('--test_num', type=int, default= 250,
                    help='Number of testing samples')
    parser.add_argument('--fs', type=int, default= 16000,
                    help='sample rate')
    parser.add_argument('--val_path', type=str, default= '/dsi/scratch/from_netapp/users/shustea1/Data/lstm-speech/data/processed/val/',
                    help='val folder path')
    parser.add_argument('--test_path', type=str, default= '/dsi/scratch/from_netapp/users/shustea1/Data/lstm-speech/data/processed/test/',
                    help='test folder path')
    parser.add_argument('--target_path', type=str, default= '/dsi/scratch/from_netapp/users/shustea1/Data/lstm-speech/data/processed/target/',
                    help='test folder path')
    parser.add_argument('--wav_path', type=str, default= '/dsi/scratch/from_netapp/users/shustea1/Data/lstm-speech/data/raw/WSJ0/',
                    help='wav folder path - for original, non-mixed wav files')
    parser.add_argument('--eps', type=float, default= 10 ** -8,
                    help = 'very small number for numerical stability')
    parser.add_argument('--win_len', type=int, default= 512, 
                    help = 'window length for stft')

    # parser.add_argument('--train_num', type=int, default= 30000,

    args = parser.parse_args()

    main(args)