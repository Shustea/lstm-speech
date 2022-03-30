from ast import operator
#from asyncio.windows_events import NULL
import librosa
import argparse
import numpy as np
import numpy.random as random
import soundfile as sf
import os


def convert_wv12wav(args):
    # input - original path, intended wav path, sampling rate
    # output - void
    # function converts between WV1 file to in our original path to a WAV file in our intended path, file SR is fs 

    speakers_dirs = os.listdir(args.original_path)

    for speaker in speakers_dirs:
        if not os.path.isdir(args.wav_path + speaker): 
            os.mkdir(args.wav_path + speaker)
        files = os.listdir(args.original_path + '/' + speaker)
        files = [file for file in files if file.split('.')[-1] == 'wv1']
        for signal in files:
            wv1_signal, fs = librosa.load(args.original_path + '/' + speaker + '/' +signal, fs)
            sf.write(args.wav_path + speaker + '/' + signal.split('.')[0] + '.wav', wv1_signal, fs)


def mix_signal(wav_data_path, p_silence, p_one,args):
    # input - wav files path, first speaker porbabilty, second speaker probabilty
    # output - mixed signal of speakers according to their defined probabilty
    # function uses given probabilty for each speaker to speak, mixes them to 5 sec intervals, and returns (meant to be used twice for our data)
    five_sec = 5*args.fs
    first_speaker = np.random.normal(0,1e-4,five_sec) 
    second_speaker = np.zeros(five_sec)
    speakers_dirs = os.listdir(wav_data_path)
    prob = np.random.rand()
    snr = 0
    g = 0
    if prob > p_silence :
        first_speaker_id = random.choice(speakers_dirs)
        first_file_id = random.choice(os.listdir(wav_data_path + first_speaker_id))
        first_speaker, fs = librosa.load(wav_data_path + first_speaker_id + '/' + first_file_id, args.fs)
        if len(first_speaker) > five_sec :
            rand_start = random.randint(0,len(first_speaker)-five_sec-1) 
            first_speaker = first_speaker[rand_start:rand_start+five_sec]
        else:
            first_speaker = np.concatenate((first_speaker,np.random.normal(0,1e-4,five_sec-len(first_speaker))))
        speakers_dirs.remove(first_speaker_id)
    else:
        first_file_id = 'NONE'
    if prob > p_one + p_silence: # probabilty for 2 speakers
        second_speaker_id = random.choice(speakers_dirs)
        second_file_id = random.choice(os.listdir(wav_data_path + second_speaker_id))
        snr = np.random.uniform(0,5)
        second_speaker, fs = librosa.load(wav_data_path + second_speaker_id + '/' + second_file_id, fs)
        g = np.sqrt(10**(-snr/10) * (np.std(first_speaker)**2 / np.std(second_speaker)**2))
        if len(second_speaker) > five_sec :
            rand_start = random.randint(0,len(second_speaker)-five_sec-1) #clips bigger then 10 sec
            second_speaker = second_speaker[rand_start:rand_start+five_sec]
        else:
            second_speaker = np.concatenate((second_speaker,np.random.normal(0,1e-4,five_sec-len(second_speaker))))
        speakers_dirs.remove(second_speaker_id)
    else:
        second_file_id = 'NONE'
    mixed_signal = first_speaker + g * second_speaker
    return mixed_signal, first_file_id.split('.')[0] + '-' + second_file_id.split('.')[0] + '-' + str(snr)

def createSample(args): #change to generic values
    # for i in range(args.train_num):
    #     mix1, mix1_id = mix_signal(args.wav_path, 0, 0.5, args)
    #     mix2, mix2_id = mix_signal(args.wav_path, 0.55, 0.15, args)
    #     final_sample = np.concatenate((mix1, mix2))
    #     sf.write(args.train_path + mix1_id + '_' + mix2_id + '.wav' , final_sample, args.fs)
    #     print('finished file {}/{}'.format(i+1, args.train_num))
    for i in range(args.val_num):
        mix1, mix1_id = mix_signal(args.wav_path, 0, 0.5, args)
        mix2, mix2_id = mix_signal(args.wav_path, 0.55, 0.15, args)
        final_sample = np.concatenate((mix1, mix2))
        sf.write(args.val_path + mix1_id + '_' + mix2_id + '.wav' , final_sample, args.fs)
        print('finished file {}/{}'.format(i+1, args.val_num))

def getOriginal(sample , wav_data_path):
    first_part_first_speaker, first_part_second_speaker, second_part_first_speaker, second_part_second_speaker = []
    first_snr, second_snr = 0
    first_part = sample.split('_')[0]
    second_part = sample.split('_')[1].split('.')[0]
    first_part_first_speaker, fs = librosa.load(wav_data_path + first_part.split('-')[0])
    if not first_part.split('-')[1] == 'NONE':
        first_part_second_speaker, [] = librosa.load(wav_data_path + first_part.split('-')[1])
        first_snr = first_part.split('-')[-1]
    second_part_first_speaker, [] = librosa.load(wav_data_path + second_part.split('-')[0])
    if not second_part.split('-')[1] == 'NONE':
        second_part_second_speaker, [] = librosa.load(wav_data_path + second_part.split('-')[1])
        second_snr = second_part.split('-')[-1]
    return first_part_first_speaker, first_part_second_speaker, first_snr, second_part_first_speaker, second_part_second_speaker, second_snr


if __name__ == "__main__":
    # Define 

    parser = argparse.ArgumentParser(description= 'Diarization, Counting and Separation')

    parser.add_argument('--train_num', type=int, default= 30000,
                    help='Number of training samples')
    parser.add_argument('--val_num', type=int, default= 250,
                    help='Number of validation samples')
    parser.add_argument('--test_num', type=int, default= 250,
                    help='Number of testing samples')
    parser.add_argument('--fs', type=int, default= 16000,
                    help='sample rate')
    parser.add_argument('--original_path', type=str, default= '/mnt/dsi_vol1/users/shustea1/data/sd_et_20',
                    help='original data folder path - saved as wv1 and wv2 files')
    parser.add_argument('--train_path', type=str, default= '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/processed/train/',
                    help='train folder path - saved as 10 sec mixed wav files')
    parser.add_argument('--val_path', type=str, default= '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/processed/val/',
                    help='val folder path')
    parser.add_argument('--test_path', type=str, default= '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/processed/test/',
                    help='test folder path')
    parser.add_argument('--wav_path', type=str, default= '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/raw/WSJ0/',
                    help='wav folder path - for original, non-mixed wav files')
    # parser.add_argument('--train_num', type=int, default= 30000,
    #                 help='Number of training samples')

    args = parser.parse_args()


   # convert_wv12wav(original_data_dir, wav_data_path, fs)
    createSample(args)
    #createSample(wav_data_path,processed_path, val_num)
    #createSample(wav_data_path,processed_path, test_num)

        
