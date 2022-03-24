from ast import operator
#from asyncio.windows_events import NULL
import librosa
import numpy as np
import numpy.random as random
import soundfile as sf
import os
fs = 16000

def convert_wv12wav(original_data_dir, wav_data_path, fs):
    # input - original path, intended wav path, sampling rate
    # output - void
    # function converts between WV1 file to in our original path to a WAV file in our intended path, file SR is fs 

    speakers_dirs = os.listdir(original_data_dir)

    for speaker in speakers_dirs:
        if not os.path.isdir(wav_data_path + speaker): 
            os.mkdir(wav_data_path + speaker)
        files = os.listdir(original_data_dir + '/' + speaker)
        files = [file for file in files if file.split('.')[-1] == 'wv1']
        for signal in files:
            wv1_signal, fs = librosa.load(original_data_dir + '/' + speaker + '/' +signal, fs)
            sf.write(wav_data_path + speaker + '/' + signal.split('.')[0] + '.wav', wv1_signal, fs)


def mix_signal(wav_data_path, p_silence, p_one):
    # input - wav files path, first speaker porbabilty, second speaker probabilty
    # output - mixed signal of speakers according to their defined probabilty
    # function uses given probabilty for each speaker to speak, mixes them to 5 sec intervals, and returns (meant to be used twice for our data)
    fs = 16000
    five_sec = 5*fs
    first_speaker = np.random.normal(0,1e-4,five_sec) 
    second_speaker = np.zeros(five_sec)
    speakers_dirs = os.listdir(wav_data_path)
    prob = np.random.rand()
    snr = 0
    g = 0
    if prob > p_silence :
        first_speaker_id = random.choice(speakers_dirs)
        first_file_id = random.choice(os.listdir(wav_data_path + first_speaker_id))
        first_speaker, fs = librosa.load(wav_data_path + first_speaker_id + '/' + first_file_id, fs)
        if len(first_speaker) > five_sec :
            rand_start = random.randint(0,len(first_speaker)-five_sec-1) 
            first_speaker = first_speaker[rand_start:rand_start+five_sec]
        else:
            first_speaker = np.concatenate((first_speaker,np.random.normal(0,1e-4,five_sec-len(first_speaker))))
        speakers_dirs.remove(first_speaker_id)
    else:
        first_speaker_id = 'no_first'
    if prob > p_one + p_silence: # probabilty for 2 speakers
        second_speaker_id = random.choice(speakers_dirs)
        second_file_id = random.choice(os.listdir(wav_data_path + second_speaker_id))
        snr = np.random.uniform(0,5)
        g = np.sqrt(10**(-snr/10) * np.std(first_speaker)**2 / np.std(second_speaker)**2)
        second_speaker, fs = librosa.load(wav_data_path + second_speaker_id + '/' + second_file_id, fs)
        if len(second_speaker) > five_sec :
            rand_start = random.randint(0,len(second_speaker)-five_sec-1) #clips bigger then 10 sec
            second_speaker = second_speaker[rand_start:rand_start+five_sec]
        else:
            second_speaker = np.concatenate((second_speaker,np.random.normal(0,1e-4,five_sec-len(second_speaker))))
        speakers_dirs.remove(second_speaker_id)
    else:
        second_speaker_id = 'no_second'
    mixed_signal = first_speaker + g * second_speaker
    return mixed_signal, first_file_id.split[0] + '-' + second_file_id.split[0] + '-' + str(snr)

def CreateSample(wav_data_path, processed_path):
    fs = 16000
    for [] in range(5):
        mix1, mix1_id = mix_signal(wav_data_path, 0, 0.5)
        mix2, mix2_id = mix_signal(wav_data_path, 0.55, 0.15)
        final_sample = np.concatenate((mix1, mix2))
        sf.write(processed_path + mix1_id + '_' + mix2_id + '.wav' , final_sample, fs)

if __name__ == "__main__":
    wav_data_path = '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/raw/WSJ0/'
    original_data_dir = '/mnt/dsi_vol1/users/shustea1/data/sd_et_20'
    processed_path = '/mnt/dsi_vol1/users/shustea1/Data/lstm-speech/data/processed/WSJ0/'
    fs = 16000

   # convert_wv12wav(original_data_dir, wav_data_path, fs)
    CreateSample(wav_data_path,processed_path)