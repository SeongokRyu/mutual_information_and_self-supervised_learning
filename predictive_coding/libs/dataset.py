import os
import glob
import tensorflow as tf
import soundfile as sf

def read_labels_txt(path):
    file_name = path+'SPEAKERS.TXT'
    os.system('grep \'train-clean-100\' ' + file_name + ' > tmp.txt')

    f = open('tmp.txt')
    lines = f.readlines()

    speaker_id = 0
    idx_gender_speaker = {}
    for l in lines:
        idx = l.split('|')[0].strip()
        gender =  0
        if l.split('|')[1].strip() == 'F':    
            gender = 1

        idx_gender_speaker[idx] = [gender, speaker_id]
        speaker_id += 1

    return idx_gender_speaker

def get_speaker_label(idx_gender_speaker, file_name, label=1):
    label_list = []
    file_idx = file_name.split('/')[-1].split('-')[0]
    return idx_gender_speaker[file_idx][label]

def read_file_list(path, train_or_test='train'):
    list_path = train_or_test+'_split.txt'

    f = open(path + list_path, 'r')
    lines = f.readlines()
    file_list = []
    for l in lines:
        file_list.append(l.strip())
    return file_list

def read_speech_file(path, file_name):
    idx = [file_name.split('-')[0], file_name.split('-')[1]]
    path += 'train-clean-100/'
    path += idx[0] + '/'
    path += idx[1] + '/'
    path += file_name + '.flac'

    data, _ = sf.read(path)
    return data

def read_splited_speech_file(path, train_or_test='train'):
    path += 'train-clean-100/'
    path += train_or_test + '_split/'

    file_list = glob.glob(path+'*.npy')
    return file_list
