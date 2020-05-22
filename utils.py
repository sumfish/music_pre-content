import os
import numpy as np
import librosa
import librosa.display
import json
import random
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


def list2txt(output_file,out_list):
    with open(output_file,'w') as o:
        for label in out_list:
            #print(label)
            o.write(label[0]+' '+label[1]+' '+label[2]+'\n')

def mkdir(directory):
    if not os.path.exists(directory):
        print('making dir:{0}'.format(directory))
        os.makedirs(directory)
    else:
        print('already exist: {0}'.format(directory))

def load_npy(data_path):
    data=np.load(data_path)
    return data

def __RandomCropNumpy(a, p, n, length):
    h, w = a.shape
    #print(x.shape) #512, 173
    j = random.randint(0, w - length)
    res_a = a[:, j:j+length]
    res_p = p[:, j:j+length]
    res_n = n[:, j:j+length]
    return res_a, res_p, res_n

def read_data_minmax(data_path):
    print('Data min/max:')
    data=load_npy('../dataset/npy_txt/train/'+data_path+'/data.npy')
    print(max(np.ravel(data)))
    print(min(np.ravel(data)))

def compute_mean_std(data_path):  ####train dataset
    data=load_npy('../dataset/npy_txt/train/'+data_path+'/data.npy')
    pixels=np.ravel(data)
    print('Data Mean:{}, Variance:{}'.format(np.mean(pixels),np.std(pixels)))
    return np.mean(pixels), np.std(pixels)

class Logger(object): #tensorboard --logdir=./log --bind_all
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)


