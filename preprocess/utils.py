import os
import numpy as np
import librosa
import librosa.display
import json
import random
import matplotlib.pyplot as plt


def list2txt(output_file,out_list):
    with open(output_file,'w') as o:
        for label in out_list:
            #print(label)
            #o.write(str(label[0])+' '+str(label[1])+' '+str(label[2])+b'\n')
            o.write(label[0]+' '+label[1]+' '+label[2]+'\n')

def label2txt(output_file,out_list):
    with open(output_file,'w') as o:
        for label in out_list:
            o.write(label[0]+' '+label[1]+' '+label[2]+'\n')
            #o.write(label[0].decode()+' '+label[1].decode()+' '+label[2].decode()+'\n')

def mkdir(directory):
    if not os.path.exists(directory):
        print('making dir:{0}'.format(directory))
        os.makedirs(directory)
    else:
        print('already exist: {0}'.format(directory))

def load_npy(data_path):
    data=np.load(data_path)
    return data



