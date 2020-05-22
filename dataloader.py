import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch import Tensor
from utils import *
from utils import __RandomCropNumpy

npy_dir='../for_content_dataset_no4/npy_txt'

class Content_Dataset(Dataset):
    def __init__(self, path, audio_data, label_data, index_txt, length=128, transform=None):
        triplets=[]
        txt_file=os.path.join(path,index_txt)
        with open(os.path.join(npy_dir,txt_file)) as f:
            for line in f:
                index=line.split(' ')
                triplets.append([int(index[0]),int(index[1]),int(index[2])])
                #print([int(index[0]),int(index[1]),int(index[2])])
                #input()
        self.triplet=triplets
        self.transform = transform
        self.length= length
        self.data=load_npy(os.path.join(npy_dir,path,audio_data))
        self.label=load_npy(os.path.join(npy_dir,path,label_data))

    def __getitem__(self,index):
        
        #[-1, -1, -1] ------> blank, blank, any
        if int(self.triplet[index][0])==-1:
            audio1=np.zeros((512,128), dtype=np.float32)
            audio2=np.zeros((512,128), dtype=np.float32)

            # pick any seg
            rand_index=random.randint(0, len(self.triplet)-101)
            audio3=self.data[self.triplet[rand_index][2]]

            # make it to 128
            j = random.randint(0, audio1.shape[1] - self.length)
            audio3 = audio3[:, j:j+self.length]

        else:
            audio1=self.data[self.triplet[index][0]]
            audio2=self.data[self.triplet[index][1]]
            audio3=self.data[self.triplet[index][2]]

            ####class for visualization
            label1=self.label[self.triplet[index][0]][1] #[1]=> song instru seg_count
            label2=self.label[self.triplet[index][1]][1]
            label3=self.label[self.triplet[index][2]][1]
            #print(audio1.shape) #(128,Time) 

            # make it to 128
            j = random.randint(0, audio1.shape[1] - self.length)
            audio1 = audio1[:, j:j+self.length]
            audio2 = audio2[:, j:j+self.length]
            audio3 = audio3[:, j:j+self.length]

        if self.transform is not None:
            
            audio1=self.transform(audio1) #is already float data
            audio2=self.transform(audio2)
            audio3=self.transform(audio3)
            
        else:
            audio1=Tensor(audio1).view(1,audio1.shape[0],audio1.shape[1])
            audio2=Tensor(audio2).view(1,audio2.shape[0],audio2.shape[1])
            audio3=Tensor(audio3).view(1,audio3.shape[0],audio3.shape[1])
        #print(audio1.shape) # torch.Size([1, 128, 130])
        
        #return audio1,audio2,audio3,int(label1),int(label2),int(label3)  ####for visualization
        #return audio1,audio2,audio3,int(label1),int(label2),int(label3), self.triplet[index][0] ####for visualization2
        return audio1,audio2,audio3 ####for training

    def __len__(self):
        #print('%d'%len(self.triplet))
        return len(self.triplet)

'''
audio_transform=transforms.Compose([
        #transforms.Lambda(lambda x: __RandomCropNumpy(x, 128)),  ##cut 128 frames
        transforms.ToTensor(),
    #    transforms.Normalize(mean=(train_mean,),std=(train_std,))
    ])
# train & validation dataset
train_data=Content_Dataset('hung/2s_4class_512_no_normalize/train','data.npy', 'label.npy', 'triplet_blank.txt', transform = audio_transform).__getitem__(13525)
#k=Content_Dataset('test/2s_have2track(16_17)/','data.npy', 'label.npy', 'triplet.txt').__getitem__(30)
#print(k)
#print(len(k))
#print(Content_Dataset('train/2s_have2track(16_17)/','data.npy','triplet.txt').__getitem__(1))
#print(Content_Dataset('train/2s_have2track(16_17)/','data.npy','triplet.txt').__len__)
'''