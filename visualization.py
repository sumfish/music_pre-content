import numpy as np
import os 
import glob
import librosa
from utils import *

### settings
config = {
    ##### ------------ basic parameters----------
    'sr': 22050,
    '''
    'n_fft': 1024, # window size     #remy2048,512 #suli 2048,256
    'hop_length': 512,
    '''
    'n_fft': 2048, # window size     #remy2048,512 #suli 2048,256
    'hop_length': 256,
    # mels
    'n_mels': 128,

    #####---------- dataset-------------
    # change path for test/training data

    #'npy_path' : '../dataset/npy_txt/test/2s_have2track(16_17)',
    'npy_path' : '../dataset/npy_txt/test',
    'visual_trip_path' : '../dataset/npy_txt/train/2s_have2track(16_17)',  ##triplet set
    'tsne':'../plots/tsne/2s/no_connected_spec128/adap_weight/margin4',
}
#print(config)

def get_triplet_index(dataset):
    triplets=[]
    with open(os.path.join(config['npy_path']+dataset,'triplet.txt')) as f:
            for line in f:
                index=line.split(' ')
                triplets.append([int(index[0]),int(index[1]),int(index[2])])
    return triplets

def get_visual_point(triplets, seg, total_class):
    visual_index=[]   
    point=seg*total_class
    visual_index.append(triplets[point][0])  #anchor
    print(triplets[point][0])

    for count in range(total_class):
        visual_index.append(triplets[point+count][1]) #positive

    return visual_index

def turn2wave_pic(points,seg, distance, path, dataset):
    audio=load_npy(os.path.join(config['npy_path']+dataset,'audio.npy'))
    data=load_npy(os.path.join(config['npy_path']+dataset,'data.npy'))
    print("the same content points:{}".format(points))
    plt.figure(figsize=(6,4))
    
    #normalize ?
    #train_mean, train_std=compute_mean_std('/2s_512')
    
    for count in range(len(points)):
        # output audio
        outputname=os.path.join(path,'cut{:04d}.wav'.format(int(points[count])))
        print(outputname)
        librosa.output.write_wav(outputname,audio[int(points[count])],config['sr'])
        
        # draw spec

        # nor
        #data_n=(data[int(points[count])]-train_mean)/train_std
        #librosa.display.specshow(data_n, hop_length=config['hop_length'])
        librosa.display.specshow(data[int(points[count])], hop_length=config['hop_length'])
        plt.savefig(path+"/"+str(points[count])+'.png', dpi='figure', bbox_inches='tight')
        plt.clf()

#### 128 v.s. 512
def draw_mel(filename, mel):
    y, sr = librosa.load(filename)
    S= librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel)
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',
                        y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('../plots/mel_'+str(mel)+'.png', dpi='figure', bbox_inches='tight')

#####show the A,P,N set pictures
def visualize_triplet_data():  
    data=load_npy(os.path.join(config['visual_trip_path'],'data.npy'))
    
    triplets=[]
    with open(os.path.join(config['visual_trip_path'],'triplet.txt')) as f:
        for line in f:
            index=line.split(' ')
            triplets.append([int(index[0]),int(index[1]),int(index[2])])
    print('triplets length:{}'.format(len(triplets)))
    plt.figure(1,figsize=(3*4,7*4))
    plt.clf()
    print('Generating Pics......')

    num=20 ### change this index
    seg_index=num*7
    for i in range(7):
        print('Anchor index:{}'.format(triplets[seg_index+i][0]))
        for j in range(3):
            plt.subplot(7, 3, (i*3)+j+1)
            librosa.display.specshow(data[triplets[seg_index+i][j]], hop_length=config['hop_length'])
            #plt.title('instrument '+ str(j))
    plt.savefig('../plots/triplet_data/viz'+str(seg_index)+'.png', dpi='figure', bbox_inches='tight')
    plt.clf()

def visualize_testing_data(dataset, seg, distance, num_seg, total_class, path):
    main_path= os.path.join(path,str(seg)+'_'+str(distance))
    mkdir(main_path)
    triplets_list=get_triplet_index(dataset)
    
    for i in range(num_seg):
        path=os.path.join(main_path,str(seg))
        mkdir(path)
        positive_index=get_visual_point(triplets_list, seg, total_class)
        turn2wave_pic(positive_index, seg, distance, path, dataset)
        seg=seg+distance

def main():
    
    ### visualize testdataset
    num_seg = 8 ## visualize 8 segment
    total_class=4
    total_class=total_class-1 ## positive
    seg=400
    distance=10
    
    ### evaluation testn data
    #visualize_testing_data(seg, distance, num_seg, total_class, config['tsne'])
    
    ###visualiza triplet dataset 
    #visualize_triplet_data()

    '''
    ###visualize different mel-bank
    draw_mel('cut0210.wav',32)
    draw_mel('cut0210.wav',512)
    '''

if __name__ == '__main__':
    main()