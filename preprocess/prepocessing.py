import numpy as np
import os 
import glob
import librosa
import pickle
from utils import mkdir, label2txt, load_npy
from tacotron.utils import get_spectrograms

### settings
config = {
    ##### ----------preprocessing way---------------
    'pre':'from hung-yi', ###else original
    'nor':'no',

    ##### ------------ basic parameters----------
    'sr': 22050,

    # window size     #remy2048,512 #suli 2048,256 #hung 2048,256
    'n_fft' :2048,
    'hop_length' :256,
    ##original
    #'n_fft': 1024, 
    #'hop_length': 512,
    
    # mels
    'n_mels': 512,

    # for slicing and overlapping (now is no overlapping)
    'audio_samples_frame_size': 44010, # sec * sr
    'audio_samples_hop_length': 44010,

    #####---------- dataset-------------
    # path/file
    # change path for test/training data
    'class':4,
    'dataset_path' : '../../for_content_dataset_no4/test',
    #'npy_path' : '../../for_style_dataset/npy_txt/test/hung/1s_4class_128(normalize)',
    'npy_path' : '../../for_content_dataset_no4/npy_txt/hung/2s_4class_512(no_normalize)/test',
    'attr_path' : '../../for_content_dataset_no4/npy_txt/hung/2s_4class_512(no_normalize)/train',
    'label_txt' : 'label.txt',
    'list_txt' : 'list.txt',
    'data_npy' : 'data.npy',
    'audio_npy' : 'audio.npy',
    'label_npy' : 'label.npy'
}
print(config)


def get_melspec(y, config):
    S = librosa.feature.melspectrogram(y, sr=config['sr'], n_fft=config['n_fft'], 
                                    hop_length=config['hop_length'], n_mels=config['n_mels'])
    S = np.log(1+10000*S) #filter
    #S = np.log(S)*20 
    #print(S.shape) #(128,T)
    #rint(min(np.ravel(S)))
    #log10000 14.248 0
    #s 154 0
    #20*log 100 0
    #input()
    return S

def audio2feature():
    data_list = []
    label_list = []
    audio_list = []
    
    # read file
    #files = glob.glob(config['dataset_path']+'/*/*.mp3')
    paths = glob.glob(config['dataset_path']+'/*')
    for p in paths:  ####song
        print('song_name:'+p)
        
        files = glob.glob(p+'/*.mp3')
        for filename in files:
            index1=filename.find('\\')+1 ####../dataset/train\01
            index2=filename.find('\\',index1)
            song_label=p[index1:index2]
            #print('song label:{}'.format(song_label))  ###song label
            print(filename)
            
            if filename.find('solo')!=-1:
                instru_class = filename[-11:-9] #class label
            else:
                instru_class= filename[-6:-4]
            if int(instru_class)>=config['class']:
                print('dump %s cause it is class %s'%(filename,instru_class))
                continue
            y, sr = librosa.core.load(filename, offset=0.0, sr=config['sr']) # read after 1 seconds #mono=True(convert signal to mono)
            #y, sr = librosa.load(filename, sr=config['sr'])

            Len = y.shape[0]
            count = 0   #segment count
            st_idx = 0 
            end_idx = st_idx + config['audio_samples_frame_size']
            next_idx = st_idx + config['audio_samples_hop_length']
            
            while  st_idx < Len:
                #### label
                label=[song_label,instru_class,count]
                label_list.append(label)
                if end_idx > Len:
                    end_idx = Len ####last is too short?
                    print(label) #[song,instrument,count]

                #### audio
                audio = np.zeros(config['audio_samples_frame_size'], dtype='float32')
                audio[:end_idx-st_idx] = y[st_idx:end_idx]

                if config['pre']=='from hung-yi':
                    feature, _ = get_spectrograms(audio)
                else:
                    feature = get_melspec(audio,config)
                data_list.append(feature)
                audio_list.append(audio)

                '''
                # output audio
                outputname=os.path.join('cut{:03d}.wav'.format(count))
                print(outputname)
                librosa.output.write_wav(outputname,audio,config['sr'])
                input()
                '''

                count +=1 
                st_idx = next_idx
                end_idx = st_idx + config['audio_samples_frame_size']
                next_idx = st_idx + config['audio_samples_hop_length']

    ###### normal
    if config['nor']=='yes':
        print('starting normalize')
        if(config['dataset_path' ][-4:]=='test'):
            with open(os.path.join(config['attr_path'], 'attr.pkl'), 'rb') as f:
                attr = pickle.load(f)
            mean = attr['mean']
            std = attr['std']
            print('mean, std:{},{}'.format(mean, std))
        else:
            data_temp = np.concatenate(data_list)
            mean = np.mean(data_temp, axis=0)
            std = np.std(data_temp, axis=0)
            '''
            data_temp shape:(48440, 128)
            mean shape:(128,)
            '''
            attr = {'mean': mean, 'std': std}
            with open(os.path.join(config['npy_path'], 'attr.pkl'), 'wb') as f:
                pickle.dump(attr, f)
        
        ##### normalize!!!!!!
        data_list=(data_list-mean)/std

    if config['pre']=='from hung-yi':
        print('hung-yi transpose')
        temp=[]
        for val in data_list:
            val=val.T #numpy T /// pytorch transpose
            temp.append(val)
        data_list=temp


    # save
    data_name=os.path.join(config['npy_path'],config['data_npy'])
    label_name=os.path.join(config['npy_path'],config['label_npy'])
    np.save(data_name,data_list)
    np.save(label_name,label_list)
    ### to wave
    audio_name=os.path.join(config['npy_path'],config['audio_npy'])
    np.save(audio_name,audio_list)

    #list2txt(config['label_txt'],label)
    #print(print(data[52]))
    #print(len(label))
    #return label_list

def main():
    
    mkdir(config['npy_path'])
    audio2feature()
    
    # for print out
    label=load_npy(os.path.join(config['npy_path'],'label.npy'))
    label2txt(os.path.join(config['npy_path'],'label.txt'),label)


if __name__ == '__main__':
    main()

