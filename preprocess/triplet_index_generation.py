import numpy as np
import random
from utils import *

def index_gene(data,label,class_num,same_class_test=False):
    assert len(data)==len(label),'length ok'
    print('Original length:{}'.format(len(data)))
    print('Generate triplet index file....')
    triplets=[]
    for i in range(len(label)): 
        ## poss
        #print(label[i]) #[song,instru,index]
        pos=[p for p in range(len(label)) if (label[p][0]==label[i][0] 
                                                and label[p][2]==label[i][2]
                                                and label[p][1]!=label[i][1])]
        #print(pos)

        if(len(pos)<class_num): ####for some song which is too long
            #print(label[i])
            continue

        ## negs not the same song
        if same_class_test:
            negs=[x for x in range(len(label)) if (label[x][0]!=label[i][0]
                                                        and label[x][1]==label[i][1])]
        else:
            negs=[x for x in range(len(label)) if label[x][0]!=label[i][0]]

        neg=np.random.choice(negs,class_num,replace=False)
        #print(negs)
        
        for index in range(class_num):
            triplets.append([str(i),str(pos[index]),str(neg[index])])
    print('triplets length:{}'.format(len(triplets)))
    return triplets

def add_blank_pairs(triplets):
    # blank, blank, any
    # train 100, test 10
    for i in range(100):
        triplets.append([str(-1),str(-1),str(-1)])
    return triplets


###### settings
same_class_test=False   ####special test(not use)
add_blank_pieces=True
class_num =4 ###for how many class is positive
#data_type='1s_have2track(16_17)'
data_type='hung/2s_4class_512(no_normalize)/test'
npy_path='../../for_content_dataset_no4/npy_txt/'
file_path= os.path.join(npy_path,data_type)

data=load_npy(file_path+'/data.npy')
label=load_npy(file_path+'/label.npy') #[song,instrument,count]

if same_class_test:
    print('Generating special(neg is the same instrument) for content test')
elif add_blank_pieces:
    print('Generating special dataset(add blank pairs)')
else:
    print('Generating original dataset')
print('path name:{}'.format(file_path))

triplets=index_gene(data,label,class_num-1,same_class_test)

if same_class_test:
    list2txt(file_path+'/triplet_s.txt',triplets)
elif add_blank_pieces:
    add_blank_pairs(triplets)
    list2txt(file_path+'/triplet_blank.txt',triplets)
else:
    list2txt(file_path+'/triplet.txt',triplets)

# for print out
label2txt(file_path+'/label.txt',label)
